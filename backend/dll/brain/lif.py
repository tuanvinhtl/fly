"""Leaky integrate-and-fire whole-brain model (NumPy implementation).

This is a faithful port of the Brian2 model published with Shiu et al. 2024,
"A Drosophila computational brain model reveals sensorimotor processing"
(Nature 634, 210–219; code: github.com/philshiu/Drosophila_brain_model, MIT):

    dv/dt = (v_0 - v + g) / t_mbr      (unless refractory)
    dg/dt = -g / tau                   (unless refractory)
    spike:  v > v_th   ->  v = v_rst, g = 0, refractory for t_rfc
    synapse: g_post += w_syn * signed_synapse_count, after delay t_dly
             (dropped if the target is refractory — Brian2 semantics, verified in tests)
    external drive: Poisson events add w_syn * f_poi directly to v

Both ODEs are linear, so each step is integrated exactly (Brian2 `method='linear'`).
Per-step operation order mirrors Brian2's default schedule:
    state update -> threshold -> synaptic delivery / Poisson input -> reset.

Why not Brian2 at runtime?  The closed-loop simulator needs to advance the brain in
small chunks interleaved with physics and to change inputs every chunk; Brian2's
`Network.run` start-up cost per call makes this impractical on CPU. Brian2 is kept
as a reference backend (see brian2_ref.py) and the tests compare the two.

Engines:
  numba (default) — exact lazy integration (see _lif_kernel.py): neurons whose analytic
      upper bound on future membrane potential stays below threshold are advanced in
      closed form only when they receive input; only near-threshold neurons are stepped.
  numpy — dense step-by-step reference. Optional `rest_tol` snaps neurons within that
      distance of rest (off by default). Tests check the engines agree.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .connectome import Connectome

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LIFParams:
    """Parameters from Shiu et al. 2024 default_params (units: s, V, Hz)."""

    v_0: float = -52e-3  # resting potential (Kakaria & de Bivort 2017)
    v_rst: float = -52e-3  # reset potential
    v_th: float = -45e-3  # spike threshold
    t_mbr: float = 20e-3  # membrane time constant
    tau: float = 5e-3  # synaptic time constant (Jürgensen et al. 2021)
    t_rfc: float = 2.2e-3  # refractory period (Lazar et al. 2021)
    t_dly: float = 1.8e-3  # synaptic delay (Paul et al. 2015)
    w_syn: float = 0.275e-3  # weight per synapse — FREE PARAMETER fit by Shiu et al.
    f_poi: float = 250.0  # Poisson input scaling (one event forces a spike)
    dt: float = 1e-4  # Brian2 default clock
    rest_tol: float = 0.0  # NumPy reference engine only: optional rest-snapping tolerance (V)
    # --- EXTENSION (not in Shiu et al.): short-term synaptic depression -------------
    # Tsodyks-Markram resource model per presynaptic neuron. std_U = 0 disables it and
    # recovers the published model exactly. See RESEARCH.md §4 for why it is needed.
    std_U: float = 0.0  # fraction of resources used per spike
    std_tau_rec: float = 0.5  # recovery time constant (s)


@dataclass
class LIFState:
    v: np.ndarray
    g: np.ndarray
    ref_until: np.ndarray  # first step at which the neuron integrates and accepts input again
    buffer: np.ndarray  # (delay_steps, N) pending synaptic input
    step: int
    rng_state: dict
    weight_scale: np.ndarray | None  # per-connection scale; None when all ones (saves 120 MB)
    std_x: np.ndarray
    std_last: np.ndarray
    spike_count: np.ndarray


@dataclass
class StepResult:
    spikes: list[np.ndarray] = field(default_factory=list)  # spike indices per step (record=True)
    flat: np.ndarray | None = None  # all spike indices of the chunk (None if the buffer overflowed)


class LIFBrain:
    """Whole-brain LIF network over a Connectome."""

    def __init__(
        self,
        connectome: Connectome,
        params: LIFParams | None = None,
        seed: int = 0,
        engine: str = "numba",
    ):
        if engine not in ("numba", "numpy"):
            raise ValueError(f"unknown engine {engine!r}")
        self.engine = engine
        self.c = connectome
        self.p = params or LIFParams()
        p = self.p
        self.n = connectome.n_neurons
        self.delay_steps = max(1, int(round(p.t_dly / p.dt)))
        self.rfc_steps = int(round(p.t_rfc / p.dt))
        self.decay_v = float(np.exp(-p.dt / p.t_mbr))
        self.decay_g = float(np.exp(-p.dt / p.tau))
        # contribution of g to v over one step (exact solution of the linear system)
        self.g_to_v = float(p.tau / (p.tau - p.t_mbr) * (self.decay_g - self.decay_v))
        self.base_w = connectome.counts.astype(np.float64) * p.w_syn
        self.poisson_w = p.w_syn * p.f_poi
        # closed-form propagators for the lazy (event-driven) numba kernel
        d = np.arange(20_000, dtype=np.float64) * p.dt
        self.gv_coef = p.tau / (p.tau - p.t_mbr)
        self.dv_tab = np.exp(-d / p.t_mbr)
        self.dg_tab = np.exp(-d / p.tau)
        self.gv_tab = self.gv_coef * (self.dg_tab - self.dv_tab)
        a_, b_ = 1.0 / p.t_mbr, 1.0 / p.tau
        s_peak = np.log(b_ / a_) / (b_ - a_)
        self.m_peak = float(np.exp(-a_ * s_peak) - np.exp(-b_ * s_peak))  # max of e^-as - e^-bs
        self.c_coef = p.tau / (p.t_mbr - p.tau)
        self.seed = seed
        self.input_idx = np.zeros(0, dtype=np.int64)
        self.input_rate = np.zeros(0, dtype=np.float64)
        self.silenced = np.zeros(self.n, dtype=bool)
        self.std_u = np.full(self.n, p.std_U)  # per-neuron depression (0 = none)
        self.reset(seed)
        log.info(
            "LIFBrain: N=%d, E=%d, dt=%.2gms, delay=%d steps, active-set=%s, STD U=%.3g",
            self.n,
            connectome.n_connections,
            p.dt * 1e3,
            self.delay_steps,
            p.rest_tol > 0,
            p.std_U,
        )

    # ------------------------------------------------------------------ state
    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        p = self.p
        self.rng = np.random.default_rng(self.seed)
        self.v = np.full(self.n, p.v_0)
        self.g = np.zeros(self.n)
        self.ref_until = np.zeros(self.n, dtype=np.int64)
        self.buffer = np.zeros((self.delay_steps, self.n))
        self.t_step = 0
        self.active = np.zeros(self.n, dtype=bool)
        self.weight_scale = np.ones(self.c.n_connections)
        self._w = self.base_w.copy()
        self.spike_count = np.zeros(self.n, dtype=np.int64)
        self.std_x = np.ones(self.n)
        self.std_last = np.full(self.n, -(10**9), dtype=np.int64)
        self._touched_dirty = True
        self._apply_silencing()

    def snapshot(self) -> LIFState:
        return LIFState(
            v=self.v.copy(),
            g=self.g.copy(),
            ref_until=self.ref_until.copy(),
            buffer=self.buffer.copy(),
            step=self.t_step,
            rng_state=self.rng.bit_generator.state,
            weight_scale=None if np.all(self.weight_scale == 1.0) else self.weight_scale.copy(),
            std_x=self.std_x.copy(),
            std_last=self.std_last.copy(),
            spike_count=self.spike_count.copy(),
        )

    def restore(self, s: LIFState) -> None:
        self.v, self.g = s.v.copy(), s.g.copy()
        self.ref_until = s.ref_until.copy()
        self.buffer = s.buffer.copy()
        self.t_step = s.step
        self.rng.bit_generator.state = s.rng_state
        self.set_weight_scale(np.ones(self.c.n_connections) if s.weight_scale is None else s.weight_scale)
        self.std_x, self.std_last = s.std_x.copy(), s.std_last.copy()
        self.spike_count = s.spike_count.copy()
        self.active = (np.abs(self.v - self.p.v_0) > 0) | (self.g != 0) | (self.buffer != 0).any(axis=0)
        self._touched_dirty = True

    @property
    def time(self) -> float:
        return self.t_step * self.p.dt

    # ------------------------------------------------------------------ inputs
    def set_input_rates(self, idx: np.ndarray, rates_hz: np.ndarray) -> None:
        """Replace all external Poisson drives. Neurons with rate 0 are dropped."""
        idx = np.asarray(idx, dtype=np.int64)
        rates = np.asarray(rates_hz, dtype=np.float64)
        if idx.shape != rates.shape:
            raise ValueError("idx and rates must have identical shape")
        if np.any(rates < 0):
            raise ValueError("rates must be non-negative")
        keep = rates > 0
        self.input_idx, self.input_rate = idx[keep], rates[keep]

    def set_depression(self, u: np.ndarray) -> None:
        """Per-neuron utilisation U for short-term depression (0 exempts a neuron)."""
        if u.shape != (self.n,) or np.any((u < 0) | (u >= 1)):
            raise ValueError("depression U must be shape (N,) within [0, 1)")
        self.std_u = u.astype(np.float64).copy()

    def silence(self, idx: np.ndarray) -> None:
        """Silence neurons by zeroing their outgoing weights (as in Shiu et al.)."""
        self.silenced[np.asarray(idx, dtype=np.int64)] = True
        self._apply_silencing()

    def unsilence_all(self) -> None:
        self.silenced[:] = False
        self._apply_silencing()

    def set_weight_scale(self, scale: np.ndarray) -> None:
        if scale.shape != self.weight_scale.shape:
            raise ValueError("weight scale must have one entry per connection")
        self.weight_scale = scale.copy()
        self._apply_silencing()

    def _apply_silencing(self) -> None:
        self._w = self.base_w * self.weight_scale
        if self.silenced.any():
            pre_silenced = np.repeat(self.silenced, np.diff(self.c.indptr))
            self._w[pre_silenced] = 0.0

    # ------------------------------------------------------------------ dynamics
    def step(self, n_steps: int = 1, record: bool = False) -> StepResult:
        """Advance n_steps. Poisson events for the whole chunk are drawn up-front so that
        both engines consume the RNG identically (determinism across engines)."""
        lam = self.input_rate * self.p.dt
        events = self.rng.poisson(lam, size=(n_steps, lam.size)).astype(np.int32) if lam.size else None
        if self.engine == "numba":
            return self._step_numba(n_steps, events, record)
        return self._step_numpy(n_steps, events, record)

    def _step_numpy(self, n_steps: int, events: np.ndarray | None, record: bool) -> StepResult:
        p = self.p
        res = StepResult()
        indptr, indices = self.c.indptr, self.c.indices
        use_active = p.rest_tol > 0
        is_input = np.zeros(self.n, dtype=bool)
        is_input[self.input_idx] = True
        self._touched_dirty = True
        for k in range(n_steps):
            t = self.t_step
            slot = t % self.delay_steps
            # 1. state update for non-refractory neurons
            if use_active:
                a = np.flatnonzero(self.active)
                a = a[self.ref_until[a] <= t]
            else:
                a = np.flatnonzero(self.ref_until <= t)
            u = self.v[a] - p.v_0
            ga = self.g[a]
            self.v[a] = p.v_0 + u * self.decay_v + ga * self.g_to_v
            self.g[a] = ga * self.decay_g
            # 2. threshold (only non-refractory neurons can cross)
            spk = a[self.v[a] > p.v_th]
            # 3a. delayed synaptic input
            #     Brian2 discards increments to `(unless refractory)` variables while a
            #     neuron is refractory, so arrivals at refractory neurons are dropped.
            arrivals = self.buffer[slot]
            hit = np.flatnonzero(arrivals)
            hit = hit[self.ref_until[hit] <= t]
            self.g[hit] += arrivals[hit]
            self.active[hit] = True
            arrivals[:] = 0.0
            # 3b. Poisson external drive acts directly on v
            if events is not None:
                ev = events[k]
                tgt = self.input_idx[ev > 0]
                self.v[tgt] += ev[ev > 0] * self.poisson_w
                self.active[tgt] = True
            # 3c. schedule outgoing synaptic input from this step's spikes
            if spk.size:
                starts, ends = indptr[spk], indptr[spk + 1]
                lens = ends - starts
                eff = np.ones(spk.size)
                if p.std_U > 0:
                    rec = np.exp(-(t - self.std_last[spk]) * p.dt / p.std_tau_rec)
                    eff = 1.0 - (1.0 - self.std_x[spk]) * rec
                    self.std_x[spk] = eff * (1.0 - self.std_u[spk])
                    self.std_last[spk] = t
                if lens.sum():
                    offs = np.repeat(starts - np.cumsum(lens) + lens, lens) + np.arange(lens.sum())
                    wts = self._w[offs] * np.repeat(eff, lens)
                    self.buffer[slot] += np.bincount(indices[offs], weights=wts, minlength=self.n)
                # 4. reset
                self.v[spk] = p.v_rst
                self.g[spk] = 0.0
                self.ref_until[spk] = t + np.where(is_input[spk], 0, self.rfc_steps)
                self.spike_count[spk] += 1
            if use_active:
                quiet = a[(np.abs(self.v[a] - p.v_0) < p.rest_tol) & (np.abs(self.g[a]) < p.rest_tol)]
                self.v[quiet] = p.v_0
                self.g[quiet] = 0.0
                self.active[quiet] = False
                self.active[spk] = True
            res.spikes.append(spk)
            self.t_step += 1
        res.flat = np.concatenate(res.spikes) if res.spikes else np.zeros(0, dtype=np.int64)
        if not record:
            res.spikes = []
        return res

    def _rebuild_touched(self) -> None:
        d, n = self.buffer.shape
        self._bt_flag = self.buffer != 0
        self._bt_idx = np.zeros((d, n), dtype=np.int32)
        self._bt_n = np.zeros(d, dtype=np.int64)
        for s_ in range(d):
            nz = np.flatnonzero(self._bt_flag[s_])
            self._bt_idx[s_, : nz.size] = nz
            self._bt_n[s_] = nz.size
        self._touched_dirty = False

    def _step_numba(self, n_steps: int, events: np.ndarray | None, record: bool) -> StepResult:
        from ._lif_kernel import lif_steps

        p = self.p
        if self._touched_dirty:
            self._rebuild_touched()
        is_input = np.zeros(self.n, dtype=np.bool_)
        is_input[self.input_idx] = True
        if events is None:
            events = np.zeros((n_steps, 0), dtype=np.int32)
        spk_out = np.empty(max(65536, n_steps * 1024), dtype=np.int32)
        spk_ptr = np.zeros(n_steps + 1, dtype=np.int64)
        truncated = lif_steps(
            n_steps, self.t_step, self.delay_steps, self.rfc_steps,
            p.v_0, p.v_rst, p.v_th, self.decay_v, self.decay_g, self.g_to_v, self.poisson_w,
            self.std_u, p.dt / p.std_tau_rec,
            self.dv_tab, self.dg_tab, self.gv_tab, p.dt / p.t_mbr, p.dt / p.tau, self.gv_coef, self.c_coef, self.m_peak,
            self.c.indptr, self.c.indices, self._w,
            self.input_idx, events, is_input,
            self.v, self.g, self.ref_until, self.buffer,
            self._bt_idx, self._bt_n, self._bt_flag, self.active, self.spike_count,
            self.std_x, self.std_last,
            spk_out, spk_ptr,
        )
        if truncated and record:
            log.warning("spike record buffer full: per-step spike lists truncated (counts are exact)")
        self.t_step += n_steps
        res = StepResult(flat=None if truncated else spk_out[: spk_ptr[n_steps]].astype(np.int64))
        if record:
            res.spikes = [spk_out[spk_ptr[k]:spk_ptr[k + 1]].astype(np.int64) for k in range(n_steps)]
        return res

    def run(self, duration_s: float, record: bool = False) -> StepResult:
        return self.step(int(round(duration_s / self.p.dt)), record=record)

    @property
    def dt(self) -> float:
        return self.p.dt
