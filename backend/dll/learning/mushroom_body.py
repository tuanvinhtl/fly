"""Dopamine-gated plasticity at Kenyon cell -> MBON synapses (associative olfactory learning).

Biology this follows (well established in Drosophila):
  * odours are represented by sparse sets of Kenyon cells (KCs);
  * KCs synapse onto mushroom body output neurons (MBONs) in compartments, each
    innervated by its own dopaminergic neurons (DANs);
  * dopamine coincident with KC activity depresses the active KC->MBON synapses in that
    compartment (Hige et al. 2015; Cohn et al. 2015; Aso & Rubin 2016).
  * PAM DANs signal reward such as sugar (Liu et al. 2012; Burke et al. 2012);
    PPL1 DANs signal punishment such as bitter taste or heat (Das et al. 2014).

What is ours (EXPERIMENTAL / APPROXIMATION):
  * Compartment membership comes from the connectome: the dopamine "seen" by MBON j is
    the spike rate of DANs averaged with weights = their synapse counts onto MBON j.
  * Reinforcement injection: in the connectome model, sugar or bitter taste does NOT
    drive PAM / PPL1 (measured: 0 / 307 PAM active under strong sugar input), so during
    taste we add Poisson drive to PAM (reward) or PPL1 (punishment) proportional to the
    sensory rate. This stands in for pathways missing from the model.
  * Rule:  dw_kj = -eta * e_k * D_j * dt   (w = multiplicative scale, floored at w_min),
    with e_k an exponentially filtered KC spike count (tau_elig) and D_j in Hz.
    Optional passive forgetting: weights relax toward 1 with time constant tau_forget.

Nothing here touches behaviour directly: learned changes act only by altering MBON firing,
which reaches descending neurons through whatever wiring the connectome provides.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..brain.connectome import Connectome
from ..brain.lif import LIFBrain

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MBParams:
    eta: float = 0.004  # per (KC spike-trace unit x DA Hz x s)
    w_min: float = 0.05
    tau_elig: float = 1.0  # s
    tau_forget: float = 0.0  # s; 0 = no forgetting
    reward_hz: float = 60.0  # PAM drive at full sugar rate
    punish_hz: float = 60.0  # PPL1 drive at full bitter rate
    taste_ref_hz: float = 100.0  # GRN rate considered "full" taste


class MushroomBodyPlasticity:
    def __init__(self, connectome: Connectome, params: MBParams | None = None):
        self.p = params or MBParams()
        ann = connectome.annotations
        cls = np.array(ann["cell_class"])
        ctype = np.array(ann["cell_type"].astype(str))
        kc, mbon, dan = cls == "Kenyon_Cell", cls == "MBON", cls == "DAN"
        self.pam = np.flatnonzero(dan & np.char.startswith(ctype.astype("U"), "PAM"))
        self.ppl1 = np.flatnonzero(dan & np.char.startswith(ctype.astype("U"), "PPL1"))
        self.dan = np.flatnonzero(dan)
        self.kc = np.flatnonzero(kc)
        self.mbon = np.flatnonzero(mbon)
        n = connectome.n_neurons
        pre = np.repeat(np.arange(n), np.diff(connectome.indptr))
        post = connectome.indices
        # plastic connections
        self.conn = np.flatnonzero(kc[pre] & mbon[post])
        self.conn_kc = pre[self.conn]
        mbon_pos = np.full(n, -1)
        mbon_pos[self.mbon] = np.arange(self.mbon.size)
        self.conn_mbon = mbon_pos[post[self.conn]]
        # DAN -> MBON compartment matrix (rows: MBON, cols: DAN), synapse-count weights
        dm = np.flatnonzero(dan[pre] & mbon[post])
        dan_pos = np.full(n, -1)
        dan_pos[self.dan] = np.arange(self.dan.size)
        self.dm_mbon = mbon_pos[post[dm]]
        self.dm_dan = dan_pos[pre[dm]]
        self.dm_w = np.abs(connectome.counts[dm]).astype(np.float64)
        norm = np.bincount(self.dm_mbon, weights=self.dm_w, minlength=self.mbon.size)
        self.dm_w /= np.where(norm[self.dm_mbon] > 0, norm[self.dm_mbon], 1.0)
        self.elig = np.zeros(n)
        self.scale = np.ones(self.conn.size)
        self.dopamine = np.zeros(self.mbon.size)  # Hz per MBON compartment (last tick)
        self.n_updates = 0
        self._forget_acc = 0.0
        log.info(
            "MB plasticity: %d KC->MBON connections, %d MBONs, %d PAM, %d PPL1",
            self.conn.size, self.mbon.size, self.pam.size, self.ppl1.size,
        )

    # ---------------------------------------------------------------- reinforcement
    def reinforcement_inputs(self, sugar_hz: float, bitter_hz: float) -> list[tuple[np.ndarray, float]]:
        """Poisson drive on PAM (reward) / PPL1 (punishment) DANs for this tick."""
        p = self.p
        out = []
        if sugar_hz > 0:
            out.append((self.pam, p.reward_hz * min(1.0, sugar_hz / p.taste_ref_hz)))
        if bitter_hz > 0:
            out.append((self.ppl1, p.punish_hz * min(1.0, bitter_hz / p.taste_ref_hz)))
        return out

    # ---------------------------------------------------------------- plasticity
    def update(self, brain: LIFBrain, spike_idx: np.ndarray, spike_cnt: np.ndarray, dt: float) -> int:
        """Advance traces with this tick's spikes and apply the learning rule.
        Returns the number of synapses changed."""
        p = self.p
        self.elig[self.kc] *= np.exp(-dt / p.tau_elig)
        self.elig[spike_idx] += spike_cnt  # only KC entries are read below
        dan_rate = np.zeros(self.dan.size)
        pos = np.searchsorted(self.dan, spike_idx)
        hit = (pos < self.dan.size) & (self.dan[np.minimum(pos, self.dan.size - 1)] == spike_idx)
        dan_rate[pos[hit]] = spike_cnt[hit] / dt
        self.dopamine = np.bincount(self.dm_mbon, weights=self.dm_w * dan_rate[self.dm_dan], minlength=self.mbon.size)
        changed = np.zeros(self.conn.size, dtype=bool)
        if self.dopamine.any():
            e = self.elig[self.conn_kc]
            d = self.dopamine[self.conn_mbon]
            dw = p.eta * e * d * dt
            changed = dw > 0
            self.scale[changed] = np.maximum(p.w_min, self.scale[changed] - dw[changed])
        if changed.any():
            brain.scale_connections(self.conn[changed], self.scale[changed])
            self.n_updates += 1
        return int(changed.sum())

    def forget(self, brain: LIFBrain, dt: float, every: float = 1.0) -> int:
        """Passive forgetting: depressed weights relax toward 1 with time constant
        tau_forget (exact exponential, applied in batches of `every` seconds). Runs whether
        or not learning is switched on."""
        if self.p.tau_forget <= 0:
            return 0
        self._forget_acc += dt
        if self._forget_acc < every:
            return 0
        k = float(np.exp(-self._forget_acc / self.p.tau_forget))
        self._forget_acc = 0.0
        moved = self.scale < 1.0
        if not moved.any():
            return 0
        self.scale[moved] = 1.0 - (1.0 - self.scale[moved]) * k
        brain.scale_connections(self.conn[moved], self.scale[moved])
        return int(moved.sum())

    def apply_to(self, brain: LIFBrain) -> None:
        """Re-apply learned weights (e.g. after brain.reset())."""
        brain.scale_connections(self.conn, self.scale)

    def reset(self, keep_memory: bool = False) -> None:
        self.elig[:] = 0
        self.dopamine[:] = 0
        self._forget_acc = 0.0
        if not keep_memory:
            self.scale[:] = 1.0
            self.n_updates = 0

    def summary(self) -> dict:
        s = self.scale
        return {
            "plastic_synapses": int(s.size),
            "depressed": int((s < 0.99).sum()),
            "mean_weight": round(float(s.mean()), 4),
            "min_weight": round(float(s.min()), 4),
            "updates": self.n_updates,
        }
