"""Reference implementation using Brian2, mirroring Shiu et al.'s `create_model`/`poi`.

Used only for validation (tests/test_brian2_reference.py): too slow to call in 10 ms
chunks inside the closed loop.
"""

from __future__ import annotations

from textwrap import dedent

import numpy as np

from .connectome import Connectome
from .lif import LIFParams


def run_brian2(
    connectome: Connectome,
    input_idx: np.ndarray,
    rate_hz: float,
    duration_s: float,
    seed: int,
    params: LIFParams | None = None,
) -> np.ndarray:
    """Return per-neuron spike counts from a Brian2 run of the published model."""
    import brian2 as b2

    p = params or LIFParams()
    b2.start_scope()
    b2.prefs.codegen.target = "numpy"
    b2.seed(seed)
    b2.defaultclock.dt = p.dt * b2.second
    ns = {
        "v_0": p.v_0 * b2.volt, "v_rst": p.v_rst * b2.volt, "v_th": p.v_th * b2.volt,
        "t_mbr": p.t_mbr * b2.second, "tau": p.tau * b2.second,
    }
    eqs = dedent("""
        dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
        dg/dt = -g / tau               : volt (unless refractory)
        rfc                            : second
    """)
    neu = b2.NeuronGroup(
        connectome.n_neurons, eqs, method="linear", threshold="v > v_th",
        reset="v = v_rst; g = 0 * mV", refractory="rfc", namespace=ns,
    )
    neu.v = p.v_0 * b2.volt
    neu.g = 0 * b2.volt
    neu.rfc = p.t_rfc * b2.second
    pre = np.repeat(np.arange(connectome.n_neurons), np.diff(connectome.indptr))
    syn = b2.Synapses(neu, neu, "w : volt", on_pre="g += w", delay=p.t_dly * b2.second)
    syn.connect(i=pre, j=connectome.indices.astype(np.int64))
    syn.w = connectome.counts.astype(float) * p.w_syn * b2.volt
    pois = []
    for i in input_idx:
        pois.append(b2.PoissonInput(neu[int(i):int(i) + 1], "v", N=1, rate=rate_hz * b2.Hz, weight=p.w_syn * p.f_poi * b2.volt))
        neu.rfc[int(i)] = 0 * b2.second
    mon = b2.SpikeMonitor(neu)
    net = b2.Network(neu, syn, mon, *pois)
    net.run(duration_s * b2.second)
    return np.bincount(np.asarray(mon.i), minlength=connectome.n_neurons)
