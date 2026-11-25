"""EXPERIMENTAL: reward-gated plasticity at Kenyon cell -> MBON synapses.

Biological motivation: dopaminergic neurons (DANs) induce long-term depression of
KC->MBON synapses for KCs active during reinforcement (Hige et al. 2015; Aso & Rubin
2016). Implementation here (a hypothesis, not validated in this model):

  * eligibility e_k: exponentially filtered spike count of each KC (tau = 2 s)
  * on a reward event (sugar ingestion), every KC->MBON connection from KC k is scaled
    by (1 - eta * e_k / max(e)), floored at w_min.

The baseline connectome contains no plasticity; this module edits the brain's
per-connection weight scale. Off by default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..brain.connectome import Connectome
from ..brain.lif import LIFBrain

log = logging.getLogger(__name__)


@dataclass
class KCMBONPlasticity:
    eta: float = 0.05
    w_min: float = 0.2
    tau_elig: float = 2.0

    def __init__(self, connectome: Connectome, eta: float = 0.05, w_min: float = 0.2, tau_elig: float = 2.0):
        self.eta, self.w_min, self.tau_elig = eta, w_min, tau_elig
        ann = connectome.annotations
        kc = (ann["cell_class"] == "Kenyon_Cell").to_numpy()
        mbon = (ann["cell_class"] == "MBON").to_numpy()
        pre = np.repeat(np.arange(connectome.n_neurons), np.diff(connectome.indptr))
        self.conn = np.flatnonzero(kc[pre] & mbon[connectome.indices])
        self.conn_pre = pre[self.conn]
        self.kc_idx = np.flatnonzero(kc)
        self.elig = np.zeros(connectome.n_neurons)
        self._last_counts = np.zeros(connectome.n_neurons, dtype=np.int64)
        self.n_updates = 0
        log.info("KC->MBON plasticity: %d connections", self.conn.size)

    def observe(self, brain: LIFBrain, dt: float) -> None:
        new = brain.spike_count - self._last_counts
        self._last_counts = brain.spike_count.copy()
        self.elig *= np.exp(-dt / self.tau_elig)
        self.elig[self.kc_idx] += new[self.kc_idx]

    def reward(self, brain: LIFBrain) -> int:
        e = self.elig[self.conn_pre]
        if e.max() <= 0:
            return 0
        scale = brain.weight_scale.copy()
        scale[self.conn] = np.maximum(self.w_min, scale[self.conn] * (1 - self.eta * e / e.max()))
        brain.set_weight_scale(scale)
        self.n_updates += 1
        return int((e > 0).sum())

    def reset(self) -> None:
        self.elig[:] = 0
        self._last_counts[:] = 0
        self.n_updates = 0
