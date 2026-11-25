"""Mushroom body valence readout: MBON activity -> odour-guided steering (APPROXIMATION).

Why this exists: in the connectome model, learned changes at KC->MBON synapses do not
propagate to descending neurons (RESEARCH.md §6b). This module supplies the missing
MBON -> premotor link explicitly, using population-level findings:

  * MBON valence follows transmitter as a trend (Aso et al. 2014b, eLife 3:e04580):
    glutamatergic MBONs mostly promote avoidance; cholinergic and GABAergic MBONs mostly
    promote approach. (There are exceptions; this is a coarse rule.)
  * In naive flies the two groups are balanced; learning breaks the balance (reward
    depresses avoidance-MBON drive -> net approach).
  * Flies steer toward/away from odour using bilateral antennal comparison
    (Duistermars et al. 2009; Gaudry et al. 2013).

Readout:
    V = k_app * mean_rate(approach MBONs) - k_av * mean_rate(avoidance MBONs)
    k_app, k_av calibrated so that naive responses to a panel of odours average to
    V = 0 (stored per preset in data/cache).
    turn_bias = gain * tanh(V / v_ref) * clip(lateral_odour_contrast * contrast_gain, -1, 1)
    added to the VNC's raw steering drive before its tanh saturation (body/vnc.py).

Only the steering command is affected, and only while odour is sensed.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..brain.connectome import Connectome

log = logging.getLogger(__name__)

# odour panel used for calibration: single and paired glomeruli across the antennal lobe
CALIBRATION_ODOURS = [
    ["ORN_DM1", "ORN_DM2", "ORN_DM4", "ORN_DP1m", "ORN_VA2"],
    ["ORN_DL1", "ORN_VA3", "ORN_DL5", "ORN_VM2", "ORN_VC3", "ORN_DA3"],
    ["ORN_VA1d", "ORN_VA1v", "ORN_DA1"],
    ["ORN_DL3", "ORN_VL1", "ORN_VM4"],
    ["ORN_DM3", "ORN_DM5", "ORN_DM6"],
    ["ORN_VA6", "ORN_VC1", "ORN_VM7d", "ORN_VM5d"],
]


@dataclass(frozen=True)
class ValenceParams:
    tau: float = 0.3  # s, smoothing of the valence signal
    v_ref: float = 0.15  # valence (normalised units) giving ~76% of max steering
    gain: float = 3.0  # max steering added, in raw (pre-tanh) VNC turn units
    contrast_gain: float = 300.0  # left/right relative odour difference -> lateral cue (antennae ~0.2 mm apart)
    odour_floor: float = 0.02  # concentration below which no odour is sensed


class ValenceReadout:
    def __init__(self, connectome: Connectome, k_app: float = 1.0, k_av: float = 1.0, params: ValenceParams | None = None):
        self.p = params or ValenceParams()
        ann = connectome.annotations
        mbon = np.array(ann["cell_class"] == "MBON")
        nt = np.array(ann["top_nt"])
        self.approach = np.flatnonzero(mbon & np.isin(nt, ["acetylcholine", "gaba"]))
        self.avoid = np.flatnonzero(mbon & (nt == "glutamate"))
        self.k_app, self.k_av = k_app, k_av
        self.value = 0.0
        self.last_bias = 0.0
        self._is_app = np.zeros(connectome.n_neurons, dtype=bool)
        self._is_app[self.approach] = True
        self._is_av = np.zeros(connectome.n_neurons, dtype=bool)
        self._is_av[self.avoid] = True

    # ------------------------------------------------------------------ signal
    def group_rates(self, rates: np.ndarray) -> tuple[float, float]:
        return float(rates[self.approach].mean()), float(rates[self.avoid].mean())

    def observe(self, spike_idx: np.ndarray, spike_cnt: np.ndarray, dt: float) -> float:
        app = spike_cnt[self._is_app[spike_idx]].sum() / (dt * max(1, self.approach.size))
        av = spike_cnt[self._is_av[spike_idx]].sum() / (dt * max(1, self.avoid.size))
        v_inst = self.k_app * app - self.k_av * av
        self.value += (1 - math.exp(-dt / self.p.tau)) * (v_inst - self.value)
        return self.value

    def steering(self, odor_left: float, odor_right: float) -> float:
        p = self.p
        total = odor_left + odor_right
        if total < p.odour_floor:
            self.last_bias = 0.0
            return 0.0
        contrast = float(np.clip((odor_left - odor_right) / total * p.contrast_gain, -1.0, 1.0))
        self.last_bias = p.gain * math.tanh(self.value / p.v_ref) * contrast
        return self.last_bias

    def reset(self) -> None:
        self.value = 0.0
        self.last_bias = 0.0

    # ------------------------------------------------------------------ calibration
    @staticmethod
    def calibrate(connectome: Connectome, brain_factory, seconds: float = 0.5, rate_hz: float = 80.0) -> tuple[float, float]:
        """Return (k_app, k_av) that balance naive MBON group responses over the odour panel."""
        ro = ValenceReadout(connectome)
        ann = connectome.annotations
        apps, avs = [], []
        brain = brain_factory()
        for types in CALIBRATION_ODOURS:
            idx = np.flatnonzero(np.array(ann["cell_type"].isin(types)))
            brain.reset(0)
            brain.set_input_rates(idx, np.full(idx.size, rate_hz))
            brain.step(int(round(seconds / brain.dt)))
            app, av = ro.group_rates(brain.spike_count / seconds)
            apps.append(app)
            avs.append(av)
        m_app, m_av = float(np.mean(apps)), float(np.mean(avs))
        if m_app <= 0 or m_av <= 0:
            raise ValueError(f"cannot calibrate valence: naive MBON group means {m_app:.3f}/{m_av:.3f} Hz")
        log.info("valence calibration: approach %.2f Hz, avoidance %.2f Hz", m_app, m_av)
        return 1.0 / m_app, 1.0 / m_av


def load_or_calibrate(connectome: Connectome, preset: str, cache_dir: Path, brain_factory) -> tuple[float, float]:
    path = cache_dir / "valence_calibration.json"
    data = json.loads(path.read_text()) if path.is_file() else {}
    if preset not in data:
        k_app, k_av = ValenceReadout.calibrate(connectome, brain_factory)
        data[preset] = {"k_app": k_app, "k_av": k_av}
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1))
    return data[preset]["k_app"], data[preset]["k_av"]
