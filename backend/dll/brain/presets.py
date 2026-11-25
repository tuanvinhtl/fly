"""Brain model presets. See RESEARCH.md §4 for the evidence behind each."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .circuits import olfactory_system_mask
from .connectome import Connectome
from .lif import LIFBrain, LIFParams

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    deviation_from_publication: str


PRESETS: dict[str, Preset] = {
    "exact": Preset(
        "exact",
        "Shiu et al. 2024 LIF model, unmodified.",
        "none",
    ),
    "ln_std": Preset(
        "ln_std",
        "Shiu et al. 2024 + short-term synaptic depression (U=0.4, tau_rec=0.5 s) on antennal-lobe "
        "local neurons only (ALLN).",
        "Adds STD to the ~400 antennal-lobe local neurons that form the core of the recurrent "
        "excitatory attractor (RESEARCH.md §4). No persistent activity for any olfactory or thermal "
        "input; Kenyon cells respond sparsely and odour-specifically (6-12% active, as in vivo); "
        "validated gustatory, mechanosensory and looming pathways unchanged.",
    ),
    "al_std": Preset(
        "al_std",
        "Shiu et al. 2024 + short-term synaptic depression (U=0.4, tau_rec=0.5 s) on antennal-lobe, "
        "mushroom-body and lateral-horn neurons.",
        "Adds STD to ~10k olfactory-system neurons. Without it, any olfactory or thermosensory input "
        "ignites a self-sustaining whole-brain attractor (~10k neurons at up to 450 Hz) that never "
        "terminates. Validated gustatory pathways are unchanged (sugar->MN9 50/82 Hz in both presets).",
    ),
}

AL_STD_U = 0.4
AL_STD_TAU = 0.5


def build_brain(connectome: Connectome, preset: str, seed: int, engine: str = "numba") -> LIFBrain:
    if preset not in PRESETS:
        raise ValueError(f"unknown brain preset {preset!r}; choose from {sorted(PRESETS)}")
    if preset == "exact":
        return LIFBrain(connectome, LIFParams(), seed=seed, engine=engine)
    brain = LIFBrain(connectome, LIFParams(std_U=AL_STD_U, std_tau_rec=AL_STD_TAU), seed=seed, engine=engine)
    if preset == "ln_std":
        ann = connectome.annotations
        mask = np.array((ann["cell_class"] == "ALLN") & (ann["super_class"] != "sensory"))
        brain.set_depression(np.where(mask, AL_STD_U, 0.0))
        log.info("preset ln_std: depression on %d antennal-lobe local neurons", int(mask.sum()))
        return brain
    mask = olfactory_system_mask(connectome)
    brain.set_depression(np.where(mask, AL_STD_U, 0.0))
    log.info("preset al_std: depression on %d neurons", int(mask.sum()))
    return brain
