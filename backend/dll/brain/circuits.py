"""Named neuron populations used as sensory inputs and motor readouts.

Every population is resolved from FlyWire v783 annotations (Schlegel et al. 2024) —
no neuron is hand-picked by index. Each carries an `evidence` label:

  validated   — the input→output relationship was tested in the Shiu et al. 2024 model
                against experiments (e.g. sugar GRNs → MN9).
  literature  — the population's function is established experimentally, but the
                mapping used here (e.g. "DNa02 = ipsilateral steering") is our
                interpretation inside this simulator.
  approximation — a modelling choice with weak or indirect support (e.g. a single
                "food odour" driving a fixed set of ORN types).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .connectome import Connectome

SIDES = ("left", "right")


@dataclass(frozen=True)
class Population:
    key: str  # e.g. "taste_sugar_left"
    label: str
    side: str  # left | right | center | both
    indices: np.ndarray
    selector: dict[str, list[str]]
    evidence: str
    reference: str

    @property
    def size(self) -> int:
        return int(self.indices.size)


@dataclass(frozen=True)
class PopulationSpec:
    name: str
    label: str
    selector: dict[str, list[str]]
    evidence: str
    reference: str
    lateral: bool = True


SENSORY_SPECS: tuple[PopulationSpec, ...] = (
    PopulationSpec(
        "taste_sugar",
        "Sugar GRNs",
        {"cell_sub_class": ["sugar"]},
        "validated",
        "Shiu et al. 2024: sugar GRN activation drives MN9 / proboscis extension",
    ),
    PopulationSpec(
        "taste_bitter",
        "Bitter GRNs",
        {"cell_sub_class": ["bitter"]},
        "validated",
        "Shiu et al. 2024: bitter GRNs inhibit sugar-evoked MN9 activity",
    ),
    PopulationSpec(
        "odor_food",
        "Food-odour ORNs (DM1, DM2, DM4, DP1m, VA2)",
        {"cell_type": ["ORN_DM1", "ORN_DM2", "ORN_DM4", "ORN_DP1m", "ORN_VA2"]},
        "approximation",
        "Vinegar/fruit-responsive glomeruli (Semmelhack & Wang 2009; Münch & Galizia 2016 DoOR). "
        "One generic 'food odour' activating all five types equally is a simplification.",
    ),
    PopulationSpec(
        "temp_hot",
        "Hot-sensing arista neurons (TRN_VP2)",
        {"cell_type": ["TRN_VP2"]},
        "literature",
        "Gallio et al. 2011 (hot cells, VP2 glomerulus). Tonic encoding used here is an approximation "
        "of their largely phasic responses.",
    ),
    PopulationSpec(
        "temp_cold",
        "Cold-sensing arista neurons (TRN_VP3a/b)",
        {"cell_type": ["TRN_VP3a", "TRN_VP3b"]},
        "literature",
        "Gallio et al. 2011 (cold cells, VP3 glomerulus); same encoding caveat as hot cells.",
    ),
    PopulationSpec(
        "light_ocellar",
        "Ocellar photoreceptors",
        {"cell_class": ["visual"], "cell_sub_class": ["ocellar"]},
        "literature",
        "Ocelli are light-intensity detectors (Krapp 2009). Medial ('center') ocellus is driven by the mean "
        "of both sides.",
    ),
    PopulationSpec(
        "touch_head",
        "Head bristle mechanosensory neurons",
        {"cell_sub_class": ["head bristle"]},
        "literature",
        "Head bristle mechanoreceptors (grooming circuits: Hampel et al. 2015; Eichler et al. 2024). Collision of the head drives the ipsilateral set.",
    ),
    PopulationSpec(
        "mech_jo",
        "Johnston's organ wind/gravity neurons (JO-CE)",
        {"cell_sub_class": ["wind_gravity"]},
        "literature",
        "Antennal mechanosensation (Kamikouchi et al. 2009). Used for user-applied mechanical stimulation.",
    ),
)

MOTOR_SPECS: tuple[PopulationSpec, ...] = (
    PopulationSpec(
        "dn_forward",
        "DNp09 (forward walking command)",
        {"cell_type": ["DNp09"]},
        "literature",
        "Bidaye et al. 2020 (P9 neurons drive forward walking); Braun et al. 2024",
    ),
    PopulationSpec(
        "dn_steer_high",
        "DNa02 (high-gain steering)",
        {"cell_type": ["DNa02"]},
        "literature",
        "Rayshubskiy et al. 2024 eLife: DNa02 activity predicts ipsilateral turning",
    ),
    PopulationSpec(
        "dn_steer_low",
        "DNa01 (low-gain steering)",
        {"cell_type": ["DNa01"]},
        "literature",
        "Rayshubskiy et al. 2024 eLife (weaker / less certain role)",
    ),
    PopulationSpec(
        "dn_backward",
        "MDN (moonwalker, backward walking)",
        {"cell_type": ["MDN"]},
        "literature",
        "Bidaye et al. 2014 Science",
    ),
    PopulationSpec(
        "dn_escape",
        "DNp01 (giant fiber, escape take-off)",
        {"cell_type": ["DNp01"]},
        "literature",
        "von Reyn et al. 2014 Nat Neurosci",
    ),
    PopulationSpec(
        "mn_proboscis",
        "MN9 (proboscis extension motor neuron, CB0701)",
        {"cell_type": ["CB0701"]},
        "validated",
        "Gordon & Scott 2009; Shiu et al. 2024 (identified as FlyWire type CB0701)",
    ),
)


def _select(ann: pd.DataFrame, selector: dict[str, list[str]], side: str | None) -> np.ndarray:
    mask = np.ones(len(ann), dtype=bool)
    for col, values in selector.items():
        mask &= ann[col].isin(values).to_numpy()
    if side is not None:
        mask &= (ann["side"] == side).to_numpy()
    return np.flatnonzero(mask)


def resolve(connectome: Connectome, specs: tuple[PopulationSpec, ...]) -> dict[str, Population]:
    ann = connectome.annotations
    out: dict[str, Population] = {}
    for spec in specs:
        sides: tuple[str, ...] = SIDES if spec.lateral else ("both",)
        if spec.name == "light_ocellar":
            sides = ("left", "right", "center")
        for side in sides:
            idx = _select(ann, spec.selector, None if side == "both" else side)
            if idx.size == 0:
                raise ValueError(f"population {spec.name}/{side} resolved to zero neurons — annotation mismatch")
            key = f"{spec.name}_{side}"
            out[key] = Population(key, spec.label, side, idx, spec.selector, spec.evidence, spec.reference)
    return out


#: Neuron classes receiving short-term depression in the `al_std` preset. These host the
#: recurrent excitatory attractor documented in RESEARCH.md §4.
OLFACTORY_SYSTEM_CLASSES = (
    "ALLN",
    "ALPN",
    "ALIN",
    "ALON",
    "Kenyon_Cell",
    "LHLN",
    "LHCENT",
    "MBON",
    "DAN",
    "MBIN",
)


def olfactory_system_mask(connectome: Connectome) -> np.ndarray:
    ann = connectome.annotations
    return (ann["cell_class"].isin(OLFACTORY_SYSTEM_CLASSES) & ~ann["super_class"].isin(["sensory"])).to_numpy()
