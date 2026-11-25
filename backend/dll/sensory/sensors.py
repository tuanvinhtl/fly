"""Sensory engine: physical stimuli at the body -> firing rates of sensory populations.

Encoding functions are APPROXIMATIONS chosen for plausibility, not fitted to data:
  * saturating (Michaelis-Menten) rate codes for chemosensation and light
  * tonic linear codes for temperature (real hot/cold cells are largely phasic)
  * fixed-rate burst for head contact
Rates are what is fed to the connectome model as Poisson drive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..body.body import FlyBody
from ..environment.world import World

LEFT, RIGHT = "left", "right"


@dataclass(frozen=True)
class EncoderParams:
    sugar_rmax: float = 150.0
    sugar_k: float = 0.3
    bitter_rmax: float = 150.0
    bitter_k: float = 0.3
    odor_rmax: float = 80.0
    odor_k: float = 0.3
    odor_threshold: float = 0.01  # concentration below which ORNs are silent
    temp_ref: float = 25.0  # °C neutral
    temp_span: float = 12.0  # °C for full response
    temp_rmax: float = 80.0
    light_rmax: float = 120.0
    light_k: float = 1.0
    touch_rate: float = 150.0
    loom_rmax: float = 200.0  # Hz on LC4/LPLC2 at full expansion rate
    loom_ref: float = 4.0  # rad/s angular expansion for full response
    loom_min: float = 0.3  # rad/s below which the neurons stay silent


@dataclass
class SensoryReading:
    """Physical quantities and the resulting population rates (Hz)."""

    physical: dict[str, float] = field(default_factory=dict)
    rates: dict[str, float] = field(default_factory=dict)
    food_contact: str | None = None  # id of food under the labellum


def _mm(c: float, k: float) -> float:
    return c / (c + k) if c > 0 else 0.0


class SensoryEngine:
    def __init__(self, params: EncoderParams | None = None):
        self.p = params or EncoderParams()
        self.external: dict[str, float] = {}  # user-applied stimulation, Hz per population
        self._theta: dict[str, float] = {}  # last angular size of each threat (rad)

    def read(self, world: World, body: FlyBody, dt: float = 0.01) -> SensoryReading:
        p = self.p
        r = SensoryReading()
        st = body.state()
        fwd = np.array([math.cos(st.heading), math.sin(st.heading)])
        left_vec = np.array([-fwd[1], fwd[0]])

        # --- olfaction (antennae)
        for side, site in ((LEFT, "antenna_L"), (RIGHT, "antenna_R")):
            pos = body.site_pos(site)
            c = world.odor(pos[0], pos[1])
            r.physical[f"odor_{side}"] = c
            r.rates[f"odor_food_{side}"] = p.odor_rmax * _mm(c, p.odor_k) if c > p.odor_threshold else 0.0

        # --- taste: labellum (both sides) + front-leg tarsi (ipsilateral)
        lab = body.site_pos("labellum")
        sugar_lab, food = world.sugar_at(lab[0], lab[1])
        bitter_lab = world.bitter_at(lab[0], lab[1])
        r.food_contact = food.id if food is not None else None
        for side, leg in ((LEFT, "L1"), (RIGHT, "R1")):
            tp = body.data.geom_xpos[body.foot_geoms[leg]]
            sugar = max(sugar_lab, world.sugar_at(tp[0], tp[1])[0])
            bitter = max(bitter_lab, world.bitter_at(tp[0], tp[1]))
            r.physical[f"sugar_{side}"] = sugar
            r.physical[f"bitter_{side}"] = bitter
            r.rates[f"taste_sugar_{side}"] = p.sugar_rmax * _mm(sugar, p.sugar_k)
            r.rates[f"taste_bitter_{side}"] = p.bitter_rmax * _mm(bitter, p.bitter_k)

        # --- temperature (arista, at antennae)
        for side, site in ((LEFT, "antenna_L"), (RIGHT, "antenna_R")):
            pos = body.site_pos(site)
            t = world.temperature(pos[0], pos[1])
            r.physical[f"temperature_{side}"] = t
            r.rates[f"temp_hot_{side}"] = p.temp_rmax * float(np.clip((t - p.temp_ref) / p.temp_span, 0, 1))
            r.rates[f"temp_cold_{side}"] = p.temp_rmax * float(np.clip((p.temp_ref - t) / p.temp_span, 0, 1))

        # --- light (ocelli face up-and-sideways)
        lum = {}
        for side, site, vec in ((LEFT, "ocellus_L", left_vec), (RIGHT, "ocellus_R", -left_vec)):
            pos = body.site_pos(site)
            facing = 0.5 * fwd + vec
            facing /= np.linalg.norm(facing)
            lum[side] = world.light(pos[0], pos[1], facing)
            r.physical[f"light_{side}"] = lum[side]
            r.rates[f"light_ocellar_{side}"] = p.light_rmax * _mm(lum[side], p.light_k)
        center = 0.5 * (lum[LEFT] + lum[RIGHT])
        r.rates["light_ocellar_center"] = p.light_rmax * _mm(center, p.light_k)

        # --- touch: head contacts with obstacles, side from contact position in body frame
        touch = {LEFT: 0.0, RIGHT: 0.0}
        head = body.data.geom_xpos[body.head_geom]
        for _, cpos in body.head_contacts():
            lateral = float(np.dot(cpos[:2] - head[:2], left_vec))
            touch[LEFT if lateral >= 0 else RIGHT] = 1.0
        for side in (LEFT, RIGHT):
            r.physical[f"touch_{side}"] = touch[side]
            r.rates[f"touch_head_{side}"] = p.touch_rate * touch[side]
            r.rates[f"mech_jo_{side}"] = 0.0

        # --- looming: angular expansion rate of approaching threats, split by hemifield
        loom = {LEFT: 0.0, RIGHT: 0.0}
        seen = set()
        for th in world.cfg.threats:
            rel = np.array([th.x, th.y, th.z]) - head
            d = max(float(np.linalg.norm(rel)), th.radius * 1.05)
            theta = 2.0 * math.atan(th.radius / d)
            prev = self._theta.get(th.id, theta)
            self._theta[th.id] = theta
            seen.add(th.id)
            rate = (theta - prev) / dt
            horiz = float(np.hypot(rel[0], rel[1]))
            lat = float(np.dot(rel[:2], left_vec)) / horiz if horiz > 1e-6 else 0.0
            for side, wgt in ((LEFT, 0.5 + 0.5 * lat), (RIGHT, 0.5 - 0.5 * lat)):
                loom[side] = max(loom[side], wgt * rate)
        for gone in set(self._theta) - seen:
            del self._theta[gone]
        for side in (LEFT, RIGHT):
            x = (loom[side] - p.loom_min) / p.loom_ref
            r.physical[f"looming_{side}"] = loom[side]
            r.rates[f"vision_looming_{side}"] = p.loom_rmax * float(np.clip(x, 0.0, 1.0))

        # --- user-applied stimulation adds to the physical drive
        for key, hz in self.external.items():
            r.rates[key] = r.rates.get(key, 0.0) + hz
        return r
