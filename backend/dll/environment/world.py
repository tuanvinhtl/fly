"""Environment state: arena, food, obstacles, lights, temperature and toxic zones.

Fields are analytic (no fluid simulation):
  odour   c(p) = sum_i s_i * exp(-|p - p_i|^2 / (2 sigma_i^2))     — APPROXIMATION of a
          time-averaged diffusive plume; real plumes are turbulent and intermittent.
  light   L(p) = ambient + sum_j I_j / (1 + |p - p_j|^2 / d0^2)
  temp    T(p) = ambient + sum_k (T_k - ambient) * w_k(p), w = smooth disc profile
All coordinates are millimetres in the arena plane (z up).
"""

from __future__ import annotations

import itertools
import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

_ids = itertools.count(1)


def _new_id(prefix: str) -> str:
    return f"{prefix}{next(_ids)}"


class Food(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("food"))
    x: float
    y: float
    radius: float = Field(3.0, gt=0, le=20)
    sugar: float = Field(1.0, ge=0, le=5, description="sucrose concentration (arbitrary units, 1 = strong)")
    odor: float = Field(1.0, ge=0, le=5, description="odour source strength")
    odor_sigma: float = Field(15.0, gt=0, le=100, description="plume width (mm)")
    amount: float = Field(50.0, ge=0, description="energy units remaining")


class Obstacle(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("obs"))
    x: float
    y: float
    sx: float = Field(2.0, gt=0, le=200, description="half size x (mm)")
    sy: float = Field(2.0, gt=0, le=200, description="half size y (mm)")
    height: float = Field(1.5, gt=0, le=20)


class Light(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("light"))
    x: float
    y: float
    intensity: float = Field(1.0, ge=0, le=10)
    d0: float = Field(10.0, gt=0, description="distance at which intensity halves (mm)")


class Zone(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("zone"))
    kind: Literal["temperature", "toxic"]
    x: float
    y: float
    radius: float = Field(6.0, gt=0, le=200)
    temperature: float = Field(25.0, ge=0, le=50, description="°C, temperature zones only")
    bitter: float = Field(1.0, ge=0, le=5, description="bitter concentration, toxic zones only")


class WorldConfig(BaseModel):
    arena_half: float = Field(40.0, gt=5, le=500, description="arena half width (mm), walled")
    ambient_temperature: float = Field(25.0, ge=0, le=50)
    ambient_light: float = Field(0.2, ge=0, le=10)
    food: list[Food] = Field(default_factory=list)
    obstacles: list[Obstacle] = Field(default_factory=list)
    lights: list[Light] = Field(default_factory=list)
    zones: list[Zone] = Field(default_factory=list)


def default_world() -> WorldConfig:
    return WorldConfig(
        food=[Food(x=14, y=6), Food(x=-12, y=-16, sugar=0.6, amount=30)],
        obstacles=[Obstacle(x=6, y=-8, sx=1.0, sy=6.0)],
        lights=[Light(x=-20, y=20, intensity=2.0)],
        zones=[
            Zone(kind="temperature", x=22, y=-22, radius=8, temperature=38),
            Zone(kind="temperature", x=-24, y=8, radius=7, temperature=15),
            Zone(kind="toxic", x=0, y=24, radius=5, bitter=1.0),
        ],
    )


def _disc_weight(d: float, radius: float, edge: float = 2.0) -> float:
    """1 inside the disc, smooth falloff over `edge` mm outside it."""
    if d <= radius:
        return 1.0
    u = min(1.0, (d - radius) / edge)
    return 1.0 - u * u * (3 - 2 * u)


class World:
    def __init__(self, config: WorldConfig):
        self.cfg = config.model_copy(deep=True)
        self.version = 0  # incremented when obstacles change (body must rebuild)

    # ------------------------------------------------------------- fields
    def odor(self, x: float, y: float) -> float:
        c = 0.0
        for f in self.cfg.food:
            if f.amount <= 0:
                continue
            d2 = (x - f.x) ** 2 + (y - f.y) ** 2
            c += f.odor * math.exp(-d2 / (2 * f.odor_sigma**2))
        return c

    def light(self, x: float, y: float, facing: np.ndarray | None = None) -> float:
        """Light at a point; with `facing` (unit 2-vector) sources behind are attenuated."""
        lum = self.cfg.ambient_light
        for s in self.cfg.lights:
            dx, dy = s.x - x, s.y - y
            d2 = dx * dx + dy * dy
            gain = 1.0
            if facing is not None and d2 > 1e-9:
                cos = (dx * facing[0] + dy * facing[1]) / math.sqrt(d2)
                gain = 0.25 + 0.75 * max(0.0, cos)
            lum += gain * s.intensity / (1 + d2 / s.d0**2)
        return lum

    def temperature(self, x: float, y: float) -> float:
        t = self.cfg.ambient_temperature
        for z in self.cfg.zones:
            if z.kind == "temperature":
                w = _disc_weight(math.hypot(x - z.x, y - z.y), z.radius)
                t += (z.temperature - self.cfg.ambient_temperature) * w
        return t

    def sugar_at(self, x: float, y: float) -> tuple[float, Food | None]:
        for f in self.cfg.food:
            if f.amount > 0 and math.hypot(x - f.x, y - f.y) <= f.radius:
                return f.sugar, f
        return 0.0, None

    def bitter_at(self, x: float, y: float) -> float:
        b = 0.0
        for z in self.cfg.zones:
            if z.kind == "toxic" and math.hypot(x - z.x, y - z.y) <= z.radius:
                b = max(b, z.bitter)
        return b

    # ------------------------------------------------------------- edits
    def add(self, kind: str, obj: BaseModel) -> None:
        getattr(self.cfg, kind).append(obj)
        if kind == "obstacles":
            self.version += 1

    def remove(self, obj_id: str) -> str:
        for kind in ("food", "obstacles", "lights", "zones"):
            items = getattr(self.cfg, kind)
            for i, o in enumerate(items):
                if o.id == obj_id:
                    items.pop(i)
                    if kind == "obstacles":
                        self.version += 1
                    return kind
        raise KeyError(f"no world object with id {obj_id}")

    def update(self, obj_id: str, changes: dict) -> None:
        for kind in ("food", "obstacles", "lights", "zones"):
            for i, o in enumerate(getattr(self.cfg, kind)):
                if o.id == obj_id:
                    new = o.model_validate({**o.model_dump(), **changes, "id": obj_id})
                    getattr(self.cfg, kind)[i] = new
                    if kind == "obstacles":
                        self.version += 1
                    return
        raise KeyError(f"no world object with id {obj_id}")

    def obstacle_boxes(self) -> list[tuple[str, float, float, float, float, float]]:
        """Obstacles plus the four arena walls as (name, x, y, sx, sy, h)."""
        a = self.cfg.arena_half
        walls = [
            ("wall_n", 0.0, a + 0.5, a + 1, 0.5, 3.0),
            ("wall_s", 0.0, -a - 0.5, a + 1, 0.5, 3.0),
            ("wall_e", a + 0.5, 0.0, 0.5, a + 1, 3.0),
            ("wall_w", -a - 0.5, 0.0, 0.5, a + 1, 3.0),
        ]
        return walls + [(o.id, o.x, o.y, o.sx, o.sy, o.height) for o in self.cfg.obstacles]
