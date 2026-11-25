"""Organism-level physiology: energy, health, age and the internal locomotor drive.

All quantities and rates here are APPROXIMATIONS chosen to make survival experiments
possible on minute time scales; they are not fitted to Drosophila metabolism.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..body.vnc import MotorCommand


@dataclass(frozen=True)
class PhysiologyParams:
    basal_cost: float = 0.25  # energy/s
    locomotion_cost: float = 0.6  # energy/s at full forward drive
    intake_rate: float = 6.0  # energy/s when feeding on strong sugar
    feeding_proboscis: float = 0.5  # proboscis extension needed to ingest
    heat_damage_above: float = 34.0  # °C
    cold_damage_below: float = 12.0  # °C
    thermal_damage: float = 4.0  # health/s
    toxic_damage: float = 5.0  # health/s per bitter unit while touching toxin
    starvation_damage: float = 1.0  # health/s at zero energy
    regeneration: float = 0.3  # health/s when energy > 50
    # Internal-state drive onto DNp09 (APPROXIMATION: the connectome model has no
    # spontaneous activity, so without some drive the animal never walks).
    drive_base_hz: float = 30.0
    drive_hunger_hz: float = 50.0


@dataclass
class Organism:
    id: str = "fly-001"
    species: str = "Drosophila melanogaster"
    energy: float = 70.0
    health: float = 100.0
    age: float = 0.0  # simulated seconds
    alive: bool = True
    food_eaten: float = 0.0
    distance: float = 0.0
    feeding: bool = False
    death_cause: str | None = None
    params: PhysiologyParams = field(default_factory=PhysiologyParams)

    @property
    def hunger(self) -> float:
        return 1.0 - self.energy / 100.0

    def drive_hz(self) -> float:
        return self.params.drive_base_hz + self.params.drive_hunger_hz * self.hunger if self.alive else 0.0

    def update(
        self,
        dt: float,
        cmd: MotorCommand,
        temperature: float,
        bitter: float,
        sugar: float,
        on_food: bool,
        step_distance: float,
    ) -> float:
        """Advance physiology. Returns energy ingested this step."""
        if not self.alive:
            return 0.0
        p = self.params
        self.age += dt
        self.distance += step_distance
        self.energy -= dt * (p.basal_cost + p.locomotion_cost * abs(cmd.forward))
        intake = 0.0
        self.feeding = on_food and cmd.proboscis >= p.feeding_proboscis and sugar > 0
        if self.feeding:
            intake = dt * p.intake_rate * min(1.0, sugar)
            intake = min(intake, 100.0 - max(self.energy, 0.0))
            self.energy += intake
            self.food_eaten += intake
        self.energy = max(0.0, min(100.0, self.energy))
        sources = {
            "heat": p.thermal_damage * max(0.0, temperature - p.heat_damage_above) / 4.0,
            "cold": p.thermal_damage * max(0.0, p.cold_damage_below - temperature) / 4.0,
            "toxin": p.toxic_damage * bitter,
            "starvation": p.starvation_damage if self.energy <= 0 else 0.0,
        }
        damage = sum(sources.values())
        self.health -= dt * damage
        if self.energy > 50 and damage == 0:
            self.health += dt * p.regeneration
        self.health = max(0.0, min(100.0, self.health))
        if self.health <= 0:
            self.alive = False
            self.feeding = False
            self.death_cause = max(sources, key=lambda k: sources[k])
        return intake

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("params")
        d["hunger"] = self.hunger
        d["drive_hz"] = self.drive_hz()
        return d
