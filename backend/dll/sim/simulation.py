"""Closed-loop simulation: ENVIRONMENT -> SENSORS -> BRAIN -> VNC -> BODY -> ENVIRONMENT.

One control tick (default 10 ms of simulated time):
  1. read sensors from the current world + body state
  2. encode as Poisson rates on sensory populations (+ internal DNp09 drive)
  3. advance the connectome LIF model by 10 ms (100 steps of 0.1 ms)
  4. read descending/motor neuron rates (exponentially filtered spike counts)
  5. VNC adapter converts DN rates to joint targets
  6. advance MuJoCo physics by 10 ms
  7. update physiology (energy, feeding, damage) and the world (food depletion)

Commands (world edits, stimuli, silencing...) are queued and applied at the start of a
tick and logged with the tick index, so a run is reproducible from (seed, preset,
initial world, command log). See `replay_check` in experiment/replay.py.
"""

from __future__ import annotations

import copy
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..body.body import FlyBody
from ..body.vnc import VNC, MotorCommand
from ..brain.circuits import MOTOR_SPECS, SENSORY_SPECS, Population, resolve
from ..brain.connectome import Connectome
from ..brain.lif import LIFBrain
from ..brain.presets import build_brain
from ..environment.world import Food, Light, Obstacle, World, WorldConfig, Zone, default_world
from ..learning.associative import KCMBONPlasticity
from ..learning.habituation import Habituation
from ..sensory.sensors import SensoryEngine, SensoryReading
from .organism import Organism

log = logging.getLogger(__name__)

_KINDS = {"food": Food, "obstacles": Obstacle, "lights": Light, "zones": Zone}

# Events fire when a value crosses `on` and end when it falls below `off`.
SENSORY_EVENT_ON, SENSORY_EVENT_OFF = 10.0, 5.0
NEURAL_EVENT_ON, NEURAL_EVENT_OFF = 15.0, 5.0


@dataclass
class SimEvent:
    t: float
    tick: int
    category: str  # sensory | neural | motor | movement | environment | organism | experiment
    name: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"t": round(self.t, 4), "tick": self.tick, "category": self.category, "name": self.name, "data": self.data}


class Simulation:
    TICK = 0.01  # s

    def __init__(
        self,
        connectome: Connectome,
        preset: str = "al_std",
        seed: int = 42,
        world: WorldConfig | None = None,
        engine: str = "numba",
        rate_tau: float = 0.05,
    ):
        self.c = connectome
        self.preset = preset
        self.seed = seed
        self.engine = engine
        self.initial_world = (world or default_world()).model_copy(deep=True)
        self.brain: LIFBrain = build_brain(connectome, preset, seed, engine)
        self.sensory_pops: dict[str, Population] = resolve(connectome, SENSORY_SPECS)
        self.motor_pops: dict[str, Population] = resolve(connectome, MOTOR_SPECS)
        self.drive_pop = self.motor_pops["dn_forward_left"].indices, self.motor_pops["dn_forward_right"].indices
        self.rate_tau = rate_tau
        self.steps_per_tick = int(round(self.TICK / self.brain.dt))
        self.sensors = SensoryEngine()
        self.vnc = VNC()
        self.habituation = Habituation()
        self.plasticity: KCMBONPlasticity | None = None
        self.habituation_on = False
        self.plasticity_on = False
        self._init_clusters()
        self.body: FlyBody | None = None
        self.reset(seed)

    # ------------------------------------------------------------------ setup
    def _init_clusters(self) -> None:
        ann = self.c.annotations
        cls = ann["cell_class"].where(ann["cell_class"] != "", ann["super_class"]).where(lambda s: s != "", "unannotated")
        counts = cls.value_counts()
        top = counts.index[:31].tolist()
        cls = cls.where(cls.isin(top), "other")
        self.cluster_names = sorted(cls.unique().tolist())
        lookup = {n: i for i, n in enumerate(self.cluster_names)}
        self.cluster_id = cls.map(lookup).to_numpy(np.int64)
        self.cluster_sizes = np.bincount(self.cluster_id, minlength=len(self.cluster_names))

    def reset(self, seed: int | None = None, world: WorldConfig | None = None) -> None:
        if seed is not None:
            self.seed = seed
        if world is not None:
            self.initial_world = world.model_copy(deep=True)
        self.world = World(self.initial_world)
        self.brain.reset(self.seed)
        self.brain.unsilence_all()
        boxes = self.world.obstacle_boxes()
        if self.body is None:
            self.body = FlyBody(boxes)
        else:
            self.body.rebuild(boxes)
        rng = np.random.default_rng(self.seed)
        self.body.reset(x=0.0, y=0.0, heading=float(rng.uniform(-math.pi, math.pi)))
        self._world_version = self.world.version
        self.vnc.reset()
        self.habituation.reset()
        if self.plasticity is not None:
            self.plasticity.reset()
        self.sensors.external.clear()
        self.stim_expiry: dict[str, int] = {}
        self.organism = Organism()
        self.tick_count = 0
        self.pending: deque[dict] = deque()
        self.command_log: list[dict] = []
        self.events: list[SimEvent] = []
        self._prev_counts = np.zeros(self.brain.n, dtype=np.int64)
        self.motor_rates = {k: 0.0 for k in self.motor_pops}
        self.sensory_rates = {k: 0.0 for k in self.sensory_pops}
        self.last_reading = SensoryReading()
        self.last_cmd = MotorCommand(0.0, 0.0, False, 0.0)
        self._latch: dict[str, bool] = {}
        self.cluster_rates = np.zeros(len(self.cluster_names))
        self.tick_active_idx = np.zeros(0, dtype=np.int64)
        self.tick_active_cnt = np.zeros(0, dtype=np.int64)
        self.frame_counts = np.zeros(self.brain.n, dtype=np.int64)
        self.total_spikes_tick = 0
        log.info("simulation reset: seed=%d preset=%s", self.seed, self.preset)

    @property
    def time(self) -> float:
        return self.tick_count * self.TICK

    # ------------------------------------------------------------------ commands
    def submit(self, command: dict) -> None:
        """Queue a command for the next tick (thread-safe enough for the GIL-held deque)."""
        self.validate_command(command)
        self.pending.append(command)

    def validate_command(self, command: dict) -> None:
        op = command.get("op")
        if op == "world.add":
            if command.get("kind") not in _KINDS:
                raise ValueError(f"unknown world kind {command.get('kind')!r}")
            _KINDS[command["kind"]].model_validate(command.get("object", {}))
        elif op == "stimulus":
            if command.get("population") not in self.sensory_pops:
                raise ValueError(f"unknown sensory population {command.get('population')!r}")
            if not 0 <= float(command.get("rate_hz", -1)) <= 500:
                raise ValueError("rate_hz must be within [0, 500]")
        elif op == "brain.silence":
            if not command.get("cell_types"):
                raise ValueError("cell_types required")
        elif op not in {
            "world.remove", "world.update", "world.ambient", "brain.unsilence",
            "organism.set", "learning.set", "drive.set",
        }:
            raise ValueError(f"unknown command op {op!r}")

    def _apply(self, cmd: dict) -> None:
        op = cmd["op"]
        if op == "world.add":
            obj = _KINDS[cmd["kind"]].model_validate(cmd["object"])
            self.world.add(cmd["kind"], obj)
            cmd["object"] = obj.model_dump()  # log the assigned id for replay
        elif op == "world.remove":
            self.world.remove(cmd["id"])
        elif op == "world.update":
            self.world.update(cmd["id"], cmd["changes"])
        elif op == "world.ambient":
            if "temperature" in cmd:
                self.world.cfg.ambient_temperature = float(cmd["temperature"])
            if "light" in cmd:
                self.world.cfg.ambient_light = float(cmd["light"])
        elif op == "stimulus":
            key = cmd["population"]
            self.sensors.external[key] = float(cmd["rate_hz"])
            self.stim_expiry[key] = self.tick_count + max(1, int(round(float(cmd.get("duration_s", 1.0)) / self.TICK)))
        elif op == "brain.silence":
            idx = np.flatnonzero(self.c.annotations["cell_type"].isin(cmd["cell_types"]).to_numpy())
            if idx.size == 0:
                raise ValueError(f"no neurons with cell types {cmd['cell_types']}")
            self.brain.silence(idx)
            cmd["n_silenced"] = int(idx.size)
        elif op == "brain.unsilence":
            self.brain.unsilence_all()
        elif op == "organism.set":
            for k in ("energy", "health"):
                if k in cmd:
                    setattr(self.organism, k, float(np.clip(cmd[k], 0, 100)))
        elif op == "drive.set":
            p = self.organism.params
            self.organism.params = type(p)(**{**p.__dict__, **{k: float(v) for k, v in cmd.items() if k in ("drive_base_hz", "drive_hunger_hz")}})
        elif op == "learning.set":
            if "habituation" in cmd:
                self.habituation_on = bool(cmd["habituation"])
            if "plasticity" in cmd:
                self.plasticity_on = bool(cmd["plasticity"])
                if self.plasticity_on and self.plasticity is None:
                    self.plasticity = KCMBONPlasticity(self.c)
        self._event("environment" if op.startswith("world") else "experiment", op, {k: v for k, v in cmd.items() if k != "op"})

    # ------------------------------------------------------------------ events
    def _event(self, category: str, name: str, data: dict | None = None) -> None:
        self.events.append(SimEvent(self.time, self.tick_count, category, name, data or {}))

    def _threshold_event(self, key: str, value: float, on: float, off: float, category: str, data: dict | None = None) -> None:
        state = self._latch.get(key, False)
        if not state and value >= on:
            self._latch[key] = True
            self._event(category, f"{key}:on", {"value": round(value, 2), **(data or {})})
        elif state and value < off:
            self._latch[key] = False
            self._event(category, f"{key}:off", {"value": round(value, 2), **(data or {})})

    # ------------------------------------------------------------------ main loop
    def tick(self) -> None:
        body, brain, org = self.body, self.brain, self.organism
        assert body is not None
        while self.pending:
            cmd = self.pending.popleft()
            entry = {"tick": self.tick_count, **cmd}
            self._apply(entry)
            self.command_log.append(entry)
        for key, until in list(self.stim_expiry.items()):
            if self.tick_count >= until:
                self.sensors.external.pop(key, None)
                del self.stim_expiry[key]
        if self.world.version != self._world_version:
            body.rebuild(self.world.obstacle_boxes())
            self._world_version = self.world.version

        # 1-2. sense and encode
        reading = self.sensors.read(self.world, body)
        rates = reading.rates
        if self.habituation_on:
            rates = self.habituation.apply(rates, self.TICK)
        if not org.alive:
            rates = {k: 0.0 for k in rates}
        idx_parts, rate_parts = [], []
        for key, pop in self.sensory_pops.items():
            r = rates.get(key, 0.0)
            if r > 0:
                idx_parts.append(pop.indices)
                rate_parts.append(np.full(pop.size, r))
        drive = org.drive_hz()
        if drive > 0:
            for side_idx in self.drive_pop:
                idx_parts.append(side_idx)
                rate_parts.append(np.full(side_idx.size, drive))
        if idx_parts:
            brain.set_input_rates(np.concatenate(idx_parts), np.concatenate(rate_parts))
        else:
            brain.set_input_rates(np.zeros(0, dtype=np.int64), np.zeros(0))

        # 3. brain
        brain.step(self.steps_per_tick)
        delta = brain.spike_count - self._prev_counts
        self._prev_counts = brain.spike_count.copy()
        nz = np.flatnonzero(delta)
        self.tick_active_idx, self.tick_active_cnt = nz, delta[nz]
        self.frame_counts[nz] += delta[nz]
        self.total_spikes_tick = int(delta[nz].sum())
        alpha = 1 - math.exp(-self.TICK / self.rate_tau)
        inst = np.bincount(self.cluster_id[nz], weights=delta[nz], minlength=len(self.cluster_names))
        self.cluster_rates += alpha * (inst / self.cluster_sizes / self.TICK - self.cluster_rates)
        silenced = brain.silenced
        for key, pop in self.motor_pops.items():
            # a silenced DN/MN releases no transmitter, so it drives no VNC/muscle output
            r_inst = float(np.where(silenced[pop.indices], 0, delta[pop.indices]).mean()) / self.TICK
            self.motor_rates[key] += alpha * (r_inst - self.motor_rates[key])
        self.sensory_rates = {k: rates.get(k, 0.0) for k in self.sensory_pops}
        if self.plasticity_on and self.plasticity is not None:
            self.plasticity.observe(brain, self.TICK)

        # 4-6. VNC + body
        cmd = self.vnc.command(self.motor_rates) if org.alive else MotorCommand(0.0, 0.0, False, 0.0)
        before = body.state().position
        targets = self.vnc.joint_targets(cmd, self.TICK)
        if not org.alive:
            targets = {k: (0.0 if k.endswith("yaw") else (-0.4 if k.endswith("pitch") else v)) for k, v in targets.items()}
        body.set_targets(targets)
        n_phys = int(round(self.TICK / body.dt))
        if cmd.escape and not self.last_cmd.escape:
            body.apply_impulse(np.array([0.0, 0.0, 6.0 * body.total_mass * 9810.0]), duration_steps=10)
            n_phys -= 10
        body.step(n_phys)
        after = body.state()
        step_dist = float(np.linalg.norm(after.position[:2] - before[:2]))

        # 7. physiology + world
        lab = body.site_pos("labellum")
        sugar, food = self.world.sugar_at(lab[0], lab[1])
        temp = 0.5 * (reading.physical.get("temperature_left", 25.0) + reading.physical.get("temperature_right", 25.0))
        bitter = max(reading.physical.get("bitter_left", 0.0), reading.physical.get("bitter_right", 0.0))
        was_feeding, was_alive = org.feeding, org.alive
        intake = org.update(self.TICK, cmd, temp, bitter, sugar, food is not None, step_dist)
        if food is not None and intake > 0:
            food.amount = max(0.0, food.amount - intake)
            if self.plasticity_on and self.plasticity is not None and self.tick_count % 10 == 0:
                n = self.plasticity.reward(brain)
                self._event("experiment", "plasticity.reward", {"kcs": n})
            if food.amount <= 0:
                self._event("environment", "food.depleted", {"id": food.id})

        # events (sensory -> neural -> motor -> movement -> organism)
        for key, r in self.sensory_rates.items():
            if not key.startswith("mech_") and not key.endswith("_center"):
                self._threshold_event(key, r, SENSORY_EVENT_ON, SENSORY_EVENT_OFF, "sensory")
        for key, r in self.motor_rates.items():
            self._threshold_event(key, r, NEURAL_EVENT_ON, NEURAL_EVENT_OFF, "neural")
        self._threshold_event("proboscis_extension", cmd.proboscis, 0.5, 0.2, "motor")
        self._threshold_event("turn_left", cmd.turn, 0.4, 0.15, "motor")
        self._threshold_event("turn_right", -cmd.turn, 0.4, 0.15, "motor")
        self._threshold_event("backward_walking", -cmd.forward, 0.3, 0.1, "motor")
        if cmd.escape and not self.last_cmd.escape:
            self._event("motor", "escape_takeoff", {})
        speed = step_dist / self.TICK
        self._threshold_event("walking", speed, 3.0, 1.0, "movement", {"speed_mm_s": round(speed, 2)})
        if org.feeding and not was_feeding:
            self._event("organism", "feeding:on", {"food": food.id if food else None})
        elif was_feeding and not org.feeding:
            self._event("organism", "feeding:off", {})
        if any(reading.physical.get(f"touch_{s}", 0) for s in ("left", "right")) and not self._latch.get("collision"):
            self._latch["collision"] = True
            self._event("movement", "collision", {"left": reading.physical["touch_left"], "right": reading.physical["touch_right"]})
        elif not any(reading.physical.get(f"touch_{s}", 0) for s in ("left", "right")):
            self._latch["collision"] = False
        if was_alive and not org.alive:
            self._event("organism", "death", {"cause": org.death_cause, "age_s": round(org.age, 2)})

        self.last_reading, self.last_cmd = reading, cmd
        self.tick_count += 1

    def run(self, seconds: float) -> None:
        for _ in range(int(round(seconds / self.TICK))):
            self.tick()

    # ------------------------------------------------------------------ snapshots
    def snapshot(self) -> dict:
        assert self.body is not None
        return {
            "tick": self.tick_count,
            "brain": self.brain.snapshot(),
            "silenced": self.brain.silenced.copy(),
            "body": self.body.get_state_vector(),
            "ctrl": self.body.data.ctrl.copy(),
            "world": self.world.cfg.model_copy(deep=True),
            "world_version": self.world.version,
            "organism": copy.deepcopy(self.organism),
            "vnc_phase": self.vnc.phase,
            "external": dict(self.sensors.external),
            "stim_expiry": dict(self.stim_expiry),
            "motor_rates": dict(self.motor_rates),
            "prev_counts": self._prev_counts.copy(),
            "latch": dict(self._latch),
            "last_cmd": self.last_cmd,
            "habituation": dict(self.habituation.gains),
            "learning": (self.habituation_on, self.plasticity_on),
            "organism_params": self.organism.params,
            "command_log": list(self.command_log),
            "n_events": len(self.events),
        }

    def restore(self, snap: dict) -> None:
        assert self.body is not None
        self.tick_count = snap["tick"]
        self.brain.silenced = snap["silenced"].copy()
        self.brain.restore(snap["brain"])
        self.world = World(snap["world"])
        self.world.version = snap["world_version"]
        self.body.rebuild(self.world.obstacle_boxes())
        self._world_version = self.world.version
        self.body.set_state_vector(snap["body"])
        self.body.data.ctrl[:] = snap["ctrl"]
        self.organism = copy.deepcopy(snap["organism"])
        self.vnc.phase = snap["vnc_phase"]
        self.sensors.external = dict(snap["external"])
        self.stim_expiry = dict(snap["stim_expiry"])
        self.motor_rates = dict(snap["motor_rates"])
        self._prev_counts = snap["prev_counts"].copy()
        self._latch = dict(snap["latch"])
        self.last_cmd = snap["last_cmd"]
        self.habituation.gains = dict(snap["habituation"])
        self.habituation_on, self.plasticity_on = snap["learning"]
        self.command_log = list(snap["command_log"])
        self.events = self.events[: snap["n_events"]]
        self.pending.clear()
        self._event("experiment", "snapshot.restored", {"tick": self.tick_count})
