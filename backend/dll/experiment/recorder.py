"""Buffers samples/events/commands from a Simulation and writes them in batches."""

from __future__ import annotations

import logging
import math

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from ..sim.simulation import Simulation
from .db import Command, Event, Experiment, Sample

log = logging.getLogger(__name__)

SAMPLE_EVERY_TICKS = 10  # 10 Hz


class Recorder:
    def __init__(self, sessions: sessionmaker):
        self.sessions = sessions
        self.experiment_id: int | None = None
        self._samples: list[dict] = []
        self._events_written = 0
        self._commands_written = 0
        self._last_pos = None

    def start(self, sim: Simulation, name: str, params: dict | None = None) -> int:
        with self.sessions.begin() as s:
            exp = Experiment(
                name=name,
                seed=sim.seed,
                preset=sim.preset,
                initial_world=sim.initial_world.model_dump(),
                params={"engine": sim.engine, "tick_s": sim.TICK, **(params or {})},
            )
            s.add(exp)
            s.flush()
            self.experiment_id = exp.id
        self._samples.clear()
        self._events_written = 0
        self._commands_written = 0
        self._last_pos = None
        log.info("recording experiment %d (%s)", self.experiment_id, name)
        return self.experiment_id

    def observe(self, sim: Simulation) -> None:
        if self.experiment_id is None or sim.tick_count % SAMPLE_EVERY_TICKS:
            return
        st = sim.body.state()
        speed = 0.0
        if self._last_pos is not None:
            speed = math.dist(st.position[:2], self._last_pos) / (SAMPLE_EVERY_TICKS * sim.TICK)
        self._last_pos = st.position[:2].copy()
        o, c = sim.organism, sim.last_cmd
        self._samples.append(
            dict(
                experiment_id=self.experiment_id,
                t=round(sim.time, 3),
                x=float(st.position[0]),
                y=float(st.position[1]),
                heading=st.heading,
                speed=speed,
                energy=o.energy,
                health=o.health,
                forward=c.forward,
                turn=c.turn,
                proboscis=c.proboscis,
                total_spikes=sim.total_spikes_tick,
                motor={k: round(v, 2) for k, v in sim.motor_rates.items()},
                sensory={k: round(v, 2) for k, v in sim.sensory_rates.items() if v > 0},
            )
        )

    def flush(self, sim: Simulation, final: bool = False) -> None:
        if self.experiment_id is None:
            return
        events = sim.events[self._events_written :]
        cmds = sim.command_log[self._commands_written :]
        with self.sessions.begin() as s:
            if self._samples:
                s.bulk_insert_mappings(Sample, self._samples)
            if events:
                s.bulk_insert_mappings(Event, [dict(experiment_id=self.experiment_id, **e.to_dict()) for e in events])
            if cmds:
                s.bulk_insert_mappings(
                    Command,
                    [dict(experiment_id=self.experiment_id, tick=c["tick"], payload={k: v for k, v in c.items() if k != "tick"}) for c in cmds],
                )
            exp = s.get(Experiment, self.experiment_id)
            exp.duration_s = sim.time
            exp.summary = summarize(sim)
            if final:
                exp.status = "finished"
        self._samples.clear()
        self._events_written = len(sim.events)
        self._commands_written = len(sim.command_log)

    def mark_branch(self, sim: Simulation) -> None:
        """After a snapshot restore, history up to the snapshot belongs to the parent run."""
        self._events_written = len(sim.events)
        self._commands_written = len(sim.command_log)


def summarize(sim: Simulation) -> dict:
    o = sim.organism
    names = [e.name for e in sim.events]
    return {
        "alive": o.alive,
        "survival_s": round(o.age, 2),
        "death_cause": o.death_cause,
        "food_eaten": round(o.food_eaten, 2),
        "distance_mm": round(o.distance, 2),
        "energy": round(o.energy, 2),
        "health": round(o.health, 2),
        "feeding_bouts": names.count("feeding:on"),
        "collisions": names.count("collision"),
        "turn_left_bouts": names.count("turn_left:on"),
        "turn_right_bouts": names.count("turn_right:on"),
        "escapes": names.count("escape_takeoff"),
        "commands": len(sim.command_log),
    }


def load_commands(sessions: sessionmaker, experiment_id: int) -> list[dict]:
    with sessions() as s:
        rows = s.scalars(select(Command).where(Command.experiment_id == experiment_id).order_by(Command.id)).all()
        return [{"tick": r.tick, **r.payload} for r in rows]
