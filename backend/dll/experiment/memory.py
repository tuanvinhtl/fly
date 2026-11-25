"""Persistence of mushroom-body memories (level 2: an organism that remembers across experiments)."""

from __future__ import annotations

import logging

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from ..sim.simulation import Simulation
from .db import Memory

log = logging.getLogger(__name__)


def save_memory(sessions: sessionmaker, sim: Simulation, name: str, experiment_id: int | None) -> dict:
    scale = sim.export_memory()
    conn_hash = sim.memory_signature()
    with sessions.begin() as s:
        m = Memory(
            name=name,
            preset=sim.preset,
            source_experiment_id=experiment_id,
            source_t=round(sim.time, 3),
            n_connections=int(scale.size),
            conn_hash=conn_hash,
            depressed=int((scale < 0.99).sum()),
            mean_weight=float(scale.mean()),
            weights=scale.astype(np.float32).tobytes(),
        )
        s.add(m)
        s.flush()
        log.info("saved memory %d (%s): %d depressed synapses", m.id, name, m.depressed)
        return memory_info(m)


def load_memory(sessions: sessionmaker, memory_id: int, expected_hash: str) -> np.ndarray:
    with sessions() as s:
        m = s.get(Memory, memory_id)
        if m is None:
            raise KeyError(f"memory {memory_id} not found")
        if m.conn_hash != expected_hash:
            raise ValueError(f"memory {memory_id} was made for a different connectome / plastic-synapse set")
        return np.frombuffer(m.weights, dtype=np.float32).astype(np.float64)


def list_memories(sessions: sessionmaker) -> list[dict]:
    with sessions() as s:
        return [memory_info(m) for m in s.scalars(select(Memory).order_by(Memory.id.desc())).all()]


def delete_memory(sessions: sessionmaker, memory_id: int) -> None:
    with sessions.begin() as s:
        m = s.get(Memory, memory_id)
        if m is None:
            raise KeyError(f"memory {memory_id} not found")
        s.delete(m)


def memory_info(m: Memory) -> dict:
    return {
        "id": m.id, "name": m.name, "created_at": m.created_at.isoformat(), "preset": m.preset,
        "source_experiment_id": m.source_experiment_id, "source_t": m.source_t,
        "depressed": m.depressed, "mean_weight": round(m.mean_weight, 4), "n_connections": m.n_connections,
    }
