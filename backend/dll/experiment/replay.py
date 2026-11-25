"""Deterministic replay: rebuild a run from (seed, preset, initial world, command log)."""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from ..brain.connectome import Connectome
from ..environment.world import WorldConfig
from ..sim.simulation import Simulation

log = logging.getLogger(__name__)

_DERIVED = {"tick", "n_silenced"}


def rerun(
    connectome: Connectome,
    seed: int,
    preset: str,
    initial_world: dict,
    commands: list[dict],
    duration_s: float,
    engine: str = "numba",
    sample_every: int = 10,
    initial_memory: np.ndarray | None = None,
) -> tuple[Simulation, list[tuple[float, float, float]]]:
    sim = Simulation(connectome, preset=preset, seed=seed, world=WorldConfig.model_validate(initial_world), engine=engine)
    if initial_memory is not None:
        sim.import_memory(initial_memory)
        sim.events.clear()
    by_tick: dict[int, list[dict]] = defaultdict(list)
    for c in commands:
        by_tick[int(c["tick"])].append({k: v for k, v in c.items() if k not in _DERIVED})
    traj = []
    for k in range(int(round(duration_s / sim.TICK))):
        for c in by_tick.get(k, []):
            sim.submit(c)
        sim.tick()
        if sim.tick_count % sample_every == 0:
            p = sim.body.state().position
            traj.append((round(sim.time, 3), float(p[0]), float(p[1])))
    return sim, traj


def compare_trajectories(a: list[tuple[float, float, float]], b: list[tuple[float, float, float]]) -> dict:
    n = min(len(a), len(b))
    if n == 0:
        return {"n": 0, "max_error_mm": None, "identical": False}
    pa, pb = np.array(a[:n])[:, 1:], np.array(b[:n])[:, 1:]
    err = np.linalg.norm(pa - pb, axis=1)
    return {"n": n, "max_error_mm": float(err.max()), "identical": bool(err.max() < 1e-6)}
