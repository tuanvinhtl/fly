"""Level-2 memory: does a memory learned in one experiment carry into another, and fade?

    python -m scripts.memory_retention [--seeds 3] [--procs 2]

Experiment 1 (training): ln_std, plasticity + valence readout, fly held 6 s on scented
sugar food. Its KC->MBON weights are exported.
Experiment 2 (test): a NEW simulation (different seed, position and heading, fresh brain
state). Memory imported or not; learning OFF (no new learning); valence readout ON.
Optional delay with forgetting before the test. Then 30 s free walking with an
odour-only source 10 mm away.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dll.brain.connectome import load_connectome  # noqa: E402
from dll.config import load_settings  # noqa: E402
from dll.environment.world import WorldConfig  # noqa: E402
from dll.sim.simulation import Simulation  # noqa: E402

# condition -> (import memory, forgetting tau (s), delay before test (s))
CONDITIONS = {
    "remembered": (True, 0.0, 0.0),
    "no_memory": (False, 0.0, 0.0),
    "partly_forgotten": (True, 20.0, 10.0),
    "forgotten": (True, 20.0, 60.0),
}


def train(c, seed: int, preset: str) -> np.ndarray:
    sim = Simulation(c, preset=preset, seed=seed, world=WorldConfig())
    sim.submit({"op": "learning.set", "plasticity": True, "valence": True})
    sim.submit({"op": "drive.set", "drive_base_hz": 0.0, "drive_hunger_hz": 0.0})
    sim.run(0.1)
    lab = sim.body.site_pos("labellum")
    sim.submit({"op": "world.add", "kind": "food", "object": {"id": "train", "x": float(lab[0]), "y": float(lab[1]), "radius": 2.0, "odor_sigma": 12.0, "amount": 1000}})
    sim.run(6.0)
    return sim.export_memory()


def run(args: tuple[str, int, float, str]) -> dict:
    cond, seed, test_s, preset = args
    use_mem, tau, delay = CONDITIONS[cond]
    logging.basicConfig(level=logging.WARNING)
    s = load_settings()
    c = load_connectome(s.raw_dir, s.cache_dir)
    memory = train(c, seed, preset) if use_mem else None
    sim = Simulation(c, preset=preset, seed=seed + 100, world=WorldConfig())  # new experiment
    if memory is not None:
        sim.import_memory(memory)
    sim.submit({"op": "learning.set", "plasticity": False, "valence": True, "forget_tau_s": tau})
    depressed_start = int((sim.export_memory() < 0.99).sum())
    if delay > 0:
        sim.submit({"op": "drive.set", "drive_base_hz": 0.0, "drive_hunger_hz": 0.0})
        sim.run(delay)
        sim.submit({"op": "drive.set", "drive_base_hz": 30.0, "drive_hunger_hz": 50.0})
    sim.run(0.05)
    mem_left = float(1.0 - sim.export_memory().mean())
    p = sim.body.state()
    ang = p.heading - 2.0
    sx, sy = float(p.position[0] + 10 * math.cos(ang)), float(p.position[1] + 10 * math.sin(ang))
    sim.submit({"op": "world.add", "kind": "food", "object": {"id": "lure", "x": sx, "y": sy, "sugar": 0.0, "odor": 1.0, "odor_sigma": 12.0, "amount": 1000}})
    d = []
    for _ in range(int(test_s / sim.TICK)):
        sim.tick()
        q = sim.body.state().position
        d.append(math.hypot(q[0] - sx, q[1] - sy))
    return {
        "condition": cond, "seed": seed, "mean_dist": float(np.mean(d)), "frac_near": float(np.mean(np.array(d) < 4.0)),
        "depressed_at_start": depressed_start, "memory_strength_at_test": mem_left,
        "valence_end": float(sim.valence.value),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--test", type=float, default=30.0)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--preset", default="ln_std")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    jobs = [(cond, s, a.test, a.preset) for cond in CONDITIONS for s in range(a.seeds)]
    with Pool(a.procs) as pool:
        rows = pool.map(run, jobs)
    for cond in CONDITIONS:
        rs = [r for r in rows if r["condition"] == cond]
        print(
            cond.ljust(17),
            f"mean_dist={np.mean([r['mean_dist'] for r in rs]):5.2f}±{np.std([r['mean_dist'] for r in rs]):.2f}",
            f"near={np.mean([r['frac_near'] for r in rs]):.2f}",
            f"memory strength at test={np.mean([r['memory_strength_at_test'] for r in rs]):.4f}",
            f"valence_end={[round(r['valence_end'], 2) for r in rs]}",
        )
    if a.out:
        a.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
