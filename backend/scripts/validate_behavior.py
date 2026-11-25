"""Controlled behavioural experiments on the closed loop (results quoted in RESEARCH.md §6).

    python -m scripts.validate_behavior [--seconds 30] [--seeds 5] [--procs 3]

Conditions (same seeds, same world geometry, only one factor changed):
  odor_on     food 10 mm away emitting odour
  odor_off    identical food with odour strength 0 (taste still works on contact)
  dnp09_off   odor_on + DNp09 silenced (locomotion control)
Metrics: mean / minimum distance to food, time within 4 mm, food eaten, distance walked,
fraction of time turning left vs right.
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
from dll.environment.world import Food, WorldConfig  # noqa: E402
from dll.sim.simulation import Simulation  # noqa: E402

FOOD_XY = (10.0, 0.0)


def run(args: tuple[str, int, float, str]) -> dict:
    cond, seed, seconds, preset = args
    logging.basicConfig(level=logging.WARNING)
    s = load_settings()
    c = load_connectome(s.raw_dir, s.cache_dir)
    odor = 0.0 if cond == "odor_off" else 1.0
    world = WorldConfig(food=[Food(id="f", x=FOOD_XY[0], y=FOOD_XY[1], radius=2.5, odor=odor, odor_sigma=12, amount=1000)])
    sim = Simulation(c, preset=preset, seed=seed, world=world)
    if cond == "dnp09_off":
        sim.submit({"op": "brain.silence", "cell_types": ["DNp09"]})
    d, left, right = [], 0, 0
    for _ in range(int(seconds / sim.TICK)):
        sim.tick()
        p = sim.body.state().position
        d.append(math.hypot(p[0] - FOOD_XY[0], p[1] - FOOD_XY[1]))
        left += sim.last_cmd.turn > 0.2
        right += sim.last_cmd.turn < -0.2
    n = len(d)
    return {
        "condition": cond, "seed": seed, "mean_dist": float(np.mean(d)), "min_dist": float(np.min(d)),
        "frac_near": float(np.mean(np.array(d) < 4.0)), "food_eaten": sim.organism.food_eaten,
        "walked": sim.organism.distance, "turn_left": left / n, "turn_right": right / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--preset", default="al_std")
    a = ap.parse_args()
    jobs = [(c, s, a.seconds, a.preset) for c in ("odor_on", "odor_off", "dnp09_off") for s in range(a.seeds)]
    with Pool(a.procs) as pool:
        rows = pool.map(run, jobs)
    for cond in ("odor_on", "odor_off", "dnp09_off"):
        rs = [r for r in rows if r["condition"] == cond]
        agg = {k: (np.mean([r[k] for r in rs]), np.std([r[k] for r in rs])) for k in rs[0] if k not in ("condition", "seed")}
        print(cond.ljust(10), "  ".join(f"{k}={m:.2f}±{sd:.2f}" for k, (m, sd) in agg.items()))
    if a.out:
        a.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
