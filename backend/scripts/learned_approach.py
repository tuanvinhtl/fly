"""Does a learned odour change behaviour? (closed loop, MB plasticity + valence readout)

    python -m scripts.learned_approach [--seeds 5] [--test 30] [--procs 2] [--preset ln_std]

Each run: 6 s training phase with the fly held still (no locomotor drive), then 30 s test
in which the fly walks freely and an odour-only source (same odour, no sugar) sits 10 mm away.
  trained          : training on sugar food that emits the odour; plasticity + readout on
  naive            : same phases but no food during training;    plasticity + readout on
  trained_noreadout: as trained but the valence readout off (plasticity only)
Metric: mean / minimum distance to the odour source and time within 4 mm.
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

CONDITIONS = ("trained", "naive", "trained_noreadout")


def run(args: tuple[str, int, float, str]) -> dict:
    cond, seed, test_s, preset = args
    logging.basicConfig(level=logging.WARNING)
    s = load_settings()
    c = load_connectome(s.raw_dir, s.cache_dir)
    sim = Simulation(c, preset=preset, seed=seed, world=WorldConfig())
    sim.submit({"op": "learning.set", "plasticity": True, "valence": cond != "trained_noreadout"})
    sim.submit({"op": "drive.set", "drive_base_hz": 0.0, "drive_hunger_hz": 0.0})
    sim.run(0.1)
    lab = sim.body.site_pos("labellum")
    if cond != "naive":
        sim.submit({"op": "world.add", "kind": "food", "object": {"id": "train", "x": float(lab[0]), "y": float(lab[1]), "radius": 2.0, "odor": 1.0, "odor_sigma": 12.0, "amount": 1000}})
    sim.run(6.0)
    if cond != "naive":
        sim.submit({"op": "world.remove", "id": "train"})
    p = sim.body.state()
    # odour-only source 10 mm to the fly's right-rear so it is not initially ahead
    ang = p.heading - 2.0
    sx, sy = float(p.position[0] + 10 * math.cos(ang)), float(p.position[1] + 10 * math.sin(ang))
    sim.submit({"op": "world.add", "kind": "food", "object": {"id": "lure", "x": sx, "y": sy, "radius": 2.0, "sugar": 0.0, "odor": 1.0, "odor_sigma": 12.0, "amount": 1000}})
    sim.submit({"op": "drive.set", "drive_base_hz": 30.0, "drive_hunger_hz": 50.0})
    d, steer = [], []
    for _ in range(int(test_s / sim.TICK)):
        sim.tick()
        q = sim.body.state().position
        d.append(math.hypot(q[0] - sx, q[1] - sy))
        if sim.valence is not None:
            steer.append(sim.valence.last_bias)
    mb = sim.plasticity.summary() if sim.plasticity else {}
    return {
        "condition": cond, "seed": seed, "mean_dist": float(np.mean(d)), "min_dist": float(np.min(d)),
        "frac_near": float(np.mean(np.array(d) < 4.0)), "depressed": mb.get("depressed", 0),
        "valence_end": float(sim.valence.value) if sim.valence is not None else None,
        "mean_abs_steer": float(np.mean(np.abs(steer))) if steer else 0.0,
        "walked": sim.organism.distance,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--test", type=float, default=30.0)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--preset", default="ln_std")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    jobs = [(c, s, a.test, a.preset) for c in CONDITIONS for s in range(a.seeds)]
    with Pool(a.procs) as pool:
        rows = pool.map(run, jobs)
    for cond in CONDITIONS:
        rs = [r for r in rows if r["condition"] == cond]
        keys = ["mean_dist", "min_dist", "frac_near", "depressed", "mean_abs_steer", "walked"]
        print(cond.ljust(18), "  ".join(f"{k}={np.mean([r[k] for r in rs]):.2f}±{np.std([r[k] for r in rs]):.2f}" for k in keys),
              " valence_end=", [round(r["valence_end"], 3) if r["valence_end"] is not None else None for r in rs])
    if a.out:
        a.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
