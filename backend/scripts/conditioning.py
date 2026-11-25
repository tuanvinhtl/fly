"""Appetitive olfactory conditioning on the full-brain model ("tethered fly" protocol).

    python -m scripts.conditioning [--seeds 3] [--trials 6] [--out results.json]

Protocol (after Tully & Quinn 1985 / differential conditioning):
  pre-test : odour A 1 s, rest 1 s, odour B 1 s, rest 1 s
  training : trials x [ A + sugar 1 s (CS+), rest 1 s, B alone 1 s (CS-), rest 1 s ]
  post-test: as pre-test
Conditions: "learning" (plasticity on) vs "control" (same stimuli, plasticity off).
Readouts: MBON population response and descending-neuron responses to A and B.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dll.brain.connectome import Connectome, load_connectome  # noqa: E402
from dll.brain.presets import build_brain  # noqa: E402
from dll.config import load_settings  # noqa: E402
from dll.learning.mushroom_body import MushroomBodyPlasticity  # noqa: E402

ODOR_A = ["ORN_DM2", "ORN_DM3", "ORN_DM1", "ORN_DM4", "ORN_VA2", "ORN_DP1m"]
ODOR_B = ["ORN_DL1", "ORN_VA3", "ORN_DL5", "ORN_VM2", "ORN_VC3", "ORN_DA3"]
ODOR_HZ, SUGAR_HZ = 80.0, 100.0
TICK = 0.01


def sel(c: Connectome, col: str, values: list[str]) -> np.ndarray:
    return np.flatnonzero(np.array(c.annotations[col].isin(values)))


def run_protocol(c: Connectome, seed: int, trials: int, learn: bool, preset: str = "ln_std", test_reps: int = 3) -> dict:
    brain = build_brain(c, preset, seed=seed)
    mb = MushroomBodyPlasticity(c)
    A, B = sel(c, "cell_type", ODOR_A), sel(c, "cell_type", ODOR_B)
    sugar = sel(c, "cell_sub_class", ["sugar"])
    dn = np.flatnonzero(np.array(c.annotations["super_class"] == "descending"))
    dn_types = np.array(c.annotations["cell_type"])[dn]

    def phase(seconds: float, odor: np.ndarray | None, reward: bool, plastic: bool = False) -> np.ndarray:
        counts = np.zeros(c.n_neurons)
        for _ in range(int(round(seconds / TICK))):
            parts = []
            if odor is not None:
                parts.append((odor, ODOR_HZ))
            if reward:
                parts.append((sugar, SUGAR_HZ))
                parts += mb.reinforcement_inputs(SUGAR_HZ, 0.0)
            if parts:
                brain.set_input_rates(np.concatenate([i for i, _ in parts]), np.concatenate([np.full(i.size, r) for i, r in parts]))
            else:
                brain.set_input_rates(np.zeros(0, dtype=np.int64), np.zeros(0))
            res = brain.step(int(round(TICK / brain.dt)))
            idx, cnt = np.unique(res.flat, return_counts=True)
            counts[idx] += cnt
            if learn and plastic:
                mb.update(brain, idx, cnt, TICK)
        return counts / seconds

    def test(reps: int = 3) -> dict:
        # plasticity off during tests so pre-test measures the naive network
        ra = rb = 0.0
        for _ in range(reps):
            ra = ra + phase(1.0, A, False) / reps
            phase(1.0, None, False)
            rb = rb + phase(1.0, B, False) / reps
            phase(1.0, None, False)
        return {"A": ra, "B": rb}

    pre = test(test_reps)
    for _ in range(trials):
        phase(1.0, A, True, plastic=True)
        phase(1.0, None, False, plastic=True)
        phase(1.0, B, False, plastic=True)
        phase(1.0, None, False, plastic=True)
    post = test(test_reps)
    out = {"seed": seed, "learn": learn, "mb": mb.summary()}
    motor = {t: sel(c, "cell_type", [t]) for t in ("DNp09", "DNa02", "DNa01", "MDN", "DNp01", "CB0701")}
    for od in ("A", "B"):
        out[f"motor_{od}"] = {t: (round(float(pre[od][i].mean()), 1), round(float(post[od][i].mean()), 1)) for t, i in motor.items()}
        out[f"dnvec_{od}_pre"] = pre[od][dn].tolist()
        out[f"dnvec_{od}_post"] = post[od][dn].tolist()
    for od in ("A", "B"):
        m0, m1 = pre[od][mb.mbon], post[od][mb.mbon]
        out[f"mbon_{od}_pre"] = float(m0.sum())
        out[f"mbon_{od}_post"] = float(m1.sum())
        d0, d1 = pre[od][dn], post[od][dn]
        out[f"dn_{od}_change"] = float(np.abs(d1 - d0).sum())
        big = np.argsort(-np.abs(d1 - d0))[:5]
        out[f"dn_{od}_top"] = [(str(dn_types[i]), round(float(d0[i]), 1), round(float(d1[i]), 1)) for i in big]
        mtypes = np.array(c.annotations["cell_type"])[mb.mbon]
        mbig = np.argsort(-np.abs(m1 - m0))[:5]
        out[f"mbon_{od}_top"] = [(str(mtypes[i]), round(float(m0[i]), 1), round(float(m1[i]), 1)) for i in mbig]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--preset", default="ln_std")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    s = load_settings()
    c = load_connectome(s.raw_dir, s.cache_dir)
    rows = []
    for learn in (True, False):
        for seed in range(a.seeds):
            r = run_protocol(c, seed, a.trials, learn, a.preset)
            rows.append(r)
            tag = "learning" if learn else "control "
            print(
                f"{tag} seed {seed}: MBON pop. response A {r['mbon_A_pre']:.0f}->{r['mbon_A_post']:.0f} Hz, "
                f"B {r['mbon_B_pre']:.0f}->{r['mbon_B_post']:.0f} Hz | DN change A {r['dn_A_change']:.0f}, B {r['dn_B_change']:.0f} | {r['mb']}"
            )
    for learn in (True, False):
        rs = [r for r in rows if r["learn"] == learn]
        ra = np.mean([r["mbon_A_post"] / max(1e-9, r["mbon_A_pre"]) for r in rs])
        rb = np.mean([r["mbon_B_post"] / max(1e-9, r["mbon_B_pre"]) for r in rs])
        print(f"{'learning' if learn else 'control'}: MBON response post/pre  A (CS+) {ra:.2f}   B (CS-) {rb:.2f}")
    # behavioural expression: does training change the DN response to A more than to B,
    # beyond what the control (no plasticity) shows?
    for learn in (True, False):
        rs = [r for r in rows if r["learn"] == learn]
        dA = np.mean([np.abs(np.subtract(r["dnvec_A_post"], r["dnvec_A_pre"])).sum() for r in rs])
        dB = np.mean([np.abs(np.subtract(r["dnvec_B_post"], r["dnvec_B_pre"])).sum() for r in rs])
        print(f"{'learning' if learn else 'control'}: summed |change| of DN responses  A {dA:.0f} Hz  B {dB:.0f} Hz")
    for r in rows:
        print("learning" if r["learn"] else "control ", r["seed"], "motor DNs to A (pre, post):", r["motor_A"])
    ex = next(r for r in rows if r["learn"])
    print("example MBON changes to A:", ex["mbon_A_top"])
    print("example DN changes to A:", ex["dn_A_top"])
    if a.out:
        slim = [{k: v for k, v in r.items() if not k.startswith("dnvec")} for r in rows]
        a.out.write_text(json.dumps(slim, indent=1))


if __name__ == "__main__":
    main()
