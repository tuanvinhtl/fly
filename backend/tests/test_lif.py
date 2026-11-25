import dataclasses

import numpy as np
import pytest

from dll.brain.lif import LIFBrain, LIFParams
from dll.brain.presets import build_brain


def _run(conn, idx, engine, params=None, seed=3, dur=0.3, rate=150.0):
    b = LIFBrain(conn, params or LIFParams(), seed=seed, engine=engine)
    b.set_input_rates(idx, np.full(idx.size, rate))
    for _ in range(int(dur / 0.01)):
        b.step(100)
    return b


def test_numba_matches_numpy_reference_exactly(sugar_subnet):
    sub, inp, _ = sugar_subnet
    a = _run(sub, inp, "numba")
    b = _run(sub, inp, "numpy")
    assert a.spike_count.sum() > 0
    np.testing.assert_array_equal(a.spike_count, b.spike_count)
    np.testing.assert_allclose(a.v, b.v, atol=1e-12)


def test_active_set_matches_dense_integration(sugar_subnet):
    sub, inp, _ = sugar_subnet
    dense = _run(sub, inp, "numba", LIFParams(rest_tol=0.0))
    sparse = _run(sub, inp, "numba", LIFParams(rest_tol=1e-5))
    np.testing.assert_array_equal(dense.spike_count, sparse.spike_count)


def test_depression_with_u_zero_is_the_published_model(sugar_subnet):
    sub, inp, _ = sugar_subnet
    a = _run(sub, inp, "numba", LIFParams(std_U=0.0))
    b = _run(sub, inp, "numba", LIFParams(std_U=0.0, std_tau_rec=0.01))
    np.testing.assert_array_equal(a.spike_count, b.spike_count)


def test_no_input_means_no_activity(sugar_subnet):
    sub, _, _ = sugar_subnet
    b = LIFBrain(sub, seed=0)
    b.step(1000)
    assert b.spike_count.sum() == 0


def test_seed_determinism_and_seed_dependence(sugar_subnet):
    sub, inp, _ = sugar_subnet
    a, b, c = _run(sub, inp, "numba", seed=1), _run(sub, inp, "numba", seed=1), _run(sub, inp, "numba", seed=2)
    np.testing.assert_array_equal(a.spike_count, b.spike_count)
    assert not np.array_equal(a.spike_count, c.spike_count)


def test_snapshot_restore_continues_identically(sugar_subnet):
    sub, inp, _ = sugar_subnet
    b = _run(sub, inp, "numba", dur=0.1)
    snap, counts0 = b.snapshot(), b.spike_count.copy()
    b.step(500)
    expected = b.spike_count.copy()
    b.restore(snap)
    np.testing.assert_array_equal(b.spike_count, counts0)
    b.step(500)
    np.testing.assert_array_equal(b.spike_count, expected)


def test_silencing_blocks_downstream(sugar_subnet):
    sub, inp, mn9 = sugar_subnet
    base = _run(sub, inp, "numba")
    b = LIFBrain(sub, seed=3)
    b.silence(inp)  # inputs still spike but transmit nothing
    b.set_input_rates(inp, np.full(inp.size, 150.0))
    b.step(3000)
    assert base.spike_count[mn9].sum() > 0
    assert b.spike_count[mn9].sum() == 0
    assert b.spike_count[inp].sum() > 0


def test_sugar_activates_mn9_full_brain(connectome):
    """Replicates the headline result of Shiu et al. 2024 (sugar GRNs -> MN9)."""
    ann = connectome.annotations
    sugar = np.flatnonzero((ann.cell_sub_class == "sugar").to_numpy() & (ann.side == "left").to_numpy())
    mn9 = np.flatnonzero((ann.cell_type == "CB0701").to_numpy())
    for preset in ("exact", "al_std"):
        b = build_brain(connectome, preset, seed=0)
        b.set_input_rates(sugar, np.full(sugar.size, 100.0))
        b.step(5000)
        assert (b.spike_count[mn9] / 0.5 > 20).all(), preset


def _persistent_spikes(brain, idx):
    brain.set_input_rates(idx, np.full(idx.size, 100.0))
    brain.step(5000)
    brain.set_input_rates(np.zeros(0, int), np.zeros(0))
    brain.step(5000)  # 0.5 s after stimulus offset
    before = brain.spike_count.sum()
    brain.step(2000)
    return brain.spike_count.sum() - before


def test_exact_model_olfactory_runaway_and_al_std_fix(connectome):
    """Documents the attractor described in RESEARCH.md §4 and that al_std removes it."""
    ann = connectome.annotations
    orn = np.flatnonzero(ann.cell_type.isin(["ORN_DM1", "ORN_DM4"]).to_numpy() & (ann.side == "left").to_numpy())
    exact = build_brain(connectome, "exact", seed=0)
    assert _persistent_spikes(exact, orn) > 10_000
    fixed = build_brain(connectome, "al_std", seed=0)
    assert _persistent_spikes(fixed, orn) < 100


@pytest.mark.slow
def test_matches_brian2_reference(sugar_subnet):
    """Statistical comparison with Brian2 running the published equations."""
    pytest.importorskip("brian2")
    from dll.brain.brian2_ref import run_brian2

    sub, inp, _ = sugar_subnet
    trials, dur = 5, 0.5
    ours = sum(_run(sub, inp, "numba", seed=s, dur=dur).spike_count for s in range(trials)) / (trials * dur)
    ref = sum(run_brian2(sub, inp, 150.0, dur, seed=100 + s) for s in range(trials)) / (trials * dur)
    active = (ours > 2) | (ref > 2)
    assert active.sum() > 20
    r = np.corrcoef(ours[active], ref[active])[0, 1]
    assert r > 0.95, f"rate correlation with Brian2 too low: {r:.3f}"
    assert abs(ours.sum() / ref.sum() - 1) < 0.1


def test_rest_tolerance_dataclass_default():
    assert dataclasses.fields(LIFParams)  # params are a frozen, documented dataclass
    assert LIFParams().rest_tol == 1e-5
