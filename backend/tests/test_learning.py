"""Associative learning in the mushroom body (experimental module, see learning/mushroom_body.py)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dll.environment.world import WorldConfig  # noqa: E402
from dll.sim.simulation import Simulation  # noqa: E402
from scripts.conditioning import run_protocol  # noqa: E402


@pytest.fixture(scope="module")
def protocol_results(connectome):
    return {learn: run_protocol(connectome, seed=0, trials=4, learn=learn, test_reps=1) for learn in (True, False)}


def test_odour_specific_memory_in_mbons(protocol_results):
    """Pairing odour A with sugar depresses MBON responses to A, not to unpaired odour B."""
    r = protocol_results[True]
    ratio_a = r["mbon_A_post"] / r["mbon_A_pre"]
    ratio_b = r["mbon_B_post"] / r["mbon_B_pre"]
    assert r["mbon_A_pre"] > 300  # odour A drives the mushroom body output (ln_std preset)
    assert ratio_a < 0.7
    assert ratio_b > 0.8
    assert r["mb"]["depressed"] > 1000


def test_no_memory_without_plasticity(protocol_results):
    r = protocol_results[False]
    assert 0.85 < r["mbon_A_post"] / r["mbon_A_pre"] < 1.15
    assert r["mb"]["depressed"] == 0


def test_feeding_releases_dopamine_and_changes_synapses(connectome):
    sim = Simulation(connectome, seed=4, world=WorldConfig())
    sim.submit({"op": "drive.set", "drive_base_hz": 0.0, "drive_hunger_hz": 0.0})
    sim.submit({"op": "learning.set", "plasticity": True})
    sim.run(0.1)
    lab = sim.body.site_pos("labellum")
    sim.submit({"op": "world.add", "kind": "food", "object": {"x": float(lab[0]), "y": float(lab[1]), "radius": 2.0}})
    sim.run(1.0)
    names = [e.name for e in sim.events]
    assert "dopamine.reward:on" in names
    assert sim.plasticity.dopamine.max() > 0 or sim.plasticity.n_updates > 0
    sim.reset()
    assert sim.plasticity.summary()["depressed"] == 0  # reset clears memory


@pytest.mark.slow
def test_learned_odour_approach_is_associative():
    """Odour+sugar training -> approach to the odour alone; odourless sugar does not."""
    from scripts.learned_approach import run

    trained = run(("trained", 0, 20.0, "ln_std"))
    unpaired = run(("unpaired_sugar", 0, 20.0, "ln_std"))
    assert trained["valence_end"] > 0.5
    assert trained["mean_dist"] < 7.0 and trained["frac_near"] > 0.2
    assert unpaired["mean_dist"] > 12.0 and unpaired["frac_near"] == 0.0
