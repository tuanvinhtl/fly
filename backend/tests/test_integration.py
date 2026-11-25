"""End-to-end sensory -> brain -> motor -> body tests on the full 138,639-neuron model.

These are the core claims of the MVP: neural activity causally changes movement, and
movement/environment feed back into the brain. Causality is shown by silencing the
neurons in question and observing that the behaviour disappears.
"""

import math

import numpy as np
import pytest

from dll.environment.world import Food, WorldConfig
from dll.experiment.db import make_engine, make_sessionmaker
from dll.experiment.recorder import Recorder, load_commands
from dll.experiment.replay import compare_trajectories, rerun
from dll.sim.simulation import Simulation


@pytest.fixture(scope="module")
def sim(connectome):
    return Simulation(connectome, preset="al_std", seed=7, world=WorldConfig())


def _fresh(sim, seed=7, world=None, drive=None):
    sim.reset(seed=seed, world=world or WorldConfig())
    if drive is not None:
        sim.submit({"op": "drive.set", "drive_base_hz": drive, "drive_hunger_hz": 0.0})
    return sim


def test_feeding_loop_requires_mn9(sim):
    """Food contact -> sugar GRNs -> MN9 -> proboscis -> ingestion -> energy."""
    _fresh(sim, drive=0.0)
    lab = sim.body.site_pos("labellum")
    sim.submit({"op": "world.add", "kind": "food", "object": {"id": "f", "x": float(lab[0]), "y": float(lab[1]), "radius": 2.0}})
    sim.submit({"op": "organism.set", "energy": 40})
    sim.run(1.5)
    names = [e.name for e in sim.events]
    assert sim.motor_rates["mn_proboscis_left"] + sim.motor_rates["mn_proboscis_right"] > 20
    assert "feeding:on" in names
    assert sim.organism.energy > 40

    # Same situation with MN9 silenced: no proboscis extension, no ingestion.
    _fresh(sim, drive=0.0)
    sim.submit({"op": "brain.silence", "cell_types": ["CB0701"]})
    sim.submit({"op": "world.add", "kind": "food", "object": {"id": "f", "x": float(lab[0]), "y": float(lab[1]), "radius": 2.0}})
    sim.submit({"op": "organism.set", "energy": 40})
    sim.run(1.5)
    assert sim.organism.food_eaten == 0
    assert "feeding:on" not in [e.name for e in sim.events]


def test_locomotion_requires_dnp09(sim):
    """Internal drive acts on DNp09; silencing DNp09 must abolish walking."""
    _fresh(sim, drive=60.0)
    sim.run(2.0)
    control = sim.organism.distance
    _fresh(sim, drive=60.0)
    sim.submit({"op": "brain.silence", "cell_types": ["DNp09"]})
    sim.run(2.0)
    assert control > 5.0
    assert sim.organism.distance < 0.2 * control


def test_head_collision_is_sensed(sim):
    """Placing an obstacle against the left of the head fires left head-bristle input."""
    _fresh(sim, drive=0.0)
    sim.body.reset(x=0.0, y=0.0, heading=0.0)
    sim.run(0.2)
    head = sim.body.data.geom_xpos[sim.body.head_geom]
    sim.submit({"op": "world.add", "kind": "obstacles",
                "object": {"id": "wall", "x": float(head[0]), "y": float(head[1]) + 0.42, "sx": 0.6, "sy": 0.15, "height": 2.0}})
    sim.tick()
    assert sim.sensory_rates["touch_head_left"] > 0
    assert sim.sensory_rates["touch_head_right"] == 0
    assert any(e.name == "touch_head_left:on" for e in sim.events)


def _turn_after_touch(sim, silence=None):
    _fresh(sim, drive=0.0)
    sim.body.reset(x=0.0, y=0.0, heading=0.0)
    if silence:
        sim.submit({"op": "brain.silence", "cell_types": silence})
    sim.run(0.2)
    prev = sim.body.state().heading
    sim.submit({"op": "stimulus", "population": "touch_head_left", "rate_hz": 150.0, "duration_s": 1.0})
    peak_l = peak_r = dh = 0.0
    for _ in range(100):
        sim.tick()
        peak_l = max(peak_l, sim.motor_rates["dn_steer_high_left"])
        peak_r = max(peak_r, sim.motor_rates["dn_steer_high_right"])
        h = sim.body.state().heading
        dh += math.atan2(math.sin(h - prev), math.cos(h - prev))  # unwrapped cumulative rotation
        prev = h
    return peak_l, peak_r, dh


def test_left_head_touch_turns_body_left_via_dna02(sim):
    """Head bristles (left) -> connectome -> DNa02 (left) -> VNC -> left turn of the body."""
    peak_l, peak_r, dh = _turn_after_touch(sim)
    assert peak_l > 8 and peak_l > 2 * peak_r
    assert dh > math.radians(10)
    _, _, dh_silenced = _turn_after_touch(sim, silence=["DNa02", "DNa01"])
    assert abs(dh_silenced) < math.radians(5)


def test_same_seed_same_trajectory(connectome):
    trajs = []
    for _ in range(2):
        s = Simulation(connectome, preset="al_std", seed=11, world=WorldConfig(food=[Food(x=5, y=5)]))
        s.run(1.0)
        trajs.append(s.body.get_state_vector())
    np.testing.assert_array_equal(trajs[0], trajs[1])


def test_replay_from_command_log_is_exact(connectome, tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'db.sqlite3'}")
    sessions = make_sessionmaker(engine)
    world = WorldConfig(food=[Food(x=6, y=-3)])
    s = Simulation(connectome, preset="al_std", seed=5, world=world)
    rec = Recorder(sessions)
    exp_id = rec.start(s, "replay test")
    traj = []
    for k in range(100):
        if k == 20:
            s.submit({"op": "stimulus", "population": "touch_head_right", "rate_hz": 150, "duration_s": 0.3})
        if k == 50:
            s.submit({"op": "world.add", "kind": "lights", "object": {"x": 0, "y": 10, "intensity": 3}})
        s.tick()
        rec.observe(s)
        if s.tick_count % 10 == 0:
            p = s.body.state().position
            traj.append((round(s.time, 3), float(p[0]), float(p[1])))
    rec.flush(s, final=True)
    cmds = load_commands(sessions, exp_id)
    assert [c["op"] for c in cmds] == ["stimulus", "world.add"]
    _, traj2 = rerun(connectome, 5, "al_std", world.model_dump(), cmds, 1.0)
    cmp = compare_trajectories(traj, traj2)
    assert cmp["identical"], cmp


def test_snapshot_restore_reproduces_future(connectome):
    s = Simulation(connectome, preset="al_std", seed=3)
    s.run(0.5)
    snap = s.snapshot()
    s.run(0.5)
    expected = s.body.get_state_vector()
    s.restore(snap)
    s.run(0.5)
    np.testing.assert_allclose(s.body.get_state_vector(), expected, atol=1e-9)


def test_environment_change_mid_run_reaches_brain(sim):
    """Raising temperature changes thermosensory drive and network activity while running."""
    _fresh(sim, drive=0.0)
    sim.run(0.3)
    before = sim.total_spikes_tick
    sim.submit({"op": "world.ambient", "temperature": 37.0})
    sim.run(0.3)
    assert sim.sensory_rates["temp_hot_left"] > 50
    assert sim.total_spikes_tick > before
    assert math.isclose(sim.world.temperature(0, 0), 37.0)


def _loom_trial(sim, silence=None):
    _fresh(sim, drive=0.0)
    if silence:
        sim.submit({"op": "brain.silence", "cell_types": silence})
    sim.run(0.3)
    h = sim.body.data.geom_xpos[sim.body.head_geom]
    sim.submit({"op": "world.add", "kind": "threats", "object": {"id": "hawk", "x": float(h[0]) + 8, "y": float(h[1]) + 6, "z": 15, "speed": 40}})
    zmax = 0.0
    for _ in range(400):
        sim.tick()
        zmax = max(zmax, float(sim.body.state().position[2]))
    return zmax, [e.name for e in sim.events]


def test_looming_triggers_giant_fiber_escape_flight(sim):
    """Approaching object -> LC4/LPLC2 -> (connectome) -> DNp01 giant fiber -> take-off."""
    zmax, names = _loom_trial(sim)
    assert "vision_looming_left:on" in names or "vision_looming_right:on" in names
    assert "takeoff" in names
    assert zmax > 2.5
    assert "predator.strike" not in names
    assert "landing" in names  # myogenic flight drive decays and the fly lands
    assert sim.organism.health == 100


def test_escape_requires_giant_fiber(sim):
    zmax, names = _loom_trial(sim, silence=["DNp01"])
    assert "takeoff" not in names
    assert zmax < 1.5
    assert "predator.strike" in names
    assert sim.organism.health < 100
