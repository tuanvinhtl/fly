import math

import numpy as np

from dll.body.body import FlyBody
from dll.environment.world import Food, Light, Obstacle, World, WorldConfig, Zone
from dll.sensory.sensors import SensoryEngine


def test_odor_decays_with_distance_and_depletion():
    w = World(WorldConfig(food=[Food(x=0, y=0, odor=1.0, odor_sigma=10)]))
    assert w.odor(0, 0) > w.odor(5, 0) > w.odor(20, 0) > 0
    w.cfg.food[0].amount = 0
    assert w.odor(0, 0) == 0


def test_temperature_zone_profile():
    w = World(WorldConfig(ambient_temperature=25, zones=[Zone(kind="temperature", x=0, y=0, radius=5, temperature=40)]))
    assert w.temperature(0, 0) == 40
    assert 25 < w.temperature(6, 0) < 40
    assert w.temperature(20, 0) == 25


def test_world_edit_operations():
    w = World(WorldConfig())
    w.add("obstacles", Obstacle(id="o1", x=1, y=1))
    assert w.version == 1
    w.update("o1", {"x": 5})
    assert w.cfg.obstacles[0].x == 5 and w.version == 2
    assert w.remove("o1") == "obstacles"


def test_sensors_lateralised_odor_and_light():
    body = FlyBody()  # at origin facing +x; left = +y
    world = World(WorldConfig(food=[Food(x=2, y=6, odor_sigma=4)], lights=[Light(x=0, y=10, intensity=3)]))
    r = SensoryEngine().read(world, body)
    assert r.physical["odor_left"] > r.physical["odor_right"]
    assert r.rates["odor_food_left"] > r.rates["odor_food_right"] > 0
    assert r.rates["light_ocellar_left"] > r.rates["light_ocellar_right"]


def test_taste_only_on_contact():
    body = FlyBody()
    lab = body.site_pos("labellum")
    on = World(WorldConfig(food=[Food(x=float(lab[0]), y=float(lab[1]), radius=1.0)]))
    off = World(WorldConfig(food=[Food(x=20, y=20, radius=1.0)]))
    eng = SensoryEngine()
    assert eng.read(on, body).rates["taste_sugar_left"] > 50
    assert eng.read(off, body).rates["taste_sugar_left"] == 0


def test_head_collision_drives_ipsilateral_touch():
    body = FlyBody([("box", 0.75, 0.42, 0.2, 0.15, 2.0)])  # box overlapping the head's left side
    # contacts are computed by mj_forward at reset; stepping would push the light fly away
    r = SensoryEngine().read(World(WorldConfig()), body)
    assert r.rates["touch_head_left"] > 0
    assert r.rates["touch_head_right"] == 0


def test_external_stimulation_adds_rate():
    eng = SensoryEngine()
    eng.external["mech_jo_left"] = 100.0
    r = eng.read(World(WorldConfig()), FlyBody())
    assert r.rates["mech_jo_left"] == 100.0
    assert math.isclose(r.rates["mech_jo_right"], 0.0)
    assert np.isfinite(list(r.rates.values())).all()
