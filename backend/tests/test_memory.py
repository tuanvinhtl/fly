"""Level-2 memory: persistence across experiments, forgetting, replay."""

import math

import numpy as np

from dll.environment.world import WorldConfig
from dll.sim.simulation import Simulation


def _fake_memory(sim: Simulation, depressed: int = 500, w: float = 0.3) -> np.ndarray:
    m = sim.export_memory()
    m[:depressed] = w
    return m


def test_export_import_roundtrip_changes_brain_weights(connectome):
    sim = Simulation(connectome, seed=1, world=WorldConfig())
    mem = _fake_memory(sim)
    sim.import_memory(mem)
    mb = sim.plasticity
    np.testing.assert_allclose(sim.export_memory(), mem)
    np.testing.assert_allclose(sim.brain.weight_scale[mb.conn[:500]], 0.3)
    sim.reset()
    assert (sim.export_memory() == 1.0).all()  # plain reset forgets
    assert (sim.brain.weight_scale[mb.conn] == 1.0).all()


def test_forgetting_is_exponential_and_runs_with_learning_off(connectome):
    sim = Simulation(connectome, seed=1, world=WorldConfig())
    sim.import_memory(_fake_memory(sim, w=0.2))
    sim.submit({"op": "learning.set", "plasticity": False, "forget_tau_s": 10.0})
    sim.run(10.0)
    remaining = (1 - sim.export_memory()[:500]).mean() / 0.8
    assert abs(remaining - math.exp(-1.0)) < 0.02


def test_api_memory_save_load_keep_and_replay(connectome, tmp_path):
    from fastapi.testclient import TestClient

    from dll.api.app import create_app
    from dll.config import load_settings
    import dataclasses

    settings = dataclasses.replace(load_settings(), database_url=f"sqlite:///{tmp_path / 'm.sqlite3'}")
    with TestClient(create_app(settings, connectome)) as c:
        c.post("/api/sim/reset", json={"seed": 3, "preset": "ln_std"})
        runner_sim = c.app.state.runner.sim
        with c.app.state.runner.lock:
            runner_sim.import_memory(_fake_memory(runner_sim))
        saved = c.post("/api/memories", json={"name": "test memory"}).json()
        assert saved["depressed"] == 500
        assert any(m["id"] == saved["id"] for m in c.get("/api/memories").json())
        # new experiment starting from the saved memory
        r = c.post("/api/sim/reset", json={"seed": 9, "memory_id": saved["id"], "name": "with memory"}).json()
        assert r["memory_id"] == saved["id"]
        assert (runner_sim.export_memory() < 0.99).sum() == 500
        # keep_memory carries it on (auto-saved), plain reset clears it
        r2 = c.post("/api/sim/reset", json={"seed": 10, "keep_memory": True}).json()
        assert r2["memory_id"] is not None and (runner_sim.export_memory() < 0.99).sum() == 500
        c.post("/api/sim/step", json={"ticks": 50})
        rep = c.post(f"/api/experiments/{r2['experiment_id']}/replay-check", json={"max_seconds": 0.5}).json()
        assert rep["identical"], rep
        c.post("/api/sim/reset", json={"seed": 11})
        assert (runner_sim.export_memory() < 0.99).sum() == 0
        assert c.post("/api/sim/reset", json={"memory_id": 99999}).status_code == 404
        assert c.delete(f"/api/memories/{saved['id']}").status_code == 200
