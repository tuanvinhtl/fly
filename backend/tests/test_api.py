import dataclasses
import time

import pytest
from fastapi.testclient import TestClient

from dll.api.app import create_app
from dll.config import load_settings


@pytest.fixture(scope="module")
def client(connectome, tmp_path_factory):
    db = tmp_path_factory.mktemp("db") / "api.sqlite3"
    settings = dataclasses.replace(load_settings(), database_url=f"sqlite:///{db}")
    with TestClient(create_app(settings, connectome)) as c:
        yield c


def test_status_and_brain_info(client):
    assert client.get("/api/health").json() == {"ok": True}
    info = client.get("/api/brain/info").json()
    assert info["n_neurons"] == 138_639
    assert {p["evidence"] for p in info["sensory"]} <= {"validated", "literature", "approximation"}
    assert any(p["key"] == "mn_proboscis_left" for p in info["motor"])


def test_layout_binary(client):
    body = client.get("/api/brain/layout").content
    n = int.from_bytes(body[:4], "little")
    assert n == 138_639 and len(body) == 4 + n * 12 + n


def test_step_world_edit_and_stimulus(client):
    r = client.post("/api/sim/reset", json={"seed": 1, "name": "api test", "default_world": True}).json()
    exp_id = r["experiment_id"]
    client.post("/api/sim/step", json={"ticks": 20})
    assert client.get("/api/status").json()["tick"] == 20
    assert client.post("/api/world/food", json={"x": 3, "y": 3, "radius": 2}).status_code == 200
    world = client.get("/api/world").json()["world"]
    food_id = next(f["id"] for f in world["food"] if f["x"] == 3)
    assert client.patch(f"/api/world/{food_id}", json={"amount": 5}).status_code == 200
    assert client.delete(f"/api/world/{food_id}").status_code == 200
    assert client.delete("/api/world/nope").status_code == 404
    assert client.post("/api/stimulus", json={"population": "bogus", "rate_hz": 10}).status_code == 400
    assert client.post("/api/stimulus", json={"population": "taste_sugar_left", "rate_hz": 120, "duration_s": 0.5}).status_code == 200
    client.post("/api/sim/step", json={"ticks": 30})
    events = client.get(f"/api/experiments/{exp_id}/events").json()
    assert any(e["name"] == "stimulus" for e in events)
    assert any(e["name"] == "world.add" for e in events)
    samples = client.get(f"/api/experiments/{exp_id}/samples").json()
    assert len(samples) >= 5


def test_run_pause_and_websocket(client):
    client.post("/api/sim/reset", json={"seed": 2})
    client.post("/api/sim/speed", json={"multiplier": 0})
    client.post("/api/sim/start")
    with client.websocket_connect("/ws") as ws:
        kinds = set()
        frame = None
        for _ in range(10):
            msg = ws.receive_json()
            kinds.add(msg["type"])
            if msg["type"] == "frame":
                frame = msg
        assert {"world", "frame"} <= kinds
        assert "thorax" in frame["body"]["geoms"]
        assert len(frame["brain"]["clusters"]) > 10
    time.sleep(0.5)
    client.post("/api/sim/pause")
    t = client.get("/api/status").json()["t"]
    assert t > 0.05
    time.sleep(0.3)
    assert client.get("/api/status").json()["t"] == t


def test_neuron_inspect_and_search(client):
    hits = client.get("/api/brain/search", params={"q": "DNa02"}).json()
    assert len(hits) == 2
    n = client.get(f"/api/brain/neuron/{hits[0]['idx']}").json()
    assert n["cell_type"] == "DNa02" and n["inputs"] and n["outputs"]
    assert client.get("/api/brain/neuron/99999999").status_code == 404


def test_snapshot_restore_and_replay_check(client):
    exp = client.post("/api/sim/reset", json={"seed": 4, "name": "replay"}).json()["experiment_id"]
    client.post("/api/sim/step", json={"ticks": 50})
    client.post("/api/stimulus", json={"population": "touch_head_right", "rate_hz": 150, "duration_s": 0.2})
    client.post("/api/sim/step", json={"ticks": 50})
    snap = client.post("/api/snapshots", json={"label": "t1"}).json()
    client.post("/api/sim/step", json={"ticks": 10})
    rep = client.post(f"/api/experiments/{exp}/replay-check", json={"max_seconds": 1.0}).json()
    assert rep["identical"], rep
    restored = client.post(f"/api/snapshots/{snap['id']}/restore").json()
    assert restored["tick"] == snap["tick"]
    assert client.get("/api/status").json()["tick"] == snap["tick"]
