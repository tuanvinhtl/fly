"""FastAPI application: REST control surface + WebSocket frame stream."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from ..brain.connectome import Connectome, load_connectome
from ..brain.presets import PRESETS
from ..config import Settings, load_settings
from ..environment.world import WorldConfig
from ..experiment.db import Event, Experiment, Sample, make_engine, make_sessionmaker
from ..experiment.memory import delete_memory, list_memories, load_memory, save_memory
from ..experiment.recorder import Recorder, load_commands
from ..experiment.replay import compare_trajectories, rerun
from ..sim.runner import SimulationRunner
from ..sim.simulation import Simulation

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- request models
class ResetRequest(BaseModel):
    seed: int | None = None
    name: str = "experiment"
    preset: Literal["exact", "al_std", "ln_std"] | None = None
    world: WorldConfig | None = None
    default_world: bool = False
    memory_id: int | None = Field(None, description="start with a saved mushroom-body memory")
    keep_memory: bool = Field(False, description="carry the current memory into the new experiment")


class MemorySaveRequest(BaseModel):
    name: str = Field("memory", max_length=200)


class SpeedRequest(BaseModel):
    multiplier: float = Field(..., ge=0, le=16, description="0 = as fast as possible")


class StepRequest(BaseModel):
    ticks: int = Field(1, ge=1, le=1000)


class StimulusRequest(BaseModel):
    population: str
    rate_hz: float = Field(..., ge=0, le=500)
    duration_s: float = Field(1.0, gt=0, le=60)


class SilenceRequest(BaseModel):
    cell_types: list[str] = Field(..., min_length=1)


class AmbientRequest(BaseModel):
    temperature: float | None = Field(None, ge=0, le=50)
    light: float | None = Field(None, ge=0, le=10)


class LearningRequest(BaseModel):
    habituation: bool | None = None
    plasticity: bool | None = None
    valence: bool | None = None
    forget_tau_s: float | None = Field(None, ge=0, le=86400, description="memory decay time constant; 0 = no forgetting")


class OrganismRequest(BaseModel):
    energy: float | None = Field(None, ge=0, le=100)
    health: float | None = Field(None, ge=0, le=100)


class DriveRequest(BaseModel):
    drive_base_hz: float | None = Field(None, ge=0, le=200)
    drive_hunger_hz: float | None = Field(None, ge=0, le=200)


class SnapshotRequest(BaseModel):
    label: str = "snapshot"


class ReplayRequest(BaseModel):
    max_seconds: float = Field(10.0, gt=0, le=120)


def create_app(settings: Settings | None = None, connectome: Connectome | None = None) -> FastAPI:
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        conn = connectome or load_connectome(settings.raw_dir, settings.cache_dir)
        engine = make_engine(settings.database_url)
        sessions = make_sessionmaker(engine)
        sim = Simulation(conn, preset=settings.brain_preset, seed=settings.seed, engine=settings.brain_engine)
        runner = SimulationRunner(sim, Recorder(sessions))
        runner.start_thread()
        app.state.connectome, app.state.sessions, app.state.runner = conn, sessions, runner
        log.info("Digital Life Lab API ready")
        yield
        runner.shutdown()

    app = FastAPI(title="Digital Life Lab", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["*"], allow_headers=["*"])

    def runner() -> SimulationRunner:
        return app.state.runner

    def submit(cmd: dict) -> dict:
        try:
            runner().submit(cmd)
        except (ValueError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"ok": True, "queued": cmd}

    # ------------------------------------------------------------ simulation control
    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/api/status")
    def status() -> dict:
        r = runner()
        with r.lock:
            sim = r.sim
            return {"t": round(sim.time, 3), "tick": sim.tick_count, "running": r.running, "speed": r.speed,
                    "realtime_factor": r.realtime_factor, "experiment_id": r.recorder.experiment_id,
                    "error": r.last_error, "organism": sim.organism.to_dict()}

    @app.post("/api/sim/start")
    @app.post("/api/sim/resume")
    def start() -> dict:
        r = runner()
        if not r.sim.organism.alive:
            raise HTTPException(409, "organism is dead — reset the experiment")
        r.last_error = None
        r.set_running(True)
        return {"running": True}

    @app.post("/api/sim/pause")
    def pause() -> dict:
        runner().set_running(False)
        return {"running": False}

    @app.post("/api/sim/step")
    def step(req: StepRequest) -> dict:
        r = runner()
        r.set_running(False)
        r.step(req.ticks)
        return {"t": r.sim.time}

    @app.post("/api/sim/speed")
    def speed(req: SpeedRequest) -> dict:
        runner().speed = req.multiplier
        return {"speed": req.multiplier}

    @app.post("/api/sim/reset")
    def reset(req: ResetRequest) -> dict:
        from ..environment.world import default_world

        world = req.world or (default_world() if req.default_world else None)
        r = runner()
        memory_id = req.memory_id
        if req.keep_memory and memory_id is None:
            with r.lock:
                info = save_memory(app.state.sessions, r.sim, f"carried from experiment {r.recorder.experiment_id}", r.recorder.experiment_id)
            memory_id = info["id"]
        memory = None
        if memory_id is not None:
            try:
                memory = load_memory(app.state.sessions, memory_id, r.sim.memory_signature())
            except (KeyError, ValueError) as exc:
                raise HTTPException(404 if isinstance(exc, KeyError) else 409, str(exc)) from exc
        params = {"memory_id": memory_id} if memory_id is not None else None
        exp_id = r.reset(seed=req.seed, world=world, name=req.name, preset=req.preset, memory=memory, params=params)
        return {"experiment_id": exp_id, "memory_id": memory_id}

    # ------------------------------------------------------------ memories (level 2)
    @app.get("/api/memories")
    def memories() -> list[dict]:
        return list_memories(app.state.sessions)

    @app.post("/api/memories")
    def save_current_memory(req: MemorySaveRequest) -> dict:
        r = runner()
        with r.lock:
            return save_memory(app.state.sessions, r.sim, req.name, r.recorder.experiment_id)

    @app.delete("/api/memories/{memory_id}")
    def remove_memory(memory_id: int) -> dict:
        try:
            delete_memory(app.state.sessions, memory_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"ok": True}

    # ------------------------------------------------------------ world
    @app.get("/api/world")
    def get_world() -> dict:
        r = runner()
        with r.lock:
            return {"world_seq": r.world_seq, "world": r.sim.world.cfg.model_dump()}

    @app.post("/api/world/{kind}")
    def add_world(kind: Literal["food", "obstacles", "lights", "zones", "threats"], obj: dict[str, Any]) -> dict:
        return submit({"op": "world.add", "kind": kind, "object": obj})

    @app.patch("/api/world/{obj_id}")
    def update_world(obj_id: str, changes: dict[str, Any]) -> dict:
        return submit({"op": "world.update", "id": obj_id, "changes": changes})

    @app.delete("/api/world/{obj_id}")
    def delete_world(obj_id: str) -> dict:
        r = runner()
        with r.lock:
            ids = {o.id for k in ("food", "obstacles", "lights", "zones", "threats") for o in getattr(r.sim.world.cfg, k)}
        if obj_id not in ids:
            raise HTTPException(404, f"no world object {obj_id}")
        return submit({"op": "world.remove", "id": obj_id})

    @app.post("/api/world-ambient")
    def ambient(req: AmbientRequest) -> dict:
        return submit({"op": "world.ambient", **req.model_dump(exclude_none=True)})

    # ------------------------------------------------------------ experiments on the organism
    @app.post("/api/stimulus")
    def stimulus(req: StimulusRequest) -> dict:
        return submit({"op": "stimulus", **req.model_dump()})

    @app.post("/api/brain/silence")
    def silence(req: SilenceRequest) -> dict:
        return submit({"op": "brain.silence", "cell_types": req.cell_types})

    @app.post("/api/brain/unsilence")
    def unsilence() -> dict:
        return submit({"op": "brain.unsilence"})

    @app.post("/api/learning")
    def learning(req: LearningRequest) -> dict:
        return submit({"op": "learning.set", **req.model_dump(exclude_none=True)})

    @app.post("/api/organism")
    def organism(req: OrganismRequest) -> dict:
        return submit({"op": "organism.set", **req.model_dump(exclude_none=True)})

    @app.post("/api/drive")
    def drive(req: DriveRequest) -> dict:
        return submit({"op": "drive.set", **req.model_dump(exclude_none=True)})

    # ------------------------------------------------------------ snapshots
    @app.post("/api/snapshots")
    def save_snapshot(req: SnapshotRequest) -> dict:
        return runner().save_snapshot(req.label)

    @app.get("/api/snapshots")
    def list_snapshots() -> list[dict]:
        return runner().list_snapshots()

    @app.post("/api/snapshots/{sid}/restore")
    def restore_snapshot(sid: int) -> dict:
        try:
            return runner().restore_snapshot(sid)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/body/model")
    def body_model() -> dict:
        r = runner()
        with r.lock:
            return {"geoms": r.sim.body.geom_specs(), "units": "mm"}

    # ------------------------------------------------------------ brain inspection
    @app.get("/api/brain/info")
    def brain_info() -> dict:
        r = runner()
        sim, c = r.sim, app.state.connectome

        def pop(p):
            return {"key": p.key, "label": p.label, "side": p.side, "size": p.size, "evidence": p.evidence, "reference": p.reference}

        return {
            "dataset": c.name,
            "n_neurons": c.n_neurons,
            "n_connections": c.n_connections,
            "n_synapses": c.n_synapses,
            "preset": sim.preset,
            "presets": {k: v.__dict__ for k, v in PRESETS.items()},
            "dt_ms": sim.brain.dt * 1e3,
            "clusters": [{"name": n, "size": int(s)} for n, s in zip(sim.cluster_names, sim.cluster_sizes)],
            "sensory": [pop(p) for p in sim.sensory_pops.values()],
            "motor": [pop(p) for p in sim.motor_pops.values()],
        }

    @app.get("/api/brain/layout")
    def brain_layout() -> Response:
        """Binary layout: uint32 N, then N*3 float32 positions (µm), then N uint8 cluster ids."""
        c, sim = app.state.connectome, runner().sim
        pos = c.positions_um.copy()
        nan = np.isnan(pos).any(axis=1)
        pos[nan] = np.nanmean(pos, axis=0)
        body = np.array([c.n_neurons], dtype=np.uint32).tobytes() + pos.astype(np.float32).tobytes() + sim.cluster_id.astype(np.uint8).tobytes()
        return Response(body, media_type="application/octet-stream")

    @app.get("/api/brain/neuron/{idx}")
    def neuron(idx: int, top: int = Query(15, ge=1, le=100)) -> dict:
        c = app.state.connectome
        if not 0 <= idx < c.n_neurons:
            raise HTTPException(404, "neuron index out of range")
        ann = c.annotations

        def partners(ids: np.ndarray, w: np.ndarray) -> list[dict]:
            order = np.argsort(-np.abs(w))[:top]
            return [
                {"idx": int(ids[i]), "root_id": str(c.root_ids[ids[i]]), "cell_type": ann.at[int(ids[i]), "cell_type"],
                 "cell_class": ann.at[int(ids[i]), "cell_class"], "side": ann.at[int(ids[i]), "side"], "synapses": int(w[i])}
                for i in order
            ]

        out_ids, out_w = c.outgoing(idx)
        in_ids, in_w = c.incoming(idx)
        sim = runner().sim
        return {
            "idx": idx,
            "root_id": str(c.root_ids[idx]),
            **{k: ann.at[idx, k] for k in ann.columns},
            "position_um": [None if np.isnan(v) else float(v) for v in c.positions_um[idx]],
            "n_outputs": int(out_ids.size),
            "n_inputs": int(in_ids.size),
            "spikes_total": int(sim.brain.spike_count[idx]),
            "membrane_mV": round(float(sim.brain.v[idx]) * 1e3, 3),
            "silenced": bool(sim.brain.silenced[idx]),
            "outputs": partners(out_ids, out_w),
            "inputs": partners(in_ids, in_w),
            "flywire_url": f"https://codex.flywire.ai/app/cell_details?root_id={c.root_ids[idx]}",
        }

    @app.get("/api/brain/search")
    def search(q: str = Query(..., min_length=2), limit: int = Query(50, le=500)) -> list[dict]:
        c = app.state.connectome
        ann = c.annotations
        ql = q.lower()
        if q.isdigit() and len(q) > 10:
            try:
                i = c.index_of(int(q))
                return [{"idx": i, "cell_type": ann.at[i, "cell_type"], "side": ann.at[i, "side"], "cell_class": ann.at[i, "cell_class"]}]
            except KeyError:
                return []
        m = ann["cell_type"].str.lower().str.startswith(ql) | ann["cell_class"].str.lower().eq(ql)
        idx = np.flatnonzero(m.to_numpy())[:limit]
        return [{"idx": int(i), "cell_type": ann.at[int(i), "cell_type"], "side": ann.at[int(i), "side"], "cell_class": ann.at[int(i), "cell_class"]} for i in idx]

    # ------------------------------------------------------------ experiments
    @app.get("/api/experiments")
    def experiments() -> list[dict]:
        with app.state.sessions() as s:
            rows = s.scalars(select(Experiment).order_by(Experiment.id.desc()).limit(100)).all()
            return [
                {"id": e.id, "name": e.name, "seed": e.seed, "preset": e.preset, "created_at": e.created_at.isoformat(),
                 "status": e.status, "duration_s": e.duration_s, "summary": e.summary}
                for e in rows
            ]

    @app.get("/api/experiments/{exp_id}")
    def experiment(exp_id: int) -> dict:
        with app.state.sessions() as s:
            e = s.get(Experiment, exp_id)
            if e is None:
                raise HTTPException(404, "experiment not found")
            counts = dict(s.execute(select(Event.category, func.count()).where(Event.experiment_id == exp_id).group_by(Event.category)).all())
            return {"id": e.id, "name": e.name, "seed": e.seed, "preset": e.preset, "status": e.status, "duration_s": e.duration_s,
                    "summary": e.summary, "params": e.params, "initial_world": e.initial_world, "event_counts": counts,
                    "commands": load_commands(app.state.sessions, exp_id)}

    @app.get("/api/experiments/{exp_id}/samples")
    def samples(exp_id: int, every: int = Query(1, ge=1, le=100)) -> list[dict]:
        with app.state.sessions() as s:
            rows = s.scalars(select(Sample).where(Sample.experiment_id == exp_id).order_by(Sample.t)).all()
            return [
                {"t": r.t, "x": r.x, "y": r.y, "heading": r.heading, "speed": r.speed, "energy": r.energy, "health": r.health,
                 "forward": r.forward, "turn": r.turn, "proboscis": r.proboscis, "total_spikes": r.total_spikes, "motor": r.motor, "sensory": r.sensory}
                for r in rows[::every]
            ]

    @app.get("/api/experiments/{exp_id}/events")
    def events(exp_id: int, category: str | None = None, limit: int = Query(2000, le=20000)) -> list[dict]:
        with app.state.sessions() as s:
            q = select(Event).where(Event.experiment_id == exp_id)
            if category:
                q = q.where(Event.category == category)
            rows = s.scalars(q.order_by(Event.id).limit(limit)).all()
            return [{"t": r.t, "tick": r.tick, "category": r.category, "name": r.name, "data": r.data} for r in rows]

    @app.post("/api/experiments/{exp_id}/replay-check")
    async def replay_check(exp_id: int, req: ReplayRequest) -> dict:
        """Re-simulate from seed + command log and compare with the recorded trajectory."""
        with app.state.sessions() as s:
            e = s.get(Experiment, exp_id)
            if e is None:
                raise HTTPException(404, "experiment not found")
            if e.params.get("snapshot"):
                raise HTTPException(400, "branched experiments start from a snapshot and cannot be re-run from t=0")
            rec = s.scalars(select(Sample).where(Sample.experiment_id == exp_id).order_by(Sample.t)).all()
            recorded = [(r.t, r.x, r.y) for r in rec]
            seed, preset, world = e.seed, e.preset, e.initial_world
            engine = e.params.get("engine", "numba")
            memory_id = e.params.get("memory_id")
        duration = min(req.max_seconds, recorded[-1][0] if recorded else 0.0)
        if duration <= 0:
            raise HTTPException(400, "experiment has no recorded samples")
        cmds = load_commands(app.state.sessions, exp_id)
        memory = None
        if memory_id is not None:
            try:
                memory = load_memory(app.state.sessions, memory_id, runner().sim.memory_signature())
            except KeyError as exc:
                raise HTTPException(409, f"initial memory {memory_id} of this experiment was deleted") from exc
        _, traj = await asyncio.to_thread(rerun, app.state.connectome, seed, preset, world, cmds, duration, engine, 10, memory)
        recorded = [r for r in recorded if r[0] <= duration + 1e-9]
        return {"experiment_id": exp_id, "seconds": duration, **compare_trajectories(recorded, traj)}

    # ------------------------------------------------------------ websocket
    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        last_seq = -1
        last_world = -1
        try:
            while True:
                r = runner()
                f = r.frame
                if f and f["seq"] != last_seq:
                    if f["world_seq"] != last_world:
                        with r.lock:
                            world = r.sim.world.cfg.model_dump()
                        await socket.send_json({"type": "world", "world": world})
                        last_world = f["world_seq"]
                    await socket.send_json(f)
                    last_seq = f["seq"]
                await asyncio.sleep(1 / 30)
        except WebSocketDisconnect:
            log.info("websocket client disconnected")

    return app


app = create_app() if __name__ != "__main__" else None  # uvicorn dll.api.app:app
