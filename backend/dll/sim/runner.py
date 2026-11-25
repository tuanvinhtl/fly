"""Background thread that advances the Simulation at a chosen speed and publishes frames."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

import numpy as np

from ..experiment.recorder import Recorder
from .simulation import Simulation

log = logging.getLogger(__name__)

FRAME_INTERVAL_S = 1 / 15  # wall-clock frame publishing rate
MAX_ACTIVE_NEURONS_PER_FRAME = 6000


class SimulationRunner:
    def __init__(self, sim: Simulation, recorder: Recorder):
        self.sim = sim
        self.recorder = recorder
        self.lock = threading.RLock()
        self.running = False
        self.speed = 1.0  # sim seconds per wall second; 0 = as fast as possible
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.frame: dict[str, Any] = {}
        self.frame_seq = 0
        self._events_sent = 0
        self.world_seq = 0
        self._world_sig: Any = None
        self.realtime_factor = 0.0
        self.snapshots: dict[int, dict] = {}
        self._snap_meta: dict[int, dict] = {}
        self._snap_next = 1
        self.on_error: Callable[[Exception], None] | None = None
        self.last_error: str | None = None
        with self.lock:
            self.recorder.start(sim, "session start")
            self._publish()

    # ------------------------------------------------------------------ thread
    def start_thread(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="sim-runner", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        with self.lock:
            self.recorder.flush(self.sim, final=True)

    def _loop(self) -> None:
        last_frame = last_flush = time.perf_counter()
        sim_t0, wall_t0 = self.sim.time, time.perf_counter()
        while not self._stop.is_set():
            if not self.running:
                time.sleep(0.02)
                sim_t0, wall_t0 = self.sim.time, time.perf_counter()
                now = time.perf_counter()
                if now - last_frame > 0.25:
                    with self.lock:
                        self._publish()
                    last_frame = now
                continue
            try:
                with self.lock:
                    self.sim.tick()
                    self.recorder.observe(self.sim)
            except Exception as exc:  # surfaced to clients; simulation pauses
                log.exception("simulation tick failed")
                self.running = False
                self.last_error = f"{type(exc).__name__}: {exc}"
                continue
            now = time.perf_counter()
            if self.speed > 0:
                ahead = (self.sim.time - sim_t0) / self.speed - (now - wall_t0)
                if ahead > 0:
                    time.sleep(min(ahead, 0.1))
            if now - last_frame >= FRAME_INTERVAL_S:
                elapsed = now - wall_t0
                self.realtime_factor = (self.sim.time - sim_t0) / elapsed if elapsed > 0 else 0.0
                with self.lock:
                    self._publish()
                last_frame = now
            if now - last_flush > 1.0:
                with self.lock:
                    self.recorder.flush(self.sim)
                last_flush = now
            if not self.sim.organism.alive and self.running:
                self.running = False
                with self.lock:
                    self.recorder.flush(self.sim, final=True)
                    self._publish()

    # ------------------------------------------------------------------ controls
    def set_running(self, running: bool) -> None:
        self.running = running

    def step(self, n_ticks: int = 1) -> None:
        with self.lock:
            for _ in range(n_ticks):
                self.sim.tick()
                self.recorder.observe(self.sim)
            self.recorder.flush(self.sim)
            self._publish()

    def reset(self, seed: int | None = None, world=None, name: str = "experiment", preset: str | None = None) -> int:
        with self.lock:
            self.running = False
            self.recorder.flush(self.sim, final=True)
            if preset is not None and preset != self.sim.preset:
                from ..brain.presets import build_brain

                self.sim.preset = preset
                self.sim.brain = build_brain(self.sim.c, preset, seed or self.sim.seed, self.sim.engine)
            self.sim.reset(seed=seed, world=world)
            self._events_sent = 0
            exp_id = self.recorder.start(self.sim, name)
            self._publish()
            return exp_id

    def submit(self, command: dict) -> None:
        with self.lock:
            self.sim.submit(command)
            if not self.running:  # apply immediately while paused so the UI reflects it
                self.sim.tick()
                self.recorder.observe(self.sim)
                self._publish()

    def save_snapshot(self, label: str) -> dict:
        with self.lock:
            sid = self._snap_next
            self._snap_next += 1
            self.snapshots[sid] = self.sim.snapshot()
            meta = {"id": sid, "label": label, "t": round(self.sim.time, 3), "tick": self.sim.tick_count, "experiment_id": self.recorder.experiment_id}
            self._snap_meta[sid] = meta
            while len(self.snapshots) > 10:  # each snapshot holds ~60 MB of brain state
                oldest = min(self.snapshots)
                del self.snapshots[oldest], self._snap_meta[oldest]
            return meta

    def list_snapshots(self) -> list[dict]:
        return list(self._snap_meta.values())

    def restore_snapshot(self, sid: int) -> dict:
        with self.lock:
            if sid not in self.snapshots:
                raise KeyError(f"snapshot {sid} not found")
            self.running = False
            self.recorder.flush(self.sim, final=True)
            self.sim.restore(self.snapshots[sid])
            self.recorder.start(self.sim, f"branch from snapshot {sid}", {"snapshot": self._snap_meta[sid]})
            self.recorder.mark_branch(self.sim)
            self._events_sent = len(self.sim.events)
            self._publish()
            return self._snap_meta[sid]

    # ------------------------------------------------------------------ frames
    def _publish(self) -> None:
        sim = self.sim
        st = sim.body.state()
        counts = sim.frame_counts
        idx = np.flatnonzero(counts)
        if idx.size > MAX_ACTIVE_NEURONS_PER_FRAME:
            idx = idx[np.argsort(counts[idx])[-MAX_ACTIVE_NEURONS_PER_FRAME:]]
        cnt = counts[idx]
        counts[:] = 0
        events = [e.to_dict() for e in sim.events[self._events_sent :]][-200:]
        self._events_sent = len(sim.events)
        world_sig = sim.world.cfg.model_dump_json()
        if world_sig != self._world_sig:
            self._world_sig = world_sig
            self.world_seq += 1
        r = sim.last_reading
        c = sim.last_cmd
        self.frame_seq += 1
        self.frame = {
            "type": "frame",
            "seq": self.frame_seq,
            "t": round(sim.time, 3),
            "tick": sim.tick_count,
            "running": self.running,
            "speed": self.speed,
            "realtime_factor": round(self.realtime_factor, 3),
            "experiment_id": self.recorder.experiment_id,
            "world_seq": self.world_seq,
            "error": self.last_error,
            "organism": sim.organism.to_dict(),
            "body": {
                "position": st.position.round(4).tolist(),
                "heading": round(st.heading, 4),
                "speed": round(float(np.linalg.norm(st.velocity[:2])), 3),
                "geoms": sim.body.geom_poses(),
            },
            "motor": {
                "rates": {k: round(v, 2) for k, v in sim.motor_rates.items()},
                "command": {"forward": round(c.forward, 3), "turn": round(c.turn, 3), "escape": c.escape, "proboscis": round(c.proboscis, 3)},
            },
            "sensory": {
                "rates": {k: round(v, 2) for k, v in sim.sensory_rates.items()},
                "physical": {k: round(v, 4) for k, v in r.physical.items()},
                "external": dict(sim.sensors.external),
            },
            "brain": {
                "clusters": [round(float(x), 3) for x in sim.cluster_rates],
                "active_idx": idx.tolist(),
                "active_cnt": cnt.tolist(),
                "spikes_tick": sim.total_spikes_tick,
                "silenced": int(sim.brain.silenced.sum()),
            },
            "learning": {"habituation": sim.habituation_on, "plasticity": sim.plasticity_on},
            "events": events,
        }
