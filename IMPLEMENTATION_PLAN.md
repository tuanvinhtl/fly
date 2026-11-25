# Implementation plan and status

Status key: ✅ done and tested · 🟡 partial · ⬜ not started

## Phase status (MVP 0.1)

| Phase | Status | What exists | Verified by |
|---|---|---|---|
| 1. Research and feasibility | ✅ | RESEARCH.md, DATA_SOURCES.md; environment survey; measurements of the dataset and the model | probe scripts, documented results |
| 2. Architecture and interfaces | ✅ | ARCHITECTURE.md; `BrainModel` protocol; `FlyBody` adapter; command and event schema | — |
| 3. Project structure | ✅ | `backend/` (package `dll`), `frontend/`, Docker Compose, `.env.example` | `docker compose config` |
| 4. Neural simulation | ✅ | Full 138,639-neuron LIF (Shiu et al. 2024), Numba plus NumPy reference, Brian2 reference, presets | `test_lif.py` (11 tests including the Brian2 comparison) |
| 5. Body | ✅ | MuJoCo fly, tripod CPG VNC, obstacles with state-preserving rebuild | `test_body.py` (7) |
| 6. Sensory feedback | ✅ | Odour, taste (sugar/bitter), temperature, light, head touch, JO stimulation | `test_world_sensors.py` (7) |
| 7. 3D world | ✅ | Arena, food, obstacles, lights, temperature/toxic zones, live editing | API + dashboard |
| 8. Dashboard | ✅ | World, Brain Explorer, Organism, Experiment Lab, Analytics, Timeline | type-checked; exercised in headless Chromium |
| 9. Logging and analytics | ✅ | SQLAlchemy recorder, comparison, trajectory playback, replay check | `test_api.py` (7, SQLite and PostgreSQL) |
| 10. Full-loop validation | ✅ | Causal silencing tests, determinism, replay, behavioural experiment | `test_integration.py` (8), `scripts/validate_behavior.py` |

Test totals: 46 tests (45 fast, plus 1 slow Brian2 comparison), all passing.

## What remains incomplete or approximate (honest list)

1. **VNC is not connectome-based.** Descending neuron rates drive a hand-designed CPG.
   Next step: add the MANC VNC connectome and match DN identities between FlyWire and MANC.
2. **No goal-directed navigation emerges.** Food odour does not produce chemotaxis in this
   model (RESEARCH.md §6). This is a result, not a bug to tune away. Next steps: (a) test
   other ORN sets and rates; (b) add the visual system's motion pathways; (c) adopt models
   with fitted synaptic weights as they appear.
3. **Compound-eye vision is not used.** Photoreceptor (R1–8) drive reaches no descending
   neuron within 0.5 s in the model, so light is sensed only through the ocelli.
4. **Internal state is a stand-in.** The "hunger drive" is Poisson input on DNp09, because the
   connectome model has no spontaneous activity and no neuromodulation.
5. **`al_std` is a model change.** Short-term depression on olfactory-system neurons is
   needed to prevent a runaway attractor. The unmodified model remains available as `exact`.
6. **Body simplifications.** 3-DoF legs, increased joint armature, no adhesion. Flight is
   quasi-steady force control rather than wing aerodynamics; take-off itself comes from the
   connectome (looming → DNp01). Collision ends a flight; a righting reflex is approximated.
7. **Speed.** ✅ Real time reached (1.0× with dashboard, 1.5× headless) on 4 CPU cores. Further options: a multi-threaded kernel (the
   integration loop parallelises trivially), or a GPU backend (the Eon repository benchmarks
   CUDA/GeNN implementations of the same model).
8. **One organism.** The world and simulation support one fly; a multi-organism ecosystem
   needs per-organism brains (~0.9 GB each) and contact between bodies.
9. **Docker images not built here.** The development container has no Docker daemon. The
   Compose file validates, and the PostgreSQL path was tested against a local PostgreSQL 16 server.

## Roadmap

| Version | Goal |
|---|---|
| 0.2 | Multi-threaded LIF kernel (target ≥1× real time); NeuroMechFly body adapter; graded ORN tuning from the DoOR database |
| 0.3 | MANC-based VNC: DN → premotor → motor neurons through connectome wiring; leg mechanosensory feedback |
| 0.4 | Visual motion pathway (optic lobe → LPTC → DNs) with a rendered eye model |
| 0.5 | Multiple organisms; population experiments; parameter sweeps on GPU |
| — | Learning: done at levels 1–2. MB memory forms (§6b), learned associative odour approach via the valence readout (§6c), and memories persist across experiments with optional forgetting (§6d). Next: replace the valence readout with connectome MBON→premotor transmission; fit forgetting to short-/long-term memory data |

## How to extend

- **New sensory channel**: add a `PopulationSpec` in `brain/circuits.py` (selector, evidence, reference),
  compute its rate in `sensory/sensors.py`, and add a test.
- **New motor readout**: add a `PopulationSpec` to `MOTOR_SPECS` and use it in `body/vnc.py`.
- **New brain model**: implement the `BrainModel` protocol and register it in `brain/presets.py`.
- **New command**: validate it in `Simulation.validate_command`, apply it in `_apply`, and expose it in `api/app.py`.
  It is then logged and replayable automatically.
