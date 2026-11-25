# Architecture

```
┌──────────────────────────── backend (Python) ─────────────────────────────┐
│                                                                            │
│  environment/world.py ──► sensory/sensors.py ──► brain/lif.py ──┐          │
│   food, odour, light,       body-relative          138,639-neuron│          │
│   temperature, toxins,      stimuli → Poisson      LIF network   │          │
│   obstacles                 rates on FlyWire       (FlyWire v783)│          │
│        ▲                    populations                          ▼          │
│        │                                         descending / motor rates  │
│        │                                                         │          │
│  sim/organism.py ◄── body/body.py (MuJoCo) ◄── body/vnc.py (CPG) ◄┘        │
│   energy, health,      physics, contacts        DN rates → joint targets   │
│   feeding                                                                  │
│                                                                            │
│  sim/simulation.py: one 10 ms tick runs the loop above; commands applied   │
│  at tick boundaries and logged → deterministic replay                      │
│  sim/runner.py: background thread, speed control, frame publishing         │
│  experiment/: SQLAlchemy models, recorder, replay      learning/: opt-in   │
│  api/app.py: FastAPI REST + WebSocket /ws                                  │
└─────────────────────────────────────▲──────────────────────────────────────┘
                                      │ JSON frames (15 Hz), REST commands
┌─────────────────────── frontend (React + R3F) ────────────────────────────┐
│ World · Brain Explorer · Organism · Experiment Lab · Analytics · Timeline │
└────────────────────────────────────────────────────────────────────────────┘
```

## Module boundaries

| Module | Responsibility | Depends on |
|---|---|---|
| `brain/connectome.py` | Load and cache the FlyWire/Shiu graph (CSR, signed counts, annotations, positions); subgraphs; partner queries | numpy, pandas |
| `brain/lif.py`, `_lif_kernel.py` | LIF dynamics (the Shiu et al. 2024 equations), Poisson inputs, silencing, per-connection weight scale (plasticity hook), optional short-term depression, snapshot/restore | connectome, numba |
| `brain/brian2_ref.py` | The published model run in Brian2, for validation only | brian2 |
| `brain/base.py` | `BrainModel` protocol: the contract that lets a different neural model be swapped in | — |
| `brain/circuits.py` | Named sensory/motor populations resolved from annotations, each with an evidence label and a reference | connectome |
| `brain/presets.py` | `exact` and `al_std` model variants | lif, circuits |
| `body/fly_model.py` | MJCF generation (body, legs, wings, proboscis, obstacles) | — |
| `body/body.py` | MuJoCo wrapper: actuators, stepping, contacts, poses, state vectors | mujoco |
| `body/vnc.py` | **Approximation of the ventral nerve cord**: DN rates → tripod CPG joint targets | — |
| `sensory/sensors.py` | World + body → physical stimuli → population firing rates | world, body |
| `environment/world.py` | Pydantic world entities and analytic fields | pydantic |
| `learning/` | Habituation and KC→MBON plasticity (experimental, opt-in) | brain |
| `sim/organism.py` | Energy, health, feeding, death, internal drive | — |
| `sim/simulation.py` | The closed loop, commands, events, snapshots | all of the above |
| `sim/runner.py` | Threaded real-time runner, frames, snapshot store | simulation, recorder |
| `experiment/` | DB schema, recorder, deterministic replay | sqlalchemy |
| `api/app.py` | HTTP/WebSocket surface | everything |

## The control tick (10 ms of simulated time)

1. Apply queued commands (world edits, stimuli, silencing…) and log them with the tick number.
2. **Sense**: antenna odour and temperature, ocellar light, labellum and front-tarsus taste,
   head contacts (`SensoryEngine.read`).
3. **Encode**: rates in Hz per sensory population, plus the internal drive on DNp09.
4. **Brain**: 100 LIF steps of 0.1 ms. Per-step order follows Brian2's schedule: state
   update → threshold → delayed synaptic delivery (dropped for refractory targets) and
   Poisson input → reset.
5. **Read out** motor populations: exponentially filtered rates (τ = 50 ms). A silenced DN or MN
   contributes nothing.
6. **VNC**: forward = (DNp09 − MDN)/60 Hz; turn = tanh((DNa02 + 0.3·DNa01)<sub>L−R</sub>/30 Hz + optional MB-valence steering);
   escape = DNp01 > 20 Hz; proboscis = MN9/20 Hz. A tripod CPG converts these into 18 leg joint
   targets plus the proboscis and wing actuators. A DNp01 burst switches to **flight**: a
   jump impulse, then quasi-steady lift, thrust and drag with PD attitude control. Power comes
   from a decaying myogenic drive plus DNg02. The fly lands when power is low and its legs
   touch down, or when a collision knocks it out of the air.
7. **Physics**: 50 MuJoCo steps of 0.2 ms.
8. **Physiology**: metabolism, feeding (proboscis extended **and** labellum on food),
   damage, death. Food is depleted.
9. **Events**: threshold crossings with hysteresis for sensory, neural, motor and movement
   channels, plus feeding, collisions and death.

Data flows in one direction each tick. The only feedback paths are physical: the body
moves, so its sensors sample different stimuli on the next tick.

## Key design decisions

**NumPy/Numba port instead of running Brian2 in the loop.** The loop advances the brain
in 10 ms chunks and changes its inputs every chunk. Brian2's per-`run()` overhead makes that
impractical on CPU. The port follows the published equations exactly (linear ODEs
integrated exactly) and is validated against Brian2: spike-for-spike on small circuits,
and a downstream spike-count ratio of 0.99 on a 9,270-neuron subnetwork (`tests/test_lif.py`).
A NumPy reference engine is kept, and the tests check that the two engines give identical results.

**Exact event-driven integration.** Between inputs each neuron follows a linear ODE with
a closed-form solution, u(s) = u₀e^(−s/τm) + C(e^(−s/τm) − e^(−s/τ)) with C = g₀τ/(τm − τ).
Since the bracket never exceeds M ≈ 0.472, max(u₀,0) + max(C,0)·M is an exact upper bound on
all future membrane potential. Neurons whose bound is below threshold ("cold") cannot spike
until new input arrives, so they are advanced in closed form only when input arrives (using
precomputed propagator tables). Only near-threshold ("hot") neurons, typically fewer than
100, are stepped every 0.1 ms. No approximation is involved: the test suite checks identical spike
counts against dense step-by-step integration on the whole brain.

**A simplified body instead of NeuroMechFly.** Six 3-DoF legs, wings, proboscis, built in
mm/mg/s units. Joint armature is raised far above the real segment inertia so that a
0.2 ms step is stable. This slows leg dynamics and is a documented approximation.
`FlyBody` is the single adapter boundary, so a NeuroMechFly body can replace it.

**An explicit VNC approximation.** The FlyWire brain dataset ends at the neck. The
literature-based DN mapping and the CPG are confined to `body/vnc.py`, which contains no
knowledge of food, light or obstacles.

**Deterministic by construction.** All randomness comes from one seeded generator inside
the brain, plus the seed-derived initial heading. MuJoCo is deterministic, and commands are
applied at tick boundaries. A replay re-simulates from (seed, preset, initial world,
command log). Tests check bit-identical trajectories, and snapshots include MuJoCo's
warm-start so a restored run continues bit-exactly.

**Threading.** One simulation thread holds an `RLock` while ticking. API handlers take the
same lock for commands and inspection. WebSocket handlers only read the latest published
frame and never hold the lock across `await`.

## Persistence

| Table | Content |
|---|---|
| `experiments` | seed, preset, initial world (JSON), params, status, duration, summary |
| `samples` | 10 Hz organism state: pose, speed, energy, health, motor command, DN rates, active sensory rates |
| `events` | every timeline event: t, tick, category, name, data |
| `commands` | every applied command with its tick, so the run can be replayed |
| `memories` | saved mushroom-body memories: KC→MBON weight scales (float32) plus provenance and a connectome signature |

SQLite is the default for local development and tests. PostgreSQL is used under Docker and
has been tested with `DLL_TEST_DATABASE_URL=postgresql+psycopg://…`.

## Performance (4 vCPU Xeon 2.3 GHz, no GPU)

| Component | Cost per 10 ms tick |
|---|---|
| Brain (138,639 neurons, typical activity of 30–45k spikes/s) | ~4–5 ms |
| MuJoCo (50 steps) | ~1.4 ms |
| Sensors, VNC, physiology, events | ~1 ms |

That gives 1.5× real time headless. With the dashboard streaming in a browser on the same
4 cores: exactly 1.0× at the 1× setting and 1.3× at max speed (measured over 15 s).
Peak memory is about 0.9 GB with one snapshot. Disk: 130 MB raw data plus 240 MB cache.

## API summary

`GET /api/status · POST /api/sim/{start,pause,resume,step,reset,speed}`
`GET /api/world · POST /api/world/{food|obstacles|lights|zones} · PATCH|DELETE /api/world/{id} · POST /api/world-ambient`
`POST /api/stimulus · POST /api/brain/{silence,unsilence} · POST /api/learning · POST /api/organism · POST /api/drive`
`POST|GET /api/snapshots · POST /api/snapshots/{id}/restore`
`GET /api/brain/{info,layout,neuron/{idx},search} · GET /api/body/model`
`GET /api/experiments[/{id}[/samples|/events]] · POST /api/experiments/{id}/replay-check`
`WS /ws`: `{"type":"world"}` whenever the world changes, and `{"type":"frame"}` at about 15 Hz.
