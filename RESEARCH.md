# Research and feasibility report

*Digital Life Lab, MVP 0.1, Drosophila melanogaster. Written 2026-11-25.*

This report separates four kinds of statement, using the same labels as the code and
the dashboard:

- **[data]**: read directly from the published datasets.
- **[validated]**: a model result that its authors compared with experiments.
- **[measured]**: a result we measured in this simulator; the script and test are named.
- **[approximation] / [experimental]**: a modelling choice of ours.

The simulated organism is a dynamical model. It is not conscious and has no subjective
experience, and nothing here implies otherwise.

---

## 1. Development environment

| | |
|---|---|
| OS | Linux 6.18 (x86_64), cloud container |
| CPU | 4 vCPU Intel Xeon @ 2.30 GHz |
| RAM | 15 GB, no swap |
| GPU | none |
| Disk | 61 GB free |
| Python | 3.13.18 |
| Node / pnpm | 22.23.5 / 10.28.0 |
| Docker | client 29.9.3; **no daemon available** in this container |
| PostgreSQL | 16.15 binaries (used for a local integration test) |
| Network | GitHub raw content and PyPI/npm reachable; codex.flywire.ai and zenodo.org not reachable |

## 2. Projects investigated

| Project | What it provides | Used? |
|---|---|---|
| **FlyWire** (Dorkenwald et al. 2024; Schlegel et al. 2024) | Whole-brain EM connectome of an adult female, v783: ~139k neurons with annotations, cell types, predicted transmitters | **Yes**, through the two GitHub releases in DATA_SOURCES.md |
| **Shiu et al. 2024** (*Nature* 634:210), `philshiu/Drosophila_brain_model`, MIT | LIF model of the entire FlyWire brain in Brian2. One free parameter (weight per synapse) fitted to data; predicted sensorimotor pathways (e.g. sugar → MN9 feeding) with ~90% accuracy against experiments [validated] | **Yes.** Ported exactly (§4) |
| **Eon Systems** (`eonsystemspbc/fly-brain`) | Public code: a benchmarking harness that runs the Shiu model on Brian2 CPU/CUDA, PyTorch, NEST GPU and GeNN, using Brian2 CPU as ground truth. Press reports (2026) describe an embodied demo with the Shiu brain in NeuroMechFly v2 and MuJoCo (walking, antennal grooming, proboscis extension). According to the same reports, the brain–body coupling code is **not released** and the work is **not peer reviewed** | Studied. Its benchmarks suggest GPU backends as a route to faster-than-real-time |
| **NeuroMechFly v2 / FlyGym** (Wang-Chen et al. 2024; `flygym` 2.1.1 on PyPI) | Anatomically detailed fly body in MuJoCo with many leg DoF, an adhesion model, vision and olfaction rendering, and CPG/rule-based walking controllers | **Not in the MVP** (§3). `FlyBody` is the adapter boundary for it. Check its licence before bundling |
| **MuJoCo** 4.0.1 | Physics engine | **Yes** |
| **Brian2** 2.11.0 | Spiking-network simulator | **Yes**, as the reference implementation for validation |
| **Virtual Fly Brain** | Ontology and anatomy browser | Not queried; annotations carry `vfb_id` / `fbbt_id` |

Other relevant work for the motor interface: Bidaye et al. 2014 (MDN, backward walking),
Bidaye et al. 2020 (P9/DNp09, forward walking), von Reyn et al. 2014 (giant fiber DNp01,
escape), Rayshubskiy et al. 2024 *eLife* (DNa02 high-gain and DNa01 low-gain steering),
Braun et al. 2024 *Nature* (command-like DNs recruit networks of other DNs), and Gordon &
Scott 2009 (MN9, proboscis extension).

## 3. Feasibility decisions

**Which connectome?** The full FlyWire v783 brain, in Shiu et al.'s signed neuron-to-neuron
format: 138,639 neurons, 15,091,983 connections, 54,492,922 synapses [data]. A subset was
not needed (see below). Using Shiu's exact files keeps us comparable with a published,
experimentally validated model.

**Can it run locally?** Yes. Measured on this machine [measured]:

| | |
|---|---|
| Load cached connectome | 0.1 s, ~320 MB |
| Full simulator, peak RSS | ~0.9 GB including one snapshot |
| Brain speed | ~4 ms per 10 ms tick during closed-loop walking (exact event-driven kernel, ARCHITECTURE.md) |
| Closed loop | 1.5× real time headless; 1.0× at the 1× setting and 1.3× at max speed while streaming to the dashboard |
| GPU | not required. A GPU would allow faster-than-real-time runs and parameter sweeps |

**What needs simplification?**

1. **The ventral nerve cord.** FlyWire v783 is a brain dataset, so descending neurons are
   its last stage. The leg motor system is replaced by a CPG driven by
   literature-identified DNs [approximation].
2. **The body.** A 3-DoF-per-leg MuJoCo model instead of NeuroMechFly, to keep the MVP's
   control stack minimal and inspectable [approximation].
3. **The sensory periphery.** Analytic stimulus fields and Michaelis–Menten rate codes
   [approximation]. Only sensory populations that are annotated in FlyWire are driven.
4. **Internal state.** The model is silent without input, so a hunger-scaled Poisson
   drive onto DNp09 stands in for arousal [approximation].

**How brain, body and environment are connected.** See ARCHITECTURE.md. In each 10 ms
tick: body-relative stimuli → Poisson rates on named FlyWire sensory populations →
100 LIF steps → filtered DN/MN rates → VNC CPG → MuJoCo → physiology. The only path from
brain output back to brain input is through the physical body.

## 4. Neural engine: port, validation and two corrections

### 4.1 The port

`brain/lif.py` implements the Shiu et al. equations and constants exactly: v₀ = v_reset = −52 mV,
v_th = −45 mV, τ_m = 20 ms, τ_syn = 5 ms, refractory period 2.2 ms, delay 1.8 ms,
w = 0.275 mV per synapse, Poisson weight = 250·w, dt = 0.1 ms. Integration is exact (linear
ODEs). The Numba kernel is checked against the NumPy reference and gives identical spikes
(`test_numba_matches_numpy_reference_exactly`).

### 4.2 Brian2 refractory semantics (bug found and fixed)

The first comparison against Brian2 showed 28% more downstream spikes in our engine.
Feeding identical presynaptic spike trains into both simulators showed the cause:
**Brian2 discards synaptic increments to variables marked `(unless refractory)` while the
target neuron is refractory.** Inputs arriving within 2.2 ms after a spike are lost. Our port
had stored them. A minimal Brian2 script confirmed the behaviour (input at 1.0 ms and 2.0 ms
after a spike at 0.8 ms was lost; input at 3.0 ms was kept).

After the fix [measured]:
- toy circuits: **spike-for-spike identical** with Brian2;
- 9,270-neuron sugar subnetwork, 10 trials: downstream spike totals ratio **0.992**
  (difference 34 spikes, standard error 62), i.e. statistically indistinguishable
  (`test_matches_brian2_reference`, r > 0.95 per-neuron rate correlation).

Every result below was re-measured after this fix.

### 4.3 Validated pathway reproduced

Left sugar GRNs (20 neurons) at 100 Hz → MN9 (FlyWire type CB0701) at **36 / 58 Hz**
(left / right) [measured]. This is the headline gustatory result of Shiu et al. [validated]
(`test_sugar_activates_mn9_full_brain`).

### 4.4 Runaway olfactory attractor and the `al_std` preset

Driving **any** olfactory receptor neurons or arista thermosensory neurons, even 3 heat
neurons at 5 Hz, ignites a whole-brain state: ~8,500 neurons fire at up to 450 Hz, and it
**never terminates**. About 95,000 spikes are still produced in the 0.5 s after the stimulus
is removed [measured] (`test_exact_model_olfactory_runaway_and_al_std_fix`). The descending-
neuron pattern is the same whichever sense ignited it, so in this state the input carries
no information.

Diagnosis:
- The ignition path is ORN/TRN → projection neurons → antennal-lobe local neurons. The core is
  the lLN1_bc population, which is cholinergic according to the curated `known_nt`
  annotations, so it is genuinely excitatory. Kenyon cells are recruited almost entirely
  (>3,800).
- Removing KC→KC axo-axonic connections (293,762 connections) or the monoaminergic "fast"
  synapses shrinks the attractor but does not abolish it.
- The published model has no mechanism that ends sustained high-rate firing: no
  adaptation and no synaptic depression. Shiu et al. validated feedforward
  sensorimotor pathways, not recurrent olfactory dynamics.

Fix [approximation]: Tsodyks–Markram short-term depression (U = 0.4, τ_rec = 0.5 s) on the
~10k neurons of the antennal lobe, mushroom body and lateral horn (`ALLN, ALPN, ALIN,
ALON, Kenyon_Cell, LHLN, LHCENT, MBON, DAN, MBIN`). Depression at these synapses is
documented experimentally (e.g. Kazama & Wilson 2008 for ORN→PN). Alternatives we tested
and rejected:

| Variant | Attractor ends? | Sugar → MN9 kept? |
|---|---|---|
| exact | no | yes |
| global STD U = 0.05–0.3 | only for U ≥ 0.2 | **no** (drops to ≤ 6 Hz) |
| STD on all central neurons, sensory exempt | only for U ≥ 0.3 | **no** |
| **STD on olfactory system, U = 0.4 (`al_std`)** | **yes** | **yes** (34 / 64 Hz) |

Both presets are selectable. `al_std` is the default for closed-loop work because, without it,
any odour or temperature input saturates the brain.

### 4.5 A narrower variant: `ln_std` (depression on antennal-lobe local neurons only)

Restricting depression to the ~400 antennal-lobe local neurons (`ALLN`), the attractor core
identified above, also abolishes persistent activity. It gives 0 persistent spikes for
food-odour, heat and cold input on either side over 4 seeds, and for all 2,282 ORNs driven
together. It keeps the validated pathways (sugar → MN9 36/58 Hz, left head touch → left DNa02
14 Hz, looming → DNp01 150/172 Hz). Unlike `al_std`, it leaves the mushroom body working with
realistic sparseness [measured]:

| | `al_std` | `ln_std` | in vivo |
|---|---|---|---|
| Kenyon cells active per odour | 0.4–0.9% | **6.8–12.2%** | ~5–10% (Turner et al. 2008; Honegger et al. 2011) |
| KC overlap between two odours (Jaccard) | — | 0.22 | low |
| MBON population response (odour A / B) | ~0 Hz | 12.6 / 1.1 Hz mean | odour-specific |

**Trade-off.** With `ln_std` in the closed loop, any food odour drives left DNa02 much harder
than right DNa02 (~22 vs ~0 Hz), so the fly spins left 100% of the time (5 seeds × 30 s; raw data
in `docs/results/`). Left and right DNa02 receive similar total input (+7,881 vs +8,365
excitatory synapses), so this is a network-level effect of uniform synaptic weights, not
missing reconstruction. `al_std` therefore remains the closed-loop default, and `ln_std` is used for
learning experiments.

## 5. Sensory → descending-neuron map (measured, before choosing the motor interface)

Each stimulus: one side, 100 Hz Poisson, 0.5 s, `al_std`, no internal drive. Values are DN
rates in Hz, left / right [measured] (probe procedure recorded in tests and in this report).

| Stimulus | DNa02 | DNa01 | DNp09 | MN9 | Notes |
|---|---|---|---|---|---|
| sugar L | 0 / 0 | 0 / 0 | 0 / 0 | **34 / 64** | validated pathway |
| head bristles L | **16 / 0** | 0 / 6 | 0 / 0 | 0 | lateralised steering signal |
| food-odour ORNs L | 10 / 2 | 6 / 2 | 0 / 0 | 0 | ORNs on *either* side favour left DNa02 (not lateralised) |
| heat L | 12 / 0 | 12 / 2 | 0 / 0 | 0 | |
| cold L | 2 / 0 | 2 / 0 | 0 / 0 | 0 | DNb05 strongly lateralised (52 / 10) |
| ocelli L | 0 / 0 | 0 / 0 | 0 / 0 | 0 | reaches DNp28 and DNp22 (not mapped to motion) |
| compound-eye R1–8 | — | — | — | — | **no DN within 0.5 s** |
| *drive on DNp09, 50 Hz* | 0 / **14** | 0 / 0 | 40 / 76 | 0 | intrinsic right-turn bias |

Consequences for the motor interface [approximation, see `body/vnc.py`]: forward = DNp09,
steering = DNa02 (gain 1) + DNa01 (gain 0.3), backward = MDN, escape = DNp01,
proboscis = MN9.

## 6. Closed-loop results (what the organism actually does)

All results use `al_std` on the full brain [measured].

**Causal tests** (`tests/test_integration.py`):

| Manipulation | Result |
|---|---|
| fly standing on food | sugar GRNs → MN9 > 20 Hz → proboscis extends → feeding → energy rises |
| same, with MN9 (CB0701) silenced | no ingestion at all |
| internal drive on DNp09 | the fly walks (> 5 mm in 2 s) |
| same, with DNp09 silenced | < 20% of control distance |
| sustained left head-bristle input | left DNa02 peak > 8 Hz and > 2× right; the body turns left by > 10° (in practice ~280° in 1 s) |
| same, with DNa02 + DNa01 silenced | no turn (< 5°) |
| obstacle against the left of the head | left head-bristle channel fires; event logged |
| ambient temperature raised mid-run | hot-cell drive > 50 Hz, whole-brain spiking increases |
| predator swooping toward the fly | looming → LC4/LPLC2 (~110 Hz) → **DNp01 giant fiber 60–175 Hz** → take-off, flight, landing; predator misses |
| same, with DNp01 silenced | no take-off; the predator strikes (health 100 → ~42) |

**Escape pathway** [measured]: driving LC4 or LPLC2 at 50–200 Hz on one side gives DNp01
37–190 Hz, together with DNp02/DNp04/DNp11/DNp103. This is the connectome counterpart of the
looming → giant fiber escape (von Reyn et al. 2014, 2017; Ache et al. 2019). DNg02 (flight
power, Namiki et al. 2022) stays silent in all conditions we tested, so flight power after
take-off comes from the flight-motor approximation in `body/vnc.py`.

**Behavioural experiment** (`scripts/validate_behavior.py`, 5 seeds × 30 s, food 10 mm away;
raw data in `docs/results/`):

| Condition | Mean distance to food | Time within 4 mm | Food eaten | Walked | Time turning right |
|---|---|---|---|---|---|
| food odour on | 10.5 ± 3.7 mm | 7% | 3.5 ± 5.1 | 168 mm | 40% |
| food odour off | 9.4 ± 2.9 mm | 11% | 4.7 ± 4.3 | 173 mm | 42% |
| odour on, DNp09 silenced | 10.0 mm (stationary) | 0% | 0 | 0.4 mm | 0% |

(Measured with the soft-saturating VNC turn mapping used since the learning work. With the earlier
hard-clipped mapping, odour-on flies ended farther from food: 13.4 vs 7.7 mm.)

Interpretation:
- **No innate chemotaxis emerges.** Odour does not bring the fly closer to food: the
  difference is within seed-to-seed variation.
- An intrinsic right-turn bias (DNp09 → right DNa02, §5) makes flies circle near their start
  point, so contact with food 10 mm away is partly chance.
- Feeding on contact works and depends causally on MN9.
- Odour approach does appear **after associative learning**, via the valence readout (§6c).

These are the expected limits of a connectome-constrained model with uniform synaptic
weights, no neuromodulation, an approximate VNC and a hand-chosen DN mapping. We report
them instead of hard-coding food-seeking, which the project brief rules out.

## 6b. Associative learning (experimental module)

`learning/mushroom_body.py` implements dopamine-gated depression of KC→MBON synapses
(Hige et al. 2015). Compartments come from DAN→MBON connectivity: all 96 MBONs receive direct
DAN synapses, 68 of them from PAM. The model contains no taste → dopamine pathway (strong sugar
input activates 0 of 307 PAM neurons [measured]), so taste injects reinforcement: sugar drives PAM
(reward; Liu et al. 2012) and bitter drives PPL1 (punishment; Das et al. 2014) [approximation].

**Differential conditioning** (`scripts/conditioning.py`, `ln_std`, 3 seeds; odour A paired with
sugar 6×, odour B unpaired; plasticity off during tests; 3 test repetitions) [measured]:

| | learning | control (plasticity off) |
|---|---|---|
| MBON response to A (CS+), post / pre | **0.45** | 1.00 |
| MBON response to B (CS−), post / pre | 0.92 | 0.99 |
| summed change of DN responses, A / B | 125 / 96 Hz | 126 / 97 Hz |

- **An odour-specific memory forms** in the mushroom body output. The most affected MBONs are
  MBON07 (α1) and MBON02 (β2β'2a), types implicated experimentally in appetitive memory
  (Ichinose et al. 2015; Owald et al. 2015). About 8,200 of 62,261 KC→MBON synapses are depressed.
- **It does not reach behaviour.** Descending-neuron responses change no more than in the
  control. Anatomical MBON → DN paths exist (2-hop: 1,813 synapses onto DNa02, 970 onto MDN), but
  silencing all MBONs barely alters DN output in the model. Learned behaviour would need either
  a model in which MBON signals propagate to premotor circuits (e.g. fitted weights or
  neuromodulation), or an explicit MBON-valence readout, which would be an approximation.

### 6c. Valence readout: from memory to steering (approximation)

`learning/valence.py` adds the MBON → premotor link that the connectome model does not
express. It uses the population-level rule that glutamatergic MBONs mostly drive avoidance and
cholinergic/GABAergic MBONs mostly drive approach (Aso et al. 2014b). The two groups are
balanced on a panel of 6 naive odours. The resulting signal steers toward (V > 0) or away from
(V < 0) the antenna that smells the odour more strongly.

Steering from the readout is summed with the descending-neuron steering drive **before**
the VNC's soft saturation (turn = tanh(raw)). It is not added after a hard clip.

**Forced-valence check of the steering mechanics** (V held fixed, 3 seeds × 20 s, odour 10 mm
away) [measured]:

| preset, readout gains | V = +1 | V = −1 | V = 0 |
|---|---|---|---|
| `al_std`, gain 2 / contrast 60 | 4.9 mm | 33.4 mm | 12.1 mm |
| `ln_std`, gain 2 / contrast 60 | 11.2 mm | 10.3 mm | 10.5 mm (spins left 99%) |
| `ln_std`, **gain 3 / contrast 300** (default) | **4.0 mm** | **33.4 mm** | — |

Under `ln_std`, odour drives left DNa02 to ~50 Hz. The cause is AOTU019 mutual inhibition,
which is asymmetric in the connectome: left → right DNa02 has 216 synapses, right → left has 121.
The readout must be strong enough to override this, and the antennae sit only ~0.2 mm apart, so
the left/right odour contrast is ~0.4% and needs a high contrast gain. Both settings are
approximations.

**Learned approach** (`scripts/learned_approach.py`, `ln_std`, 5 seeds; 6 s held on sugar food,
then 30 s free walking with an odour-only source 10 mm away; raw data in `docs/results/`)
[measured]:

| condition | MB valence V at end | mean distance | time within 4 mm |
|---|---|---|---|
| **trained** (scented sugar, plasticity + readout) | **+0.96 … +1.19** | **4.5 ± 0.2 mm** | **57 ± 13%** |
| naive (no training) | −0.24 … +0.13 | 23.8 ± 0.3 mm | 0% |
| unpaired sugar (odourless sugar) | +0.02 … +0.14 | 22.7 mm | 0% |
| trained, readout off | — | 10.7 ± 0.4 mm | 0% |

- **Learned, associative odour approach emerges.** Only flies that experienced the odour
  together with sugar approach it. Odourless sugar (reward without the odour) produces no
  approach, and the memory sits in the KC→MBON synapses (~6,200 depressed).
- Naive flies (and the unpaired group) move away. While they smell it, this food odour has negative MB
  valence for them (−1.3 to −1.6 in the earlier run with weaker gains), so they steer away. V then decays
  toward 0 once they are far from the source. Odour exposure alone also causes some non-associative
  depression (~2,400 synapses), because odour weakly activates DANs.
- **What is connectome and what is not:** odour coding, Kenyon cells, MBON responses and the
  site and specificity of the memory come from the connectome model. The reinforcement signal
  (taste → DANs), the learning rule, the valence sign rule and the valence → steering link are
  approximations based on the literature.

### 6d. Memory across experiments and forgetting ("level 2")

The KC→MBON weight scales (62,261 values) can be saved to the database and loaded into later
experiments (`experiment/memory.py`, `/api/memories`, reset with `memory_id` or
`keep_memory`). A memory is accepted only if its plastic-synapse signature matches the
connectome. The experiment records which memory it started from, so deterministic replay still
works. Optional passive forgetting relaxes weights back to 1 with a time constant τ; it is exact
exponential decay, runs whether or not learning is on, and is a phenomenological
[approximation] of memory decay.

**Retention test** (`scripts/memory_retention.py`, `ln_std`, 3 seeds). Experiment 1 is 6 s of
training on scented sugar. Experiment 2 is a **new simulation** (different seed, position and
heading; fresh brain dynamics) with learning switched off and the valence readout on, then 30 s
of free walking with an odour-only source 10 mm away [measured]:

| condition in experiment 2 | memory strength at test | mean distance | time within 4 mm |
|---|---|---|---|
| memory loaded | 100% | **4.1 ± 0.2 mm** | **58%** |
| memory loaded, τ = 20 s, 10 s delay | ~61% | 4.3 ± 0.1 mm | 44% |
| memory loaded, τ = 20 s, 60 s delay | ~5% | 18.2 ± 0.3 mm | 0% |
| no memory | 0 | 19.5 ± 0.6 mm | 0% |

The learned preference survives a complete restart of the simulation. It weakens in a graded
way as the stored synaptic change decays, and is lost once ~95% of the change has relaxed.
Real Drosophila memories range from minutes (short-term) to days (long-term, after
spaced training). τ here is a free parameter, not fitted to those data.

## 7. Limitations and open questions

- Synaptic weights are proportional to synapse counts with one global scale. Real
  efficacies vary with receptor types, dendritic location and neuromodulation.
- Transmitter signs come from predictions at the neuron level, and ~2–12% disagree with the
  newer annotation release (DATA_SOURCES.md).
- Neuromodulatory transmitters (DA, 5-HT, OA) are treated as fast excitation, as in
  Shiu et al.
- The DN → motor mapping is our interpretation of specific studies. The VNC is not modelled
  from data.
- The `al_std` parameters were chosen to satisfy two constraints (the attractor ends; the
  validated pathway is preserved). They were not fitted to electrophysiology.
- Learning modules are off by default. MB plasticity is a literature-based rule with injected
  reinforcement; it changes MBON responses but not motor output in this model (§6b).
- Flight is quasi-steady force control (lift, thrust, drag, haltere-like PD stabilisation),
  not wing aerodynamics. Cruise speed is scaled down to the arena (30 mm/s, real flies
  ~0.5 m/s), and flight ends by decay of the myogenic drive or on collision (no visual obstacle
  avoidance). In flight the walking steering DNs are assumed to produce yaw.

## 8. References

Bidaye S.S. et al. (2014) *Science* 344:97 · Bidaye S.S. et al. (2020) *Neuron* 108:469 ·
Dorkenwald S. et al. (2024) *Nature* 634:124 · Eckstein N. et al. (2024) *Cell* 187 ·
Gallio M. et al. (2011) *Cell* 144:614 · Das G. et al. (2014) *Curr Biol* 24 · Liu C. et al. (2012) *Nature* 488:512 ·
Ichinose T. et al. (2015) *eLife* 4 · Aso Y. et al. (2014b) *eLife* 3:e04580 · Owald D. et al. (2015) *Neuron* 86:417 · Turner G.C. et al. (2008) *J Neurophysiol* 99 · Honegger K.S. et al. (2011) *J Neurosci* 31 · Gordon M.D. & Scott K. (2009) *Neuron* 61:373 ·
Hige T. et al. (2015) *Neuron* 88:985 · Kazama H. & Wilson R.I. (2008) *Neuron* 58:401 ·
Rayshubskiy A. et al. (2024) *eLife* (reviewed preprint 102230) · Braun J. et al. (2024) *Nature*, "Descending networks transform command signals into population motor control" ·
Schlegel P. et al. (2024) *Nature* 634:139 · Shiu P.K. et al. (2024) *Nature* 634:210 ·
von Reyn C.R. et al. (2014) *Nat Neurosci* 17:962 · von Reyn C.R. et al. (2017) *Neuron* 94 ·
Ache J.M. et al. (2019) *Curr Biol* 29 · Namiki S. et al. (2022) *Nat Commun* (DNg02) · Wang-Chen S. et al. (2024) *Nat Methods* (NeuroMechFly v2).
Press coverage of Eon Systems: The Register (2026-03-16), The Decoder.
