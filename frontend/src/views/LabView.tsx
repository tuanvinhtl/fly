import { useEffect, useState } from "react";
import { Camera, History, Zap } from "lucide-react";
import { api } from "../lib/api";
import { useSim } from "../lib/store";
import { useBrainInfo } from "../lib/useStatic";
import { prettyKey } from "../lib/format";
import { Badge, Button, Card, Field, Input, Select, Slider } from "../components/ui";

interface Snapshot { id: number; label: string; t: number; tick: number; experiment_id: number }

export default function LabView({ onError }: { onError: (e: string | null) => void }) {
  const { frame, world } = useSim();
  const { data: info } = useBrainInfo(frame?.experiment_id?.toString());
  const run = (p: Promise<unknown>) => p.then(() => onError(null)).catch((e: Error) => onError(e.message));

  // stimulus
  const [pop, setPop] = useState("touch_head_left");
  const [rate, setRate] = useState(150);
  const [dur, setDur] = useState(1);
  // environment
  const [temp, setTemp] = useState(25);
  const [light, setLight] = useState(0.2);
  useEffect(() => {
    if (world) {
      setTemp(world.ambient_temperature);
      setLight(world.ambient_light);
    }
  }, [world?.ambient_temperature, world?.ambient_light]); // eslint-disable-line react-hooks/exhaustive-deps
  // silencing
  const [types, setTypes] = useState("DNa02, DNa01");
  // drive
  const [base, setBase] = useState(30);
  const [hunger, setHunger] = useState(50);
  // reset
  const [seed, setSeed] = useState(42);
  const [name, setName] = useState("experiment");
  const [preset, setPreset] = useState("al_std");
  // snapshots
  const [snaps, setSnaps] = useState<Snapshot[]>([]);
  const loadSnaps = () => api.get<Snapshot[]>("/api/snapshots").then(setSnaps).catch((e: Error) => onError(e.message));
  useEffect(() => {
    loadSnaps();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const sensory = info?.sensory ?? [];
  const selectedPop = sensory.find((p) => p.key === pop);

  return (
    <div className="h-full overflow-y-auto p-3 grid grid-cols-3 gap-3 content-start">
      <Card title="Sensory stimulation" actions={<Zap size={13} className="text-accent" />}>
        <div className="flex flex-col gap-3">
          <Field label="Sensory population (FlyWire-annotated neurons)">
            <Select value={pop} onChange={(e) => setPop(e.target.value)}>
              {sensory.map((p) => (
                <option key={p.key} value={p.key}>
                  {prettyKey(p.key)} — {p.size} neurons
                </option>
              ))}
            </Select>
          </Field>
          {selectedPop && (
            <div className="text-[11px] text-dim flex flex-col gap-1">
              <div className="flex items-center gap-2">
                <Badge tone={selectedPop.evidence}>{selectedPop.evidence}</Badge> {selectedPop.label}
              </div>
              <span className="text-faint">{selectedPop.reference}</span>
            </div>
          )}
          <Field label="Poisson rate per neuron (Hz)">
            <Slider value={rate} min={0} max={300} step={10} onChange={setRate} />
          </Field>
          <Field label="Duration (s)">
            <Slider value={dur} min={0.1} max={10} step={0.1} onChange={setDur} format={(v) => v.toFixed(1)} />
          </Field>
          <Button variant="accent" onClick={() => run(api.post("/api/stimulus", { population: pop, rate_hz: rate, duration_s: dur }))}>
            Apply stimulus
          </Button>
          <Button
            variant="danger"
            title="An object swoops down at the fly. It is seen only through looming-sensitive LC4/LPLC2 neurons; escape depends on the giant fiber (DNp01)."
            onClick={() => {
              const p = frame?.body.position ?? [0, 0, 0];
              run(api.post("/api/world/threats", { x: p[0] + 8, y: p[1] + 6, z: 15, speed: 40 }));
            }}
          >
            Release predator (looming threat)
          </Button>
          {frame && Object.keys(frame.sensory.external).length > 0 && (
            <div className="text-[11px] text-warn">active: {Object.entries(frame.sensory.external).map(([k, v]) => `${prettyKey(k)} ${v} Hz`).join(", ")}</div>
          )}
        </div>
      </Card>

      <Card title="Environment conditions">
        <div className="flex flex-col gap-3">
          <Field label="Ambient temperature (°C)" hint="Damage above 34 °C / below 12 °C (model physiology)">
            <Slider value={temp} min={5} max={45} step={0.5} onChange={setTemp} format={(v) => v.toFixed(1)} />
          </Field>
          <Field label="Ambient light (arbitrary units)">
            <Slider value={light} min={0} max={5} step={0.1} onChange={setLight} format={(v) => v.toFixed(1)} />
          </Field>
          <Button onClick={() => run(api.post("/api/world-ambient", { temperature: temp, light }))}>Apply conditions</Button>
          <div className="text-[11px] text-faint">
            Add or remove food, obstacles, lights and zones directly in the World view. All changes are logged as commands and replayable.
          </div>
          {world && (
            <div className="text-xs text-dim grid grid-cols-2 gap-1">
              <span>food sources: {world.food.length}</span>
              <span>obstacles: {world.obstacles.length}</span>
              <span>lights: {world.lights.length}</span>
              <span>zones: {world.zones.length}</span>
            </div>
          )}
        </div>
      </Card>

      <Card title="Neural manipulation" actions={<Badge tone="validated">Shiu-style silencing</Badge>}>
        <div className="flex flex-col gap-3">
          <Field label="Silence cell types (comma separated)" hint="Zeroes all outgoing synapses; silenced DNs/MNs drive no motor output.">
            <Input value={types} onChange={(e) => setTypes(e.target.value)} />
          </Field>
          <div className="flex flex-wrap gap-1">
            {["DNp09", "DNa02, DNa01", "CB0701", "MDN", "DNp01"].map((t) => (
              <Button key={t} size="sm" variant="ghost" onClick={() => setTypes(t)}>
                {t}
              </Button>
            ))}
          </div>
          <div className="flex gap-2">
            <Button variant="danger" onClick={() => run(api.post("/api/brain/silence", { cell_types: types.split(",").map((s) => s.trim()).filter(Boolean) }))}>
              Silence
            </Button>
            <Button onClick={() => run(api.post("/api/brain/unsilence"))}>Restore all</Button>
          </div>
          <div className="text-xs text-dim">currently silenced: {frame?.brain.silenced ?? 0} neurons</div>
        </div>
      </Card>

      <Card title="Internal state drive" actions={<Badge tone="approximation">not connectome</Badge>}>
        <div className="flex flex-col gap-3">
          <Field label="Baseline DNp09 drive (Hz)">
            <Slider value={base} min={0} max={120} step={5} onChange={setBase} />
          </Field>
          <Field label="Additional drive at full hunger (Hz)">
            <Slider value={hunger} min={0} max={120} step={5} onChange={setHunger} />
          </Field>
          <Button onClick={() => run(api.post("/api/drive", { drive_base_hz: base, drive_hunger_hz: hunger }))}>Apply drive</Button>
          <div className="flex gap-2">
            <Button size="sm" variant="ghost" onClick={() => run(api.post("/api/organism", { energy: 15 }))}>Starve (energy 15)</Button>
            <Button size="sm" variant="ghost" onClick={() => run(api.post("/api/organism", { energy: 100, health: 100 }))}>Restore vitals</Button>
          </div>
        </div>
      </Card>

      <Card title="Learning engine" actions={<Badge tone="experimental">experimental</Badge>}>
        <div className="flex flex-col gap-3 text-xs">
          <label className="flex items-start gap-2">
            <input type="checkbox" checked={!!frame?.learning.habituation} onChange={(e) => run(api.post("/api/learning", { habituation: e.target.checked }))} />
            <span>
              <span className="text-ink">Sensory habituation</span>
              <span className="block text-faint">Phenomenological gain reduction under sustained input (encoder level).</span>
            </span>
          </label>
          <label className="flex items-start gap-2">
            <input type="checkbox" checked={!!frame?.learning.plasticity} onChange={(e) => run(api.post("/api/learning", { plasticity: e.target.checked }))} />
            <span>
              <span className="text-ink">Associative learning in the mushroom body</span>
              <span className="block text-faint">
                Taste releases dopamine (sugar → PAM reward, bitter → PPL1 punishment). Dopamine depresses synapses from recently active Kenyon cells onto
                the output neurons of its compartment (Hige et al. 2015). Odours present while feeding become remembered.
              </span>
            </span>
          </label>
          {frame?.learning.memory && (
            <div className="rounded border border-line p-2 grid grid-cols-2 gap-x-3 gap-y-1">
              <span className="text-dim">synapses changed</span>
              <span className="num text-right">{frame.learning.memory.depressed.toLocaleString()} / {frame.learning.memory.plastic_synapses.toLocaleString()}</span>
              <span className="text-dim">mean KC→MBON weight</span>
              <span className="num text-right">{frame.learning.memory.mean_weight.toFixed(3)}</span>
              <span className="text-dim">dopamine now</span>
              <span className={`num text-right ${frame.learning.dopamine_max_hz > 0 ? "text-warn" : ""}`}>{frame.learning.dopamine_max_hz.toFixed(0)} Hz</span>
            </div>
          )}
          <p className="text-faint text-[11px]">
            Use the <span className="text-ink">ln_std</span> preset (New experiment, below): with the default preset the mushroom body is nearly silent.
            Memories form in the mushroom body output (MBON response to the trained odour drops to ~45%), but do not yet change motor output — see
            RESEARCH.md §6b.
          </p>
        </div>
      </Card>

      <Card title="Snapshots" actions={<Camera size={13} className="text-dim" />}>
        <div className="flex flex-col gap-2">
          <Button onClick={() => run(api.post("/api/snapshots", { label: `t=${frame?.t.toFixed(1)}s` }).then(loadSnaps))}>Save snapshot</Button>
          {snaps.length === 0 && <span className="text-xs text-faint">No snapshots in this session.</span>}
          {snaps.map((s) => (
            <div key={s.id} className="flex items-center justify-between text-xs">
              <span>
                #{s.id} {s.label} <span className="text-faint">(exp {s.experiment_id})</span>
              </span>
              <Button size="sm" variant="ghost" onClick={() => run(api.post(`/api/snapshots/${s.id}/restore`))}>
                <History size={12} /> restore
              </Button>
            </div>
          ))}
          <span className="text-[11px] text-faint">Restoring starts a new branch experiment from that exact state (brain, body, world).</span>
        </div>
      </Card>

      <Card title="New experiment" className="col-span-3">
        <div className="grid grid-cols-5 gap-3 items-end">
          <Field label="Name">
            <Input value={name} onChange={(e) => setName(e.target.value)} />
          </Field>
          <Field label="Random seed">
            <Input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
          </Field>
          <Field label="Brain model preset">
            <Select value={preset} onChange={(e) => setPreset(e.target.value)}>
              <option value="al_std">al_std — Shiu 2024 + olfactory-system STD (default)</option>
              <option value="ln_std">ln_std — AL local-neuron STD (use for learning experiments)</option>
              <option value="exact">exact — Shiu 2024 unmodified</option>
            </Select>
          </Field>
          <Button variant="accent" onClick={() => run(api.post("/api/sim/reset", { seed, name, preset, world }))}>
            Reset with current world
          </Button>
          <Button onClick={() => run(api.post("/api/sim/reset", { seed, name, preset, default_world: true }))}>Reset with default world</Button>
        </div>
        {info && <p className="text-[11px] text-faint mt-3">{info.presets[preset]?.description} Deviation: {info.presets[preset]?.deviation_from_publication}</p>}
      </Card>
    </div>
  );
}
