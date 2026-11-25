import { useEffect, useState } from "react";
import clsx from "clsx";
import { Activity, Brain, Bug, FlaskConical, Globe, ListTree, Pause, Play, RotateCcw, StepForward } from "lucide-react";
import { connect, useSim } from "./lib/store";
import { api } from "./lib/api";
import { fmtTime } from "./lib/format";
import { Button, Meter, Select } from "./components/ui";
import WorldView from "./views/WorldView";
import BrainView from "./views/BrainView";
import OrganismView from "./views/OrganismView";
import LabView from "./views/LabView";
import AnalyticsView from "./views/AnalyticsView";
import TimelineView from "./views/TimelineView";

const VIEWS = [
  { id: "world", label: "World", icon: Globe },
  { id: "brain", label: "Brain Explorer", icon: Brain },
  { id: "organism", label: "Organism", icon: Bug },
  { id: "lab", label: "Experiment Lab", icon: FlaskConical },
  { id: "analytics", label: "Analytics", icon: Activity },
  { id: "timeline", label: "Event Timeline", icon: ListTree },
] as const;
type ViewId = (typeof VIEWS)[number]["id"];

const SPEEDS = [0.25, 0.5, 1, 2, 4, 0];

export default function App() {
  const [view, setView] = useState<ViewId>(() => {
    try {
      return (localStorage.getItem("dll.view") as ViewId) || "world";
    } catch {
      return "world";
    }
  });
  const [error, setError] = useState<string | null>(null);
  const sim = useSim();
  const f = sim.frame;

  useEffect(() => connect(), []);
  useEffect(() => {
    try {
      localStorage.setItem("dll.view", view);
    } catch {
      /* storage unavailable; view just won't persist */
    }
  }, [view]);

  const call = (p: Promise<unknown>) => p.then(() => setError(null)).catch((e: Error) => setError(e.message));

  return (
    <div className="h-full flex flex-col">
      <header className="h-12 shrink-0 border-b border-line bg-panel flex items-center gap-3 px-3">
        <div className="flex items-center gap-2 pr-3 border-r border-line">
          <div className="w-6 h-6 rounded bg-accent/15 border border-accent/40 grid place-items-center">
            <Brain size={14} className="text-accent" />
          </div>
          <div className="leading-tight">
            <div className="text-sm font-semibold tracking-wide">Digital Life Lab</div>
            <div className="text-[10px] text-faint">Drosophila · FlyWire v783 connectome</div>
          </div>
        </div>
        <div className="flex items-center gap-1">
          {f?.running ? (
            <Button variant="accent" onClick={() => call(api.post("/api/sim/pause"))} title="Pause">
              <Pause size={14} /> Pause
            </Button>
          ) : (
            <Button variant="accent" onClick={() => call(api.post("/api/sim/start"))} disabled={!f?.organism.alive} title="Run">
              <Play size={14} /> Run
            </Button>
          )}
          <Button size="icon" variant="ghost" title="Step 10 ms" onClick={() => call(api.post("/api/sim/step", { ticks: 1 }))}>
            <StepForward size={15} />
          </Button>
          <Button size="icon" variant="ghost" title="Reset experiment" onClick={() => call(api.post("/api/sim/reset", { name: "experiment" }))}>
            <RotateCcw size={15} />
          </Button>
          <Select
            value={f?.speed ?? 1}
            onChange={(e) => call(api.post("/api/sim/speed", { multiplier: Number(e.target.value) }))}
            className="ml-1 h-8 text-xs"
            title="Simulation speed (sim seconds per wall second)"
          >
            {SPEEDS.map((s) => (
              <option key={s} value={s}>
                {s === 0 ? "max speed" : `${s}×`}
              </option>
            ))}
          </Select>
        </div>
        <div className="flex items-center gap-4 text-xs ml-2">
          <span className="num text-base text-ink">{fmtTime(f?.t ?? 0)}</span>
          <span className="text-dim">
            real-time factor <span className="num text-ink">{f?.running ? f.realtime_factor.toFixed(2) : "—"}</span>
          </span>
          <span className="text-dim">
            experiment <span className="num text-ink">#{f?.experiment_id ?? "—"}</span>
          </span>
        </div>
        <div className="ml-auto flex items-center gap-3 text-xs">
          {(error || f?.error) && <span className="text-bad max-w-md truncate" title={error ?? f?.error ?? ""}>{error ?? f?.error}</span>}
          <span className={clsx("flex items-center gap-1.5", sim.connected ? "text-good" : "text-bad")}>
            <span className={clsx("w-1.5 h-1.5 rounded-full", sim.connected ? "bg-good" : "bg-bad animate-pulse")} />
            {sim.connected ? "live" : "connecting…"}
          </span>
        </div>
      </header>
      <div className="flex-1 min-h-0 flex">
        <nav className="w-48 shrink-0 border-r border-line bg-panel p-2 flex flex-col gap-0.5">
          {VIEWS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setView(id)}
              className={clsx(
                "flex items-center gap-2 h-8 px-2.5 rounded-md text-sm text-left transition-colors",
                view === id ? "bg-accent/10 text-accent" : "text-dim hover:text-ink hover:bg-panel-2",
              )}
            >
              <Icon size={15} />
              {label}
            </button>
          ))}
          <div className="mt-auto p-2 text-[10px] leading-relaxed text-faint">
            Simulated organism. Not conscious; no claim of subjective experience. Labels mark what is validated, literature-based, or an
            approximation.
          </div>
        </nav>
        <main className="flex-1 min-w-0 min-h-0 overflow-hidden">
          {view === "world" && <WorldView onError={setError} />}
          {view === "brain" && <BrainView />}
          {view === "organism" && <OrganismView />}
          {view === "lab" && <LabView onError={setError} />}
          {view === "analytics" && <AnalyticsView onError={setError} />}
          {view === "timeline" && <TimelineView />}
        </main>
        <aside className="w-64 shrink-0 border-l border-line bg-panel p-3 flex flex-col gap-4 overflow-y-auto">
          <Vitals />
        </aside>
      </div>
    </div>
  );
}

function Vitals() {
  const { frame: f } = useSim();
  if (!f) return <div className="text-sm text-faint">Waiting for simulation…</div>;
  const o = f.organism;
  const c = f.motor.command;
  const r = f.motor.rates;
  const pair = (k: string) => `${(r[`${k}_left`] ?? 0).toFixed(0)} / ${(r[`${k}_right`] ?? 0).toFixed(0)}`;
  return (
    <>
      <div>
        <div className="flex items-center justify-between">
          <span className="text-[11px] uppercase tracking-[0.12em] text-dim font-semibold">{o.id}</span>
          <span className={clsx("text-[11px] font-semibold", o.alive ? "text-good" : "text-bad")}>{o.alive ? (c.flying ? "FLYING" : o.feeding ? "FEEDING" : "ALIVE") : `DEAD · ${o.death_cause}`}</span>
        </div>
        <div className="mt-2 flex flex-col gap-1.5">
          <Meter label="Energy" value={o.energy} tone={o.energy > 30 ? "good" : o.energy > 10 ? "warn" : "bad"} />
          <Meter label="Health" value={o.health} tone={o.health > 50 ? "accent" : "bad"} />
          <Meter label="Drive" value={o.drive_hz} max={100} tone="warn" />
        </div>
      </div>
      <div>
        <div className="text-[11px] uppercase tracking-[0.12em] text-dim font-semibold mb-1.5">Motor command (VNC)</div>
        <Bar label="forward" v={c.forward} />
        <Bar label="turn ←" v={c.turn} />
        <Bar label="proboscis" v={c.proboscis} />
        <Bar label="flight" v={c.flying ? c.flight_power : 0} />
        {c.flying && <div className="text-xs text-accent mt-1">airborne · altitude {f.body.position[2].toFixed(1)} mm</div>}
      </div>
      <div>
        <div className="text-[11px] uppercase tracking-[0.12em] text-dim font-semibold mb-1.5">Descending neurons (Hz, L / R)</div>
        <table className="w-full text-xs">
          <tbody>
            {[
              ["DNp09", "dn_forward"],
              ["DNa02", "dn_steer_high"],
              ["DNa01", "dn_steer_low"],
              ["MDN", "dn_backward"],
              ["DNp01", "dn_escape"],
              ["MN9", "mn_proboscis"],
            ].map(([l, k]) => (
              <tr key={k} className="h-5">
                <td className="text-dim">{l}</td>
                <td className="num text-right">{pair(k)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="text-xs text-dim flex flex-col gap-1">
        <div className="flex justify-between">
          <span>spikes / 10 ms</span>
          <span className="num text-ink">{f.brain.spikes_tick}</span>
        </div>
        <div className="flex justify-between">
          <span>speed</span>
          <span className="num text-ink">{f.body.speed.toFixed(1)} mm/s</span>
        </div>
        <div className="flex justify-between">
          <span>silenced neurons</span>
          <span className="num text-ink">{f.brain.silenced}</span>
        </div>
      </div>
    </>
  );
}

function Bar({ label, v }: { label: string; v: number }) {
  const pct = Math.min(1, Math.abs(v)) * 50;
  return (
    <div className="flex items-center gap-2 h-5">
      <span className="text-[11px] text-dim w-16">{label}</span>
      <div className="relative flex-1 h-1.5 bg-line rounded-full">
        <div className="absolute top-0 bottom-0 w-px bg-line-2 left-1/2" />
        <div className={clsx("absolute top-0 h-full rounded-full", v >= 0 ? "bg-good" : "bg-warn")} style={v >= 0 ? { left: "50%", width: `${pct}%` } : { right: "50%", width: `${pct}%` }} />
      </div>
      <span className="num text-[11px] w-10 text-right">{v.toFixed(2)}</span>
    </div>
  );
}
