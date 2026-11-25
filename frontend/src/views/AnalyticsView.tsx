import { useEffect, useMemo, useState } from "react";
import clsx from "clsx";
import { CheckCircle2, Play, RefreshCw, XCircle } from "lucide-react";
import { api } from "../lib/api";
import { useSim } from "../lib/store";
import type { ExperimentRow, SampleRow, SimEvent } from "../lib/types";
import { Button, Card, Empty, Slider } from "../components/ui";
import { LineChart } from "../components/Chart";

const SUMMARY_KEYS: [string, string][] = [
  ["survival_s", "survival (s)"],
  ["alive", "alive"],
  ["death_cause", "death cause"],
  ["food_eaten", "food eaten"],
  ["distance_mm", "distance (mm)"],
  ["feeding_bouts", "feeding bouts"],
  ["collisions", "collisions"],
  ["turn_left_bouts", "left turns"],
  ["turn_right_bouts", "right turns"],
  ["escapes", "escapes"],
  ["commands", "commands"],
];

interface ReplayCheck { experiment_id: number; seconds: number; n: number; max_error_mm: number | null; identical: boolean }

export default function AnalyticsView({ onError }: { onError: (e: string | null) => void }) {
  const [rows, setRows] = useState<ExperimentRow[]>([]);
  const [picked, setPicked] = useState<number[]>([]);
  const [focus, setFocus] = useState<number | null>(null);
  const [samples, setSamples] = useState<SampleRow[]>([]);
  const [events, setEvents] = useState<SimEvent[]>([]);
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [check, setCheck] = useState<ReplayCheck | null>(null);
  const [checking, setChecking] = useState(false);
  const { frame } = useSim();

  const load = () => api.get<ExperimentRow[]>("/api/experiments").then(setRows).catch((e: Error) => onError(e.message));
  useEffect(() => {
    load();
  }, [frame?.experiment_id]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (focus === null) return;
    setCheck(null);
    Promise.all([api.get<SampleRow[]>(`/api/experiments/${focus}/samples`), api.get<SimEvent[]>(`/api/experiments/${focus}/events?limit=5000`)])
      .then(([s, e]) => {
        setSamples(s);
        setEvents(e);
        setCursor(s.length ? s.length - 1 : 0);
      })
      .catch((e: Error) => onError(e.message));
  }, [focus]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => setCursor((c) => (c + 1 >= samples.length ? (setPlaying(false), c) : c + 1)), 100);
    return () => window.clearInterval(id);
  }, [playing, samples.length]);

  const toggle = (id: number) => setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id]));
  const compared = rows.filter((r) => picked.includes(r.id));
  const t = samples.map((s) => s.t);
  const speedHist = useMemo(() => {
    const bins = new Array(12).fill(0);
    samples.forEach((s) => (bins[Math.min(11, Math.floor(s.speed / 2))] += 1));
    return bins;
  }, [samples]);

  const runCheck = () => {
    if (focus === null) return;
    setChecking(true);
    api
      .post<ReplayCheck>(`/api/experiments/${focus}/replay-check`, { max_seconds: 10 })
      .then(setCheck)
      .catch((e: Error) => onError(e.message))
      .finally(() => setChecking(false));
  };

  return (
    <div className="h-full overflow-y-auto p-3 grid grid-cols-[380px_1fr] gap-3 items-start">
      <Card title="Experiments" actions={<Button size="icon" variant="ghost" onClick={load} title="Refresh"><RefreshCw size={13} /></Button>} bodyClass="p-0 max-h-[82vh] overflow-y-auto">
        {rows.length === 0 ? (
          <Empty>No recorded experiments yet.</Empty>
        ) : (
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-panel text-faint">
              <tr className="h-7">
                <th className="w-8" />
                <th className="text-left font-normal">#</th>
                <th className="text-left font-normal">name</th>
                <th className="text-right font-normal">t (s)</th>
                <th className="text-right font-normal pr-3">food</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.id} className={clsx("h-7 border-t border-line cursor-pointer", focus === r.id ? "bg-accent/10" : "hover:bg-panel-2")} onClick={() => setFocus(r.id)}>
                  <td className="text-center" onClick={(e) => e.stopPropagation()}>
                    <input type="checkbox" checked={picked.includes(r.id)} onChange={() => toggle(r.id)} title="compare" />
                  </td>
                  <td className="num text-dim">{r.id}</td>
                  <td className="truncate max-w-36">
                    {r.name} <span className="text-faint">· seed {r.seed} · {r.preset}</span>
                  </td>
                  <td className="num text-right">{r.duration_s.toFixed(1)}</td>
                  <td className="num text-right pr-3">{Number(r.summary.food_eaten ?? 0).toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>

      <div className="flex flex-col gap-3 min-w-0">
      <Card title={compared.length ? `Comparison (${compared.length})` : "Comparison — tick experiments to compare"}>
        {compared.length === 0 ? (
          <Empty>Select two or more experiments with the checkboxes.</Empty>
        ) : (
          <div className="overflow-x-auto">
            <table className="text-xs">
              <thead>
                <tr className="text-faint">
                  <th className="text-left font-normal pr-6">metric</th>
                  {compared.map((r) => (
                    <th key={r.id} className="text-right font-normal px-3">#{r.id} {r.name}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {SUMMARY_KEYS.map(([k, label]) => (
                  <tr key={k} className="h-6 border-t border-line">
                    <td className="text-dim pr-6">{label}</td>
                    {compared.map((r) => (
                      <td key={r.id} className="num text-right px-3">{String(r.summary[k] ?? "—")}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {focus === null ? (
        <Card title="Experiment detail">
          <Empty>Click an experiment to see its trajectory, time series and replay.</Empty>
        </Card>
      ) : (
        <>
          <div className="grid grid-cols-[1fr_1fr] gap-3">
            <Card
              title={`Trajectory replay · experiment #${focus}`}
              actions={
                <Button size="sm" variant="ghost" onClick={() => (cursor >= samples.length - 1 ? (setCursor(0), setPlaying(true)) : setPlaying(!playing))}>
                  <Play size={12} /> {playing ? "pause" : "play"}
                </Button>
              }
            >
              <Trajectory samples={samples} cursor={cursor} />
              <Slider value={cursor} min={0} max={Math.max(0, samples.length - 1)} step={1} onChange={setCursor} format={(v) => `${samples[v]?.t.toFixed(1) ?? 0}s`} />
            </Card>
            <Card title="Deterministic replay check">
              <div className="flex flex-col gap-3 text-xs">
                <p className="text-dim">
                  Re-simulates the first 10 s from the stored seed, preset, initial world and command log, then compares the trajectory with the recording.
                </p>
                <Button onClick={runCheck} disabled={checking}>{checking ? "re-simulating…" : "Run replay check"}</Button>
                {check && (
                  <div className={clsx("flex items-center gap-2", check.identical ? "text-good" : "text-bad")}>
                    {check.identical ? <CheckCircle2 size={14} /> : <XCircle size={14} />}
                    {check.identical ? "bit-identical trajectory" : "trajectory differs"} over {check.seconds.toFixed(1)} s ({check.n} samples, max error{" "}
                    {check.max_error_mm?.toExponential(2)} mm)
                  </div>
                )}
                <div className="text-dim">
                  events recorded: <span className="num text-ink">{events.length}</span>
                </div>
                <div>
                  <div className="text-faint mb-1">speed distribution (mm/s)</div>
                  <div className="flex items-end gap-0.5 h-16">
                    {speedHist.map((v, i) => (
                      <div key={i} className="flex-1 bg-accent/60 rounded-sm" style={{ height: `${(v / Math.max(1, ...speedHist)) * 100}%` }} title={`${i * 2}-${i * 2 + 2} mm/s: ${v}`} />
                    ))}
                  </div>
                  <div className="flex justify-between text-[10px] text-faint num"><span>0</span><span>12</span><span>24+</span></div>
                </div>
              </div>
            </Card>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <Card title="Energy / health">
              <LineChart x={t} yMin={0} yMax={100} series={[{ label: "energy", values: samples.map((s) => s.energy), color: "#34d399" }, { label: "health", values: samples.map((s) => s.health), color: "#38bdf8" }]} />
            </Card>
            <Card title="Neural activity (whole-brain spikes / 10 ms)">
              <LineChart x={t} series={[{ label: "spikes", values: samples.map((s) => s.total_spikes), color: "#a78bfa" }]} />
            </Card>
            <Card title="Motor commands">
              <LineChart x={t} yMin={-1} yMax={1} series={[{ label: "forward", values: samples.map((s) => s.forward), color: "#34d399" }, { label: "turn", values: samples.map((s) => s.turn), color: "#fbbf24" }]} />
            </Card>
            <Card title="Speed">
              <LineChart x={t} yMin={0} series={[{ label: "mm/s", values: samples.map((s) => s.speed), color: "#2dd4bf" }]} />
            </Card>
          </div>
        </>
      )}
      </div>
    </div>
  );
}

function Trajectory({ samples, cursor }: { samples: SampleRow[]; cursor: number }) {
  if (samples.length < 2) return <Empty>Not enough samples.</Empty>;
  const xs = samples.map((s) => s.x);
  const ys = samples.map((s) => s.y);
  const pad = 2;
  const x0 = Math.min(...xs) - pad, x1 = Math.max(...xs) + pad, y0 = Math.min(...ys) - pad, y1 = Math.max(...ys) + pad;
  const span = Math.max(x1 - x0, y1 - y0);
  const sx = (x: number) => ((x - x0) / span) * 100;
  const sy = (y: number) => 100 - ((y - y0) / span) * 100;
  const upto = samples.slice(0, cursor + 1);
  const cur = samples[cursor];
  return (
    <svg viewBox="0 0 100 100" className="w-full aspect-square bg-bg rounded border border-line">
      <polyline points={samples.map((s) => `${sx(s.x)},${sy(s.y)}`).join(" ")} fill="none" stroke="#1f2a3a" strokeWidth={0.4} />
      <polyline points={upto.map((s) => `${sx(s.x)},${sy(s.y)}`).join(" ")} fill="none" stroke="#38bdf8" strokeWidth={0.5} />
      {cur && (
        <g transform={`translate(${sx(cur.x)},${sy(cur.y)}) rotate(${(-cur.heading * 180) / Math.PI})`}>
          <path d="M1.8,0 L-1.2,1 L-1.2,-1 Z" fill="#fbbf24" />
        </g>
      )}
      <text x={2} y={97} fontSize={3} fill="#4a5568">{`${span.toFixed(0)} mm`}</text>
    </svg>
  );
}
