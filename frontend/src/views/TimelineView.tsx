import { useState } from "react";
import clsx from "clsx";
import { useSim } from "../lib/store";
import { CATEGORY_COLOR } from "../lib/format";
import { Card, Empty } from "../components/ui";

const CATS = ["sensory", "neural", "motor", "movement", "environment", "organism", "experiment"] as const;

export default function TimelineView() {
  const { events, frame } = useSim();
  const [on, setOn] = useState<Record<string, boolean>>(Object.fromEntries(CATS.map((c) => [c, true])));
  const shown = events.filter((e) => on[e.category]);
  const t1 = frame?.t ?? 0;
  const t0 = Math.max(0, t1 - 20);
  const lane = shown.filter((e) => e.t >= t0);
  return (
    <div className="h-full p-3 flex flex-col gap-3 min-h-0">
      <Card title="Causal chain lanes (last 20 s)" actions={<span className="text-[11px] text-faint">sensory input → neural response → motor command → movement</span>}>
        <div className="relative" style={{ height: CATS.length * 22 + 8 }}>
          {CATS.map((c, i) => (
            <div key={c} className="absolute left-0 right-0 flex items-center" style={{ top: i * 22 + 4, height: 18 }}>
              <span className="w-24 text-[11px] text-dim">{c}</span>
              <div className="relative flex-1 h-full border-b border-line">
                {lane
                  .filter((e) => e.category === c && !e.name.endsWith(":off"))
                  .map((e, k) => (
                    <span
                      key={k}
                      title={`${e.t.toFixed(2)}s ${e.name}`}
                      className="absolute top-1 w-1.5 h-3 rounded-sm"
                      style={{ left: `${((e.t - t0) / Math.max(1e-6, t1 - t0)) * 100}%`, background: CATEGORY_COLOR[c] }}
                    />
                  ))}
              </div>
            </div>
          ))}
        </div>
      </Card>
      <Card
        title={`Events (${shown.length})`}
        className="flex-1"
        bodyClass="flex-1 overflow-y-auto p-0"
        actions={CATS.map((c) => (
          <button
            key={c}
            onClick={() => setOn({ ...on, [c]: !on[c] })}
            className={clsx("h-6 px-2 rounded text-[11px] border", on[c] ? "border-line-2 text-ink" : "border-transparent text-faint line-through")}
          >
            <span className="inline-block w-2 h-2 rounded-full mr-1" style={{ background: CATEGORY_COLOR[c] }} />
            {c}
          </button>
        ))}
      >
        {shown.length === 0 ? (
          <Empty>No events yet — run the simulation or apply a stimulus.</Empty>
        ) : (
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-panel text-faint">
              <tr className="h-7">
                <th className="text-left font-normal pl-3 w-20">t (s)</th>
                <th className="text-left font-normal w-28">category</th>
                <th className="text-left font-normal">event</th>
                <th className="text-left font-normal">data</th>
              </tr>
            </thead>
            <tbody>
              {[...shown].reverse().map((e, i) => (
                <tr key={i} className="h-6 border-t border-line hover:bg-panel-2">
                  <td className="num pl-3 text-dim">{e.t.toFixed(2)}</td>
                  <td>
                    <span style={{ color: CATEGORY_COLOR[e.category] }}>{e.category}</span>
                  </td>
                  <td>{e.name}</td>
                  <td className="num text-faint truncate max-w-md">{Object.keys(e.data).length ? JSON.stringify(e.data) : ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  );
}
