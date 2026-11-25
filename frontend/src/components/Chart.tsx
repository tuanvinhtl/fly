import { useEffect, useRef } from "react";

export interface Series {
  label: string;
  values: number[];
  color: string;
}

/** Lightweight canvas line chart (no chart library). x is shared, y auto or fixed. */
export function LineChart({ x, series, height = 120, yMin, yMax, unit }: { x: number[]; series: Series[]; height?: number; yMin?: number; yMax?: number; unit?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth;
    const h = height;
    c.width = w * dpr;
    c.height = h * dpr;
    const g = c.getContext("2d")!;
    g.scale(dpr, dpr);
    g.clearRect(0, 0, w, h);
    const pad = { l: 34, r: 6, t: 6, b: 16 };
    if (x.length < 2) {
      g.fillStyle = "#4a5568";
      g.font = "11px ui-sans-serif";
      g.fillText("waiting for data…", pad.l, h / 2);
      return;
    }
    const all = series.flatMap((s) => s.values);
    let lo = yMin ?? Math.min(0, ...all);
    let hi = yMax ?? Math.max(1e-9, ...all);
    if (hi - lo < 1e-9) hi = lo + 1;
    const x0 = x[0];
    const x1 = x[x.length - 1];
    const sx = (v: number) => pad.l + ((v - x0) / Math.max(1e-9, x1 - x0)) * (w - pad.l - pad.r);
    const sy = (v: number) => pad.t + (1 - (v - lo) / (hi - lo)) * (h - pad.t - pad.b);
    g.strokeStyle = "#1b2330";
    g.lineWidth = 1;
    g.fillStyle = "#4a5568";
    g.font = "10px ui-monospace, monospace";
    for (let i = 0; i <= 3; i++) {
      const v = lo + ((hi - lo) * i) / 3;
      const y = sy(v);
      g.beginPath();
      g.moveTo(pad.l, y);
      g.lineTo(w - pad.r, y);
      g.stroke();
      g.fillText(Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1), 2, y + 3);
    }
    g.fillText(`${x0.toFixed(0)}s`, pad.l, h - 3);
    const end = `${x1.toFixed(0)}s${unit ? ` · ${unit}` : ""}`;
    g.fillText(end, w - pad.r - g.measureText(end).width, h - 3);
    for (const s of series) {
      g.strokeStyle = s.color;
      g.lineWidth = 1.5;
      g.beginPath();
      s.values.forEach((v, i) => (i ? g.lineTo(sx(x[i]), sy(v)) : g.moveTo(sx(x[i]), sy(v))));
      g.stroke();
    }
  }, [x, series, height, yMin, yMax, unit]);
  return (
    <div>
      <canvas ref={ref} style={{ width: "100%", height }} />
      <div className="flex flex-wrap gap-x-3 gap-y-1 mt-1">
        {series.map((s) => (
          <span key={s.label} className="flex items-center gap-1 text-[11px] text-dim">
            <span className="w-2.5 h-0.5 rounded" style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
      </div>
    </div>
  );
}

/** Rows = clusters, columns = time; color = firing rate (log-scaled). */
export function Heatmap({ rows, labels, height = 260 }: { rows: number[][]; labels: string[]; height?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth;
    const h = height;
    c.width = w * dpr;
    c.height = h * dpr;
    const g = c.getContext("2d")!;
    g.scale(dpr, dpr);
    g.fillStyle = "#06080c";
    g.fillRect(0, 0, w, h);
    const labelW = 96;
    const nT = rows.length;
    const nC = labels.length;
    if (!nT || !nC) return;
    const cw = (w - labelW) / nT;
    const ch = h / nC;
    for (let t = 0; t < nT; t++) {
      const col = rows[t];
      for (let k = 0; k < nC; k++) {
        const v = Math.log10(1 + (col[k] ?? 0)) / Math.log10(1 + 100); // 0..1 at 100 Hz mean
        const a = Math.min(1, v);
        g.fillStyle = `rgba(${Math.round(56 + 200 * a)}, ${Math.round(189 * a + 40 * (1 - a))}, ${Math.round(248 * (1 - a * 0.6))}, ${0.08 + 0.92 * a})`;
        g.fillRect(labelW + t * cw, k * ch, Math.ceil(cw), Math.ceil(ch) - 1);
      }
    }
    g.fillStyle = "#7d8a9e";
    const fs = Math.max(7, Math.min(10, ch - 1));
    g.font = `${fs}px ui-sans-serif`;
    g.textBaseline = "middle";
    labels.forEach((l, k) => g.fillText(l.length > 15 ? `${l.slice(0, 14)}…` : l, 4, k * ch + ch / 2));
  }, [rows, labels, height]);
  return <canvas ref={ref} style={{ width: "100%", height }} />;
}
