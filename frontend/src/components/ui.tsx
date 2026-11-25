/* Minimal UI primitives in the shadcn/ui style (copy-in components, Tailwind classes, no
   runtime dependency). Kept small on purpose: only what the dashboard uses. */
import clsx from "clsx";
import type { ButtonHTMLAttributes, ReactNode, SelectHTMLAttributes, InputHTMLAttributes } from "react";

type Variant = "default" | "ghost" | "outline" | "danger" | "accent";

export function Button({ variant = "default", size = "md", className, ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: "sm" | "md" | "icon" }) {
  return (
    <button
      {...props}
      className={clsx(
        "inline-flex items-center justify-center gap-1.5 rounded-md font-medium transition-colors disabled:opacity-40 disabled:pointer-events-none focus-visible:outline-2 focus-visible:outline-accent",
        size === "sm" && "h-7 px-2.5 text-xs",
        size === "md" && "h-8 px-3 text-sm",
        size === "icon" && "h-8 w-8",
        variant === "default" && "bg-panel-2 border border-line-2 hover:bg-line text-ink",
        variant === "ghost" && "hover:bg-panel-2 text-dim hover:text-ink",
        variant === "outline" && "border border-line-2 text-ink hover:bg-panel-2",
        variant === "danger" && "bg-bad/15 border border-bad/40 text-bad hover:bg-bad/25",
        variant === "accent" && "bg-accent/15 border border-accent/50 text-accent hover:bg-accent/25",
        className,
      )}
    />
  );
}

export function Card({ title, actions, children, className, bodyClass }: { title?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string; bodyClass?: string }) {
  return (
    <section className={clsx("rounded-lg border border-line bg-panel flex flex-col min-h-0", className)}>
      {(title || actions) && (
        <header className="flex items-center justify-between gap-2 px-3 h-9 border-b border-line shrink-0">
          <h3 className="text-[11px] uppercase tracking-[0.12em] text-dim font-semibold truncate">{title}</h3>
          <div className="flex items-center gap-1">{actions}</div>
        </header>
      )}
      <div className={clsx("p-3 min-h-0", bodyClass)}>{children}</div>
    </section>
  );
}

const evidenceStyle: Record<string, string> = {
  validated: "text-good border-good/40 bg-good/10",
  literature: "text-accent border-accent/40 bg-accent/10",
  approximation: "text-warn border-warn/40 bg-warn/10",
  experimental: "text-violet border-violet/40 bg-violet/10",
};

export function Badge({ children, tone = "literature", className, title }: { children: ReactNode; tone?: string; className?: string; title?: string }) {
  return (
    <span title={title} className={clsx("inline-flex items-center rounded border px-1.5 h-5 text-[10px] uppercase tracking-wider font-semibold", evidenceStyle[tone] ?? "text-dim border-line-2", className)}>
      {children}
    </span>
  );
}

export function Stat({ label, value, unit, tone }: { label: string; value: ReactNode; unit?: string; tone?: "good" | "warn" | "bad" | "accent" }) {
  return (
    <div className="min-w-0">
      <div className="text-[10px] uppercase tracking-[0.12em] text-faint">{label}</div>
      <div className={clsx("num text-lg leading-tight truncate", tone === "good" && "text-good", tone === "warn" && "text-warn", tone === "bad" && "text-bad", tone === "accent" && "text-accent")}>
        {value}
        {unit && <span className="text-xs text-dim ml-1">{unit}</span>}
      </div>
    </div>
  );
}

export function Meter({ value, max = 100, tone = "accent", label }: { value: number; max?: number; tone?: "good" | "warn" | "bad" | "accent"; label?: string }) {
  const pct = Math.max(0, Math.min(1, value / max)) * 100;
  const color = { good: "bg-good", warn: "bg-warn", bad: "bg-bad", accent: "bg-accent" }[tone];
  return (
    <div className="flex items-center gap-2">
      {label && <span className="text-[11px] text-dim w-16 shrink-0">{label}</span>}
      <div className="h-1.5 flex-1 rounded-full bg-line overflow-hidden">
        <div className={clsx("h-full rounded-full transition-[width] duration-200", color)} style={{ width: `${pct}%` }} />
      </div>
      <span className="num text-[11px] text-dim w-10 text-right">{value.toFixed(0)}</span>
    </div>
  );
}

export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return (
    <label className="flex flex-col gap-1 text-xs">
      <span className="text-dim">{label}</span>
      {children}
      {hint && <span className="text-faint text-[11px]">{hint}</span>}
    </label>
  );
}

export function Input(props: InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={clsx("h-8 rounded-md bg-panel-2 border border-line-2 px-2 text-sm text-ink placeholder:text-faint focus:outline-none focus:border-accent", props.className)} />;
}

export function Select({ className, ...props }: SelectHTMLAttributes<HTMLSelectElement>) {
  return <select {...props} className={clsx("h-8 rounded-md bg-panel-2 border border-line-2 px-2 text-sm text-ink focus:outline-none focus:border-accent", className)} />;
}

export function Slider({ value, min, max, step, onChange, format }: { value: number; min: number; max: number; step: number; onChange: (v: number) => void; format?: (v: number) => string }) {
  return (
    <div className="flex items-center gap-2">
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} className="flex-1" />
      <span className="num text-xs text-dim w-14 text-right">{format ? format(value) : value}</span>
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="h-full min-h-24 grid place-items-center text-sm text-faint text-center px-4">{children}</div>;
}
