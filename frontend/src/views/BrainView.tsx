import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, type ThreeEvent } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { ExternalLink, Search } from "lucide-react";
import { api, fetchLayout } from "../lib/api";
import { onFrame, useSim } from "../lib/store";
import { useBrainInfo } from "../lib/useStatic";
import type { NeuronDetail } from "../lib/types";
import { Badge, Button, Card, Empty, Input } from "../components/ui";
import { Heatmap, LineChart } from "../components/Chart";

// 32 distinguishable hues for neuron classes
const PALETTE = Array.from({ length: 40 }, (_, i) => new THREE.Color().setHSL((i * 0.618034) % 1, 0.55, 0.55));

interface Layout {
  n: number;
  positions: Float32Array;
  clusters: Uint8Array;
}

export default function BrainView() {
  const [layout, setLayout] = useState<Layout | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [highlight, setHighlight] = useState<number | null>(null);
  const { data: info } = useBrainInfo();
  const { history } = useSim();

  useEffect(() => {
    fetchLayout()
      .then(setLayout)
      .catch((e: Error) => setLoadErr(e.message));
  }, []);

  const clusterNames = info?.clusters.map((c) => c.name) ?? [];
  const recent = history.clusters.slice(-180);

  return (
    <div className="h-full grid grid-cols-[1fr_360px] grid-rows-[1fr_350px] gap-3 p-3 min-h-0">
      <Card
        title={info ? `${info.n_neurons.toLocaleString()} neurons · ${info.n_connections.toLocaleString()} connections · ${info.n_synapses.toLocaleString()} synapses` : "Brain"}
        actions={
          <>
            <Badge tone="validated" title={info?.dataset}>FlyWire v783</Badge>
            <Badge tone={info?.preset === "exact" ? "validated" : "approximation"} title={info?.presets[info.preset]?.deviation_from_publication}>
              preset: {info?.preset ?? "…"}
            </Badge>
          </>
        }
        className="min-h-0"
        bodyClass="p-0 flex-1 relative"
      >
        {loadErr && <Empty>Failed to load neuron layout: {loadErr}</Empty>}
        {!layout && !loadErr && <Empty>Loading 138,639 neuron positions…</Empty>}
        {layout && (
          <Canvas camera={{ position: [0, 0, 900], fov: 40, near: 1, far: 5000 }} dpr={[1, 2]}>
            <color attach="background" args={["#040609"]} />
            <NeuronCloud layout={layout} selected={selected} highlight={highlight} onPick={setSelected} />
            <OrbitControls makeDefault enableDamping />
          </Canvas>
        )}
        <div className="absolute left-3 bottom-3 text-[11px] text-faint pointer-events-none">
          Points = neuron positions in FAFB space (µm), colour = cell class; flashes = spikes in the last frame. Click a neuron to inspect it.
        </div>
      </Card>
      <div className="row-span-2 min-h-0">
        <Inspector selected={selected} onSelect={setSelected} onHighlightCluster={setHighlight} clusterNames={clusterNames} />
      </div>
      <div className="grid grid-cols-2 gap-3 min-h-0">
        <Card title="Activity heatmap by cell class (last 12 s, mean Hz)" bodyClass="flex-1">
          {recent.length ? <Heatmap rows={recent} labels={clusterNames} height={290} /> : <Empty>Run the simulation to see activity.</Empty>}
        </Card>
        <Card title="Network activity timeline" bodyClass="flex-1">
          <LineChart x={history.t} series={[{ label: "spikes per 10 ms (whole brain)", values: history.spikes, color: "#a78bfa" }]} height={260} />
        </Card>
      </div>
    </div>
  );
}

function NeuronCloud({ layout, selected, highlight, onPick }: { layout: Layout; selected: number | null; highlight: number | null; onPick: (i: number) => void }) {
  const geom = useRef<THREE.BufferGeometry>(null);
  const activity = useMemo(() => new Float32Array(layout.n), [layout]);
  const { positions, base } = useMemo(() => {
    const p = layout.positions;
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < layout.n; i++) {
      cx += p[3 * i];
      cy += p[3 * i + 1];
      cz += p[3 * i + 2];
    }
    cx /= layout.n;
    cy /= layout.n;
    cz /= layout.n;
    const pos = new Float32Array(layout.n * 3);
    const col = new Float32Array(layout.n * 3);
    for (let i = 0; i < layout.n; i++) {
      // FAFB: x = left-right, y = dorsal->ventral, z = anterior->posterior. Show a frontal view.
      pos[3 * i] = -(p[3 * i] - cx);
      pos[3 * i + 1] = -(p[3 * i + 1] - cy);
      pos[3 * i + 2] = -(p[3 * i + 2] - cz);
      const c = PALETTE[layout.clusters[i] % PALETTE.length];
      col[3 * i] = c.r;
      col[3 * i + 1] = c.g;
      col[3 * i + 2] = c.b;
    }
    return { positions: pos, base: col };
  }, [layout]);
  const colors = useMemo(() => new Float32Array(base.length), [base]);

  useEffect(
    () =>
      onFrame((f) => {
        const idx = f.brain.active_idx;
        const cnt = f.brain.active_cnt;
        for (let k = 0; k < idx.length; k++) activity[idx[k]] = Math.min(1, activity[idx[k]] + 0.35 + 0.15 * cnt[k]);
      }),
    [activity],
  );

  useFrame(() => {
    const g = geom.current;
    if (!g) return;
    const dim = highlight === null ? 0.22 : 0.06;
    for (let i = 0; i < layout.n; i++) {
      const a = activity[i];
      const inCluster = highlight !== null && layout.clusters[i] === highlight;
      const b = inCluster ? 0.75 : dim;
      colors[3 * i] = base[3 * i] * b + a * 1.0;
      colors[3 * i + 1] = base[3 * i + 1] * b + a * 0.95;
      colors[3 * i + 2] = base[3 * i + 2] * b + a * 0.8;
      if (a > 0) activity[i] = a < 0.02 ? 0 : a * 0.9;
    }
    if (selected !== null) {
      colors[3 * selected] = 1;
      colors[3 * selected + 1] = 1;
      colors[3 * selected + 2] = 1;
    }
    (g.attributes.color as THREE.BufferAttribute).needsUpdate = true;
  });

  const click = (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation();
    if (e.index !== undefined) onPick(e.index);
  };

  return (
    <points onClick={click} raycast={pointsRaycast}>
      <bufferGeometry ref={geom}>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
      </bufferGeometry>
      <pointsMaterial size={2.2} vertexColors sizeAttenuation transparent opacity={0.9} blending={THREE.AdditiveBlending} depthWrite={false} />
    </points>
  );
}

// Points raycasting with a picking radius suited to µm coordinates.
function pointsRaycast(this: THREE.Points, raycaster: THREE.Raycaster, intersects: THREE.Intersection[]) {
  raycaster.params.Points = { threshold: 3 };
  THREE.Points.prototype.raycast.call(this, raycaster, intersects);
  intersects.sort((a, b) => (a.distanceToRay ?? 0) - (b.distanceToRay ?? 0));
  intersects.splice(1);
}

function Inspector({ selected, onSelect, onHighlightCluster, clusterNames }: { selected: number | null; onSelect: (i: number) => void; onHighlightCluster: (c: number | null) => void; clusterNames: string[] }) {
  const [detail, setDetail] = useState<NeuronDetail | null>(null);
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<{ idx: number; cell_type: string; side: string; cell_class: string }[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const { frame } = useSim();
  const { data: info } = useBrainInfo();

  useEffect(() => {
    if (selected === null) return;
    api
      .get<NeuronDetail>(`/api/brain/neuron/${selected}`)
      .then((d) => {
        setDetail(d);
        setErr(null);
      })
      .catch((e: Error) => setErr(e.message));
  }, [selected, frame?.seq && Math.floor(frame.seq / 15)]);

  const search = () => {
    if (q.trim().length < 2) return;
    api
      .get<typeof hits>(`/api/brain/search?q=${encodeURIComponent(q.trim())}`)
      .then(setHits)
      .catch((e: Error) => setErr(e.message));
  };

  const quick = ["DNp09", "DNa02", "DNa01", "MDN", "DNp01", "CB0701"];

  return (
    <Card title="Neuron inspector" className="h-full" bodyClass="flex-1 overflow-y-auto flex flex-col gap-3">
      <div className="flex gap-1.5">
        <Input value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && search()} placeholder="cell type, class or FlyWire root id" className="flex-1" />
        <Button size="icon" onClick={search} title="Search">
          <Search size={14} />
        </Button>
      </div>
      <div className="flex flex-wrap gap-1">
        {quick.map((t) => (
          <Button
            key={t}
            size="sm"
            variant="ghost"
            onClick={() => {
              setQ(t);
              api.get<typeof hits>(`/api/brain/search?q=${t}`).then(setHits).catch((e: Error) => setErr(e.message));
            }}
          >
            {t}
          </Button>
        ))}
      </div>
      {hits.length > 0 && (
        <div className="max-h-32 overflow-y-auto rounded border border-line">
          {hits.map((h) => (
            <button key={h.idx} onClick={() => onSelect(h.idx)} className="w-full flex justify-between px-2 h-6 text-xs hover:bg-panel-2 text-left">
              <span>{h.cell_type || "(untyped)"} <span className="text-faint">{h.cell_class}</span></span>
              <span className="text-dim">{h.side} · #{h.idx}</span>
            </button>
          ))}
        </div>
      )}
      {err && <div className="text-xs text-bad">{err}</div>}
      {selected === null || !detail ? (
        <Empty>Select a neuron in the 3D view or search by type.</Empty>
      ) : (
        <div className="flex flex-col gap-3 text-xs">
          <div>
            <div className="flex items-center justify-between">
              <span className="text-base font-semibold">{detail.cell_type || "untyped neuron"}</span>
              {detail.silenced && <Badge tone="approximation">silenced</Badge>}
            </div>
            <div className="text-dim">
              {[detail.super_class, detail.cell_class, detail.cell_sub_class].filter(Boolean).join(" › ")} · {detail.side} · {detail.top_nt || "nt unknown"}
            </div>
            <a href={detail.flywire_url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-accent mt-1">
              root {detail.root_id} <ExternalLink size={11} />
            </a>
          </div>
          <div className="grid grid-cols-3 gap-2">
            <Mini label="inputs" v={detail.n_inputs} />
            <Mini label="outputs" v={detail.n_outputs} />
            <Mini label="spikes" v={detail.spikes_total} />
            <Mini label="Vm (mV)" v={detail.membrane_mV.toFixed(2)} />
            <Mini label="model idx" v={detail.idx} />
          </div>
          <PartnerList title="Strongest outputs (synapses, signed)" items={detail.outputs} onSelect={onSelect} />
          <PartnerList title="Strongest inputs (synapses, signed)" items={detail.inputs} onSelect={onSelect} />
        </div>
      )}
      <div>
        <div className="text-[11px] uppercase tracking-[0.12em] text-faint mb-1">Highlight cell class</div>
        <div className="flex flex-wrap gap-1">
          <Button size="sm" variant="ghost" onClick={() => onHighlightCluster(null)}>none</Button>
          {clusterNames.map((n, i) => (
            <button key={n} onClick={() => onHighlightCluster(i)} className="h-6 px-1.5 rounded text-[11px] border border-line hover:border-line-2 flex items-center gap-1">
              <span className="w-2 h-2 rounded-full" style={{ background: `#${PALETTE[i % PALETTE.length].getHexString()}` }} />
              {n}
              <span className="text-faint">{info?.clusters[i]?.size}</span>
            </button>
          ))}
        </div>
      </div>
    </Card>
  );
}

function Mini({ label, v }: { label: string; v: number | string }) {
  return (
    <div className="rounded border border-line px-2 py-1">
      <div className="text-[10px] text-faint">{label}</div>
      <div className="num">{v}</div>
    </div>
  );
}

function PartnerList({ title, items, onSelect }: { title: string; items: NeuronDetail["outputs"]; onSelect: (i: number) => void }) {
  return (
    <div>
      <div className="text-[11px] uppercase tracking-[0.12em] text-faint mb-1">{title}</div>
      <div className="rounded border border-line divide-y divide-line">
        {items.map((p) => (
          <button key={p.idx} onClick={() => onSelect(p.idx)} className="w-full flex items-center justify-between px-2 h-6 hover:bg-panel-2 text-left">
            <span className="truncate">
              {p.cell_type || "untyped"} <span className="text-faint">{p.side}</span>
            </span>
            <span className={`num ${p.synapses < 0 ? "text-bad" : "text-good"}`}>{p.synapses > 0 ? `+${p.synapses}` : p.synapses}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
