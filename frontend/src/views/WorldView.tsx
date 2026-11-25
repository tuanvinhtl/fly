import { useEffect, useMemo, useRef, useState, type ComponentRef } from "react";
import { Canvas, useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { Grid, Line, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import clsx from "clsx";
import { Crosshair, Eye, MousePointer2, Trash2 } from "lucide-react";
import { getState, useSim } from "../lib/store";
import { api } from "../lib/api";
import { useBodyModel } from "../lib/useStatic";
import type { GeomSpec, World } from "../lib/types";
import { Badge, Button, Card } from "../components/ui";

type Tool = "select" | "food" | "obstacle" | "light" | "hot" | "cold" | "toxic" | "remove";

const TOOLS: { id: Tool; label: string }[] = [
  { id: "select", label: "Inspect" },
  { id: "food", label: "Food" },
  { id: "obstacle", label: "Obstacle" },
  { id: "light", label: "Light" },
  { id: "hot", label: "Hot zone" },
  { id: "cold", label: "Cold zone" },
  { id: "toxic", label: "Toxic" },
  { id: "remove", label: "Remove" },
];

export default function WorldView({ onError }: { onError: (e: string | null) => void }) {
  const [tool, setTool] = useState<Tool>("select");
  const [follow, setFollow] = useState(true);
  const { world, frame } = useSim();
  const { data: model } = useBodyModel();

  const place = (x: number, y: number) => {
    const call = (kind: string, obj: object) => api.post(`/api/world/${kind}`, obj).then(() => onError(null)).catch((e: Error) => onError(e.message));
    if (tool === "food") call("food", { x, y });
    if (tool === "obstacle") call("obstacles", { x, y, sx: 1.5, sy: 1.5 });
    if (tool === "light") call("lights", { x, y, intensity: 2 });
    if (tool === "hot") call("zones", { kind: "temperature", x, y, radius: 6, temperature: 38 });
    if (tool === "cold") call("zones", { kind: "temperature", x, y, radius: 6, temperature: 14 });
    if (tool === "toxic") call("zones", { kind: "toxic", x, y, radius: 4, bitter: 1 });
  };
  const remove = (id: string) => {
    if (tool !== "remove") return;
    api.del(`/api/world/${id}`).then(() => onError(null)).catch((e: Error) => onError(e.message));
  };

  return (
    <div className="h-full flex flex-col">
      <div className="h-11 shrink-0 border-b border-line flex items-center gap-1 px-3">
        {TOOLS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTool(t.id)}
            className={clsx(
              "h-7 px-2.5 rounded-md text-xs flex items-center gap-1",
              tool === t.id ? "bg-accent/15 text-accent border border-accent/40" : "text-dim hover:text-ink border border-transparent",
            )}
          >
            {t.id === "select" && <MousePointer2 size={12} />}
            {t.id === "remove" && <Trash2 size={12} />}
            {t.label}
          </button>
        ))}
        <span className="text-[11px] text-faint ml-2">{tool !== "select" && tool !== "remove" ? "click the floor to place" : tool === "remove" ? "click an object to remove it" : ""}</span>
        <div className="ml-auto flex gap-1">
          <Button size="sm" variant={follow ? "accent" : "ghost"} onClick={() => setFollow(true)}>
            <Crosshair size={12} /> Follow fly
          </Button>
          <Button size="sm" variant={!follow ? "accent" : "ghost"} onClick={() => setFollow(false)}>
            <Eye size={12} /> Arena
          </Button>
        </div>
      </div>
      <div className="flex-1 min-h-0 relative">
        <Canvas camera={{ position: [6, 9, 10], fov: 45, near: 0.05, far: 2000 }} shadows dpr={[1, 2]}>
          <color attach="background" args={["#05070a"]} />
          <fog attach="fog" args={["#05070a", 60, 200]} />
          <ambientLight intensity={0.35} />
          <directionalLight position={[30, 60, 20]} intensity={1.1} castShadow shadow-mapSize={[2048, 2048]} />
          <CameraRig follow={follow} />
          {/* simulation is z-up; rotate into three.js y-up */}
          <group rotation={[-Math.PI / 2, 0, 0]}>
            {world && <Arena world={world} onPlace={place} onRemove={remove} tool={tool} />}
            {model && <Fly geoms={model.geoms} />}
            <Trail />
          </group>
        </Canvas>
        <div className="absolute left-3 bottom-3 flex gap-2 pointer-events-none">
          <Legend />
        </div>
        {frame && !frame.organism.alive && (
          <div className="absolute inset-x-0 top-6 flex justify-center pointer-events-none">
            <div className="rounded-md border border-bad/40 bg-bad/10 text-bad px-4 py-2 text-sm">
              Organism died ({frame.organism.death_cause}) after {frame.organism.age.toFixed(1)} s — reset to start a new experiment
            </div>
          </div>
        )}
        <div className="absolute right-3 top-3 w-64">
          <SensePanel />
        </div>
      </div>
    </div>
  );
}

function CameraRig({ follow }: { follow: boolean }) {
  const controls = useRef<ComponentRef<typeof OrbitControls>>(null);
  const { camera } = useThree();
  const target = useRef(new THREE.Vector3());
  useEffect(() => {
    const w = getState().world;
    if (!follow && controls.current) {
      const a = w?.arena_half ?? 40;
      camera.position.set(0, a * 2.1, a * 1.2);
      controls.current.target.set(0, 0, 0);
    }
    if (follow) camera.position.set(target.current.x + 6, 7, target.current.z + 8);
  }, [follow, camera]);
  useFrame(() => {
    const f = getState().frame;
    if (!f || !controls.current || !follow) return;
    const [x, y] = f.body.position;
    const goal = new THREE.Vector3(x, 0.5, -y);
    const delta = goal.clone().sub(target.current);
    target.current.add(delta.multiplyScalar(0.15));
    // move camera with the target so the user's orbit offset is preserved
    camera.position.add(delta);
    controls.current.target.copy(target.current);
    controls.current.update();
  });
  return <OrbitControls ref={controls} makeDefault enableDamping dampingFactor={0.12} maxPolarAngle={Math.PI * 0.49} />;
}

function geometryFor(g: GeomSpec): THREE.BufferGeometry {
  const s = g.size;
  switch (g.type) {
    case "sphere":
      return new THREE.SphereGeometry(s[0], 24, 16);
    case "ellipsoid": {
      const geo = new THREE.SphereGeometry(1, 28, 18);
      geo.scale(s[0], s[1], s[2]);
      return geo;
    }
    case "capsule": {
      const geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 6, 12);
      geo.rotateX(Math.PI / 2); // MuJoCo capsule axis = local z
      return geo;
    }
    case "box":
      return new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
    default:
      return new THREE.SphereGeometry(s[0] || 0.05, 12, 8);
  }
}

function Fly({ geoms }: { geoms: GeomSpec[] }) {
  const refs = useRef<Record<string, THREE.Mesh | null>>({});
  const items = useMemo(
    () =>
      geoms.map((g) => ({
        g,
        geo: geometryFor(g),
        mat: new THREE.MeshStandardMaterial({
          color: new THREE.Color(g.rgba[0], g.rgba[1], g.rgba[2]),
          transparent: g.rgba[3] < 1,
          opacity: g.rgba[3],
          roughness: g.name.startsWith("wing") ? 0.2 : 0.6,
          metalness: g.name.startsWith("eye") ? 0.3 : 0.05,
          side: g.name.startsWith("wing") ? THREE.DoubleSide : THREE.FrontSide,
          depthWrite: g.rgba[3] >= 1,
        }),
      })),
    [geoms],
  );
  const q = useMemo(() => new THREE.Quaternion(), []);
  useFrame(() => {
    const f = getState().frame;
    if (!f) return;
    for (const [name, [p, quat]] of Object.entries(f.body.geoms)) {
      const m = refs.current[name];
      if (!m) continue;
      m.position.set(p[0], p[1], p[2]);
      q.set(quat[1], quat[2], quat[3], quat[0]);
      m.quaternion.copy(q);
    }
  });
  return (
    <group>
      {items.map(({ g, geo, mat }) => (
        <mesh key={g.name} ref={(m) => void (refs.current[g.name] = m)} geometry={geo} material={mat} castShadow />
      ))}
    </group>
  );
}

function Trail() {
  const [pts, setPts] = useState<[number, number, number][]>([]);
  useFrame(() => {
    const tr = getState().history.trail;
    if (tr.length < 2) return;
    if (tr.length !== pts.length) setPts(tr.slice(-1500).map(([x, y]) => [x, y, 0.03]));
  });
  if (pts.length < 2) return null;
  return <Line points={pts} color="#38bdf8" lineWidth={1.2} transparent opacity={0.45} />;
}

function Arena({ world, onPlace, onRemove, tool }: { world: World; onPlace: (x: number, y: number) => void; onRemove: (id: string) => void; tool: Tool }) {
  const a = world.arena_half;
  const click = (e: ThreeEvent<PointerEvent>) => {
    if (e.button !== 0 || tool === "select" || tool === "remove") return;
    e.stopPropagation();
    onPlace(Math.round(e.point.x * 10) / 10, Math.round(-e.point.z * 10) / 10);
  };
  const hit = (id: string) => (e: ThreeEvent<PointerEvent>) => {
    if (tool !== "remove") return;
    e.stopPropagation();
    onRemove(id);
  };
  return (
    <group>
      <mesh position={[0, 0, -0.01]} receiveShadow onPointerDown={click}>
        <planeGeometry args={[2 * a, 2 * a]} />
        <meshStandardMaterial color="#0d1218" roughness={0.95} />
      </mesh>
      <group rotation={[Math.PI / 2, 0, 0]}>
        <Grid args={[2 * a, 2 * a]} cellSize={1} sectionSize={10} cellColor="#141b25" sectionColor="#1f2a3a" fadeDistance={160} infiniteGrid={false} position={[0, 0.001, 0]} />
      </group>
      {[
        [0, a + 0.5, a + 1, 0.5],
        [0, -a - 0.5, a + 1, 0.5],
        [a + 0.5, 0, 0.5, a + 1],
        [-a - 0.5, 0, 0.5, a + 1],
      ].map(([x, y, sx, sy], i) => (
        <mesh key={i} position={[x, y, 1.5]} castShadow receiveShadow>
          <boxGeometry args={[2 * sx, 2 * sy, 3]} />
          <meshStandardMaterial color="#1a2230" />
        </mesh>
      ))}
      {world.zones.map((z) => (
        <mesh key={z.id} position={[z.x, z.y, 0.005]} onPointerDown={hit(z.id)}>
          <circleGeometry args={[z.radius, 48]} />
          <meshBasicMaterial
            color={z.kind === "toxic" ? "#a855f7" : z.temperature > world.ambient_temperature ? "#f43f5e" : "#3b82f6"}
            transparent
            opacity={z.kind === "toxic" ? 0.28 : 0.12 + Math.min(0.25, Math.abs(z.temperature - world.ambient_temperature) / 60)}
            depthWrite={false}
          />
        </mesh>
      ))}
      {world.food.map((f) => (
        <group key={f.id} position={[f.x, f.y, 0]}>
          <mesh position={[0, 0, 0.004]}>
            <circleGeometry args={[f.odor_sigma * 1.5, 48]} />
            <meshBasicMaterial color="#34d399" transparent opacity={f.amount > 0 ? 0.035 * f.odor : 0} depthWrite={false} />
          </mesh>
          <mesh position={[0, 0, 0.06]} rotation={[Math.PI / 2, 0, 0]} onPointerDown={hit(f.id)} receiveShadow>
            <cylinderGeometry args={[f.radius, f.radius, 0.12, 40]} />
            <meshStandardMaterial color={f.amount > 0 ? "#22c55e" : "#334155"} emissive={f.amount > 0 ? "#065f46" : "#000"} roughness={0.5} />
          </mesh>
        </group>
      ))}
      {world.obstacles.map((o) => (
        <mesh key={o.id} position={[o.x, o.y, o.height / 2]} castShadow receiveShadow onPointerDown={hit(o.id)}>
          <boxGeometry args={[2 * o.sx, 2 * o.sy, o.height]} />
          <meshStandardMaterial color="#556274" roughness={0.7} />
        </mesh>
      ))}
      {world.lights.map((l) => (
        <group key={l.id} position={[l.x, l.y, 4]}>
          <mesh onPointerDown={hit(l.id)}>
            <sphereGeometry args={[0.8, 20, 14]} />
            <meshBasicMaterial color="#fde68a" />
          </mesh>
          <pointLight intensity={l.intensity * 40} distance={l.d0 * 4} color="#fde68a" decay={1.5} />
        </group>
      ))}
    </group>
  );
}

function Legend() {
  const item = (c: string, l: string) => (
    <span className="flex items-center gap-1.5">
      <span className="w-2.5 h-2.5 rounded-full" style={{ background: c }} />
      {l}
    </span>
  );
  return (
    <div className="rounded-md bg-panel/85 border border-line px-2.5 py-1.5 text-[11px] text-dim flex gap-3 backdrop-blur">
      {item("#22c55e", "food (odour halo)")}
      {item("#f43f5e", "hot")}
      {item("#3b82f6", "cold")}
      {item("#a855f7", "toxic (bitter)")}
      {item("#fde68a", "light")}
      {item("#38bdf8", "trajectory")}
    </div>
  );
}

function SensePanel() {
  const { frame } = useSim();
  if (!frame) return null;
  const p = frame.sensory.physical;
  const r = frame.sensory.rates;
  const row = (label: string, l: number, rr: number, unit = "") => (
    <tr className="h-5">
      <td className="text-dim">{label}</td>
      <td className="num text-right">{l.toFixed(2)}{unit}</td>
      <td className="num text-right">{rr.toFixed(2)}{unit}</td>
    </tr>
  );
  return (
    <Card title="Sensed at the body" actions={<Badge tone="approximation">encoders</Badge>} bodyClass="p-2.5">
      <table className="w-full text-[11px]">
        <thead>
          <tr className="text-faint">
            <th className="text-left font-normal">stimulus</th>
            <th className="text-right font-normal">left</th>
            <th className="text-right font-normal">right</th>
          </tr>
        </thead>
        <tbody>
          {row("odour", p.odor_left ?? 0, p.odor_right ?? 0)}
          {row("temperature °C", p.temperature_left ?? 0, p.temperature_right ?? 0)}
          {row("light", p.light_left ?? 0, p.light_right ?? 0)}
          {row("sugar", p.sugar_left ?? 0, p.sugar_right ?? 0)}
          {row("bitter", p.bitter_left ?? 0, p.bitter_right ?? 0)}
          {row("head touch", p.touch_left ?? 0, p.touch_right ?? 0)}
        </tbody>
      </table>
      <div className="mt-2 text-[10px] text-faint">
        ORN drive {(r.odor_food_left ?? 0).toFixed(0)} / {(r.odor_food_right ?? 0).toFixed(0)} Hz · GRN sugar {(r.taste_sugar_left ?? 0).toFixed(0)} /{" "}
        {(r.taste_sugar_right ?? 0).toFixed(0)} Hz
      </div>
    </Card>
  );
}
