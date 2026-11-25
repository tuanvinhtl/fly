/* Detailed procedural Drosophila rendered on top of the MuJoCo body.

   Every part is attached to the pose of a physics geom streamed from the backend, so
   what you see is exactly what the simulator's body is doing. The physics primitives
   (capsules, ellipsoids) are replaced by nicer meshes: faceted red compound eyes, a
   banded abdomen, veined translucent wings, thin segmented legs, antennae with aristae
   and thoracic bristles. In flight the wings are drawn as a motion-blurred stroke. */
import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { getState } from "../lib/store";
import type { GeomSpec } from "../lib/types";

function canvasTexture(w: number, h: number, draw: (g: CanvasRenderingContext2D) => void): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  draw(c.getContext("2d")!);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}

/** Hexagonal ommatidia pattern for the compound eyes. */
function eyeTexture() {
  return canvasTexture(512, 256, (g) => {
    g.fillStyle = "#7a0d0d";
    g.fillRect(0, 0, 512, 256);
    const r = 5;
    for (let y = 0; y < 256 + r; y += r * 1.5) {
      for (let x = 0; x < 512 + r; x += r * 1.732) {
        const ox = (Math.round(y / (r * 1.5)) % 2) * r * 0.866;
        const cx = x + ox;
        const grad = g.createRadialGradient(cx - 1, y - 1, 0.5, cx, y, r);
        grad.addColorStop(0, "#ff5a3c");
        grad.addColorStop(0.55, "#c4161a");
        grad.addColorStop(1, "#5c0808");
        g.fillStyle = grad;
        g.beginPath();
        for (let k = 0; k < 6; k++) {
          const a = (Math.PI / 3) * k + Math.PI / 6;
          g.lineTo(cx + r * 0.95 * Math.cos(a), y + r * 0.95 * Math.sin(a));
        }
        g.fill();
      }
    }
  });
}

/** Tan abdomen with dark posterior tergite bands (female Drosophila pattern). */
function abdomenTexture() {
  return canvasTexture(64, 512, (g) => {
    g.fillStyle = "#c99a5e";
    g.fillRect(0, 0, 64, 512);
    // v runs tip -> base after rotating the sphere so its poles lie on the body axis
    const bands = [0.18, 0.3, 0.42, 0.54, 0.66, 0.77];
    for (const b of bands) {
      const y = b * 512;
      const grad = g.createLinearGradient(0, y - 26, 0, y + 10);
      grad.addColorStop(0, "rgba(40,22,10,0)");
      grad.addColorStop(0.55, "rgba(48,26,12,0.95)");
      grad.addColorStop(1, "rgba(48,26,12,0.95)");
      g.fillStyle = grad;
      g.fillRect(0, y - 26, 64, 36);
    }
    g.fillStyle = "rgba(30,16,8,0.9)";
    g.fillRect(0, 0, 64, 70); // dark tip
  });
}

/** Translucent membrane with longitudinal and cross veins. */
function wingTexture() {
  return canvasTexture(512, 192, (g) => {
    g.clearRect(0, 0, 512, 192);
    g.fillStyle = "rgba(210,225,240,0.22)";
    g.fillRect(0, 0, 512, 192);
    g.strokeStyle = "rgba(70,55,40,0.85)";
    g.lineCap = "round";
    const vein = (pts: [number, number][], w: number) => {
      g.lineWidth = w;
      g.beginPath();
      pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
      g.stroke();
    };
    // x = 512 at the hinge, x = 0 at the tip (matches geometry UVs)
    vein([[512, 96], [300, 40], [40, 30]], 3); // costa / R1
    vein([[512, 96], [260, 70], [10, 70]], 2.2); // R2+3
    vein([[512, 98], [250, 100], [0, 104]], 2.2); // R4+5
    vein([[512, 100], [260, 128], [30, 150]], 2); // M
    vein([[500, 104], [330, 150], [180, 178]], 1.8); // CuA
    vein([[270, 70], [262, 100]], 1.6); // r-m crossvein
    vein([[190, 103], [180, 140]], 1.6); // dm-cu crossvein
  });
}

function wingGeometry(): THREE.BufferGeometry {
  // unit blade from hinge (x = +1) to tip (x = -1); broadest near the tip like Drosophila
  const s = new THREE.Shape();
  const pts: THREE.Vector2[] = [];
  for (let i = 0; i <= 64; i++) {
    const t = (i / 64) * Math.PI * 2;
    const x = Math.cos(t);
    const width = 0.55 + 0.45 * (1 - x) * 0.5; // narrow at hinge, wide at tip
    pts.push(new THREE.Vector2(x, Math.sin(t) * width));
  }
  s.setFromPoints(pts);
  const geo = new THREE.ShapeGeometry(s, 48);
  const uv = geo.attributes.uv as THREE.BufferAttribute;
  const pos = geo.attributes.position as THREE.BufferAttribute;
  for (let i = 0; i < pos.count; i++) uv.setXY(i, (pos.getX(i) + 1) / 2, (pos.getY(i) + 1) / 2);
  return geo;
}

interface Part {
  name: string;
  object: THREE.Object3D;
}

function buildParts(geoms: GeomSpec[]): { parts: Part[]; blur: THREE.Group } {
  const byName = Object.fromEntries(geoms.map((g) => [g.name, g]));
  const cuticle = new THREE.MeshPhysicalMaterial({ color: "#b07a45", roughness: 0.42, clearcoat: 0.6, clearcoatRoughness: 0.35, sheen: 0.4, sheenColor: new THREE.Color("#ffd9a8") });
  const dark = new THREE.MeshStandardMaterial({ color: "#3a2414", roughness: 0.6 });
  const legMat = new THREE.MeshPhysicalMaterial({ color: "#6b4426", roughness: 0.5, clearcoat: 0.3 });
  const bristle = new THREE.MeshBasicMaterial({ color: "#1a0f08" });
  const eyeMat = new THREE.MeshPhysicalMaterial({ map: eyeTexture(), roughness: 0.25, clearcoat: 1, clearcoatRoughness: 0.15, sheen: 0.6, sheenColor: new THREE.Color("#ff9a7a") });
  const abdMat = new THREE.MeshPhysicalMaterial({ map: abdomenTexture(), roughness: 0.45, clearcoat: 0.5, clearcoatRoughness: 0.4 });
  const wingTex = wingTexture();
  const wingMat = new THREE.MeshPhysicalMaterial({
    map: wingTex, transparent: true, side: THREE.DoubleSide, depthWrite: false, roughness: 0.15, iridescence: 0.8, iridescenceIOR: 1.6, clearcoat: 1,
  });
  const parts: Part[] = [];
  const add = (name: string, object: THREE.Object3D) => parts.push({ name, object });

  for (const g of geoms) {
    const s = g.size;
    if (g.name === "thorax") {
      const grp = new THREE.Group();
      const body = new THREE.Mesh(new THREE.SphereGeometry(1, 40, 28), cuticle);
      body.scale.set(s[0], s[1], s[2]);
      body.castShadow = true;
      grp.add(body);
      const scut = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 14), cuticle);
      scut.scale.set(s[0] * 0.35, s[1] * 0.45, s[2] * 0.25);
      scut.position.set(-s[0] * 0.75, 0, s[2] * 0.72);
      grp.add(scut);
      // dorsal macrochaetae
      const cone = new THREE.ConeGeometry(0.008, 0.2, 5);
      cone.translate(0, 0.1, 0);
      for (const [x, y] of [[0.25, 0.14], [0.25, -0.14], [-0.05, 0.2], [-0.05, -0.2], [-0.3, 0.12], [-0.3, -0.12], [-0.55, 0.06], [-0.55, -0.06]]) {
        const b = new THREE.Mesh(cone, bristle);
        const nx = x * s[0] * 1.3, ny = y * s[1] * 2.2;
        b.position.set(nx, ny, s[2] * Math.sqrt(Math.max(0.05, 1 - (nx / s[0]) ** 2 - (ny / s[1]) ** 2)));
        b.rotation.set(Math.sign(y) * -0.5, -1.9, 0); // lean backward
        grp.add(b);
      }
      add(g.name, grp);
    } else if (g.name === "head") {
      const grp = new THREE.Group();
      const head = new THREE.Mesh(new THREE.SphereGeometry(s[0] * 0.92, 32, 24), cuticle);
      head.scale.set(0.8, 1, 0.95);
      head.castShadow = true;
      grp.add(head);
      // ocelli triangle on the vertex
      for (const [x, y] of [[0.02, 0], [-0.04, 0.05], [-0.04, -0.05]]) {
        const o = new THREE.Mesh(new THREE.SphereGeometry(0.022, 10, 8), new THREE.MeshPhysicalMaterial({ color: "#2b0a06", clearcoat: 1, roughness: 0.1 }));
        o.position.set(x, y, s[0] * 0.9);
        grp.add(o);
      }
      // antennae: scape/pedicel, funiculus, plumose arista
      for (const side of [1, -1]) {
        const a = new THREE.Group();
        a.position.set(s[0] * 0.78, side * 0.06, s[0] * 0.25);
        const ped = new THREE.Mesh(new THREE.CapsuleGeometry(0.035, 0.06, 4, 8), cuticle);
        ped.rotation.set(0, 0, 0);
        a.add(ped);
        const fun = new THREE.Mesh(new THREE.SphereGeometry(1, 14, 10), cuticle);
        fun.scale.set(0.06, 0.05, 0.09);
        fun.position.set(0.04, 0, -0.1);
        a.add(fun);
        const arista = new THREE.Group();
        arista.position.set(0.06, side * 0.02, -0.08);
        arista.rotation.set(side * 0.6, -0.4, 0);
        const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.004, 0.007, 0.32, 4), bristle);
        shaft.position.y = 0.16;
        arista.add(shaft);
        for (let k = 1; k < 7; k++) {
          const br = new THREE.Mesh(new THREE.CylinderGeometry(0.002, 0.002, 0.07, 3), bristle);
          br.position.set(0, k * 0.045, 0);
          br.rotation.z = k % 2 ? 0.9 : -0.9;
          arista.add(br);
        }
        a.add(arista);
        grp.add(a);
      }
      add(g.name, grp);
    } else if (g.name === "eye_L" || g.name === "eye_R") {
      const m = new THREE.Mesh(new THREE.SphereGeometry(1, 40, 28), eyeMat);
      m.scale.set(s[0] * 1.08, s[1] * 1.15, s[2] * 1.08);
      add(g.name, m);
    } else if (g.name === "abdomen") {
      const geo = new THREE.SphereGeometry(1, 40, 32);
      geo.rotateZ(Math.PI / 2); // poles on the body axis so texture bands run around it
      const m = new THREE.Mesh(geo, abdMat);
      m.scale.set(s[0], s[1], s[2]);
      m.castShadow = true;
      add(g.name, m);
    } else if (g.name === "wing_L" || g.name === "wing_R") {
      const m = new THREE.Mesh(wingGeometry(), wingMat);
      m.scale.set(s[0], s[1] * 1.15, 1);
      if (g.name === "wing_R") m.scale.y *= -1;
      m.renderOrder = 2;
      add(g.name, m);
    } else if (g.type === "capsule" && (g.name.endsWith("_femur") || g.name.endsWith("_tibia"))) {
      const r = g.name.endsWith("_femur") ? 0.028 : 0.02;
      const grp = new THREE.Group();
      const seg = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.8, r, 2 * s[1], 8), legMat);
      seg.rotation.x = Math.PI / 2;
      seg.castShadow = true;
      grp.add(seg);
      const joint = new THREE.Mesh(new THREE.SphereGeometry(r * 1.15, 8, 6), dark);
      joint.position.z = -s[1];
      grp.add(joint);
      add(g.name, grp);
    } else if (g.name.endsWith("_tarsus")) {
      // tarsal segments + claws
      const grp = new THREE.Group();
      for (let k = 0; k < 3; k++) {
        const t = new THREE.Mesh(new THREE.SphereGeometry(0.022 - k * 0.003, 8, 6), dark);
        t.position.set(0, 0, -0.01 - k * 0.025);
        grp.add(t);
      }
      add(g.name, grp);
    } else if (g.name === "proboscis") {
      const m = new THREE.Mesh(new THREE.CapsuleGeometry(s[0] * 0.8, 2 * s[1], 4, 8), new THREE.MeshPhysicalMaterial({ color: "#d8b48a", roughness: 0.5 }));
      m.rotation.x = Math.PI / 2;
      const grp = new THREE.Group();
      grp.add(m);
      const lab = new THREE.Mesh(new THREE.SphereGeometry(s[0] * 1.3, 10, 8), new THREE.MeshPhysicalMaterial({ color: "#e6c49c", roughness: 0.4 }));
      lab.position.z = -s[1];
      grp.add(lab);
      add(g.name, grp);
    }
  }
  void byName;

  // motion-blurred wing stroke shown in flight (attached to the thorax)
  const blur = new THREE.Group();
  const blurTex = canvasTexture(256, 256, (g) => {
    const grad = g.createRadialGradient(128, 128, 10, 128, 128, 128);
    grad.addColorStop(0, "rgba(220,230,245,0.32)");
    grad.addColorStop(0.75, "rgba(200,215,235,0.14)");
    grad.addColorStop(1, "rgba(200,215,235,0)");
    g.fillStyle = grad;
    g.fillRect(0, 0, 256, 256);
  });
  const blurMat = new THREE.MeshBasicMaterial({ map: blurTex, transparent: true, depthWrite: false, side: THREE.DoubleSide });
  // left stroke sweeps from straight back (pi) to ~30 deg forward of lateral; right mirrors it
  for (const [side, start] of [[1, Math.PI / 3], [-1, Math.PI * 0.92]] as const) {
    const sector = new THREE.Mesh(new THREE.CircleGeometry(1.6, 40, start, 0.75 * Math.PI), blurMat);
    sector.position.set(0.0, side * 0.18, 0.32);
    sector.userData.side = side;
    blur.add(sector);
  }
  blur.visible = false;
  return { parts, blur };
}

export default function FlyModel({ geoms }: { geoms: GeomSpec[] }) {
  const root = useRef<THREE.Group>(null);
  // one memo so the parts we pose every frame are exactly the ones in the rendered group
  // (separate memos can pair objects from different StrictMode double-invocations)
  const { parts, blur, scene } = useMemo(() => {
    const built = buildParts(geoms);
    const g = new THREE.Group();
    for (const p of built.parts) g.add(p.object);
    return { ...built, scene: g };
  }, [geoms]);
  const q = useMemo(() => new THREE.Quaternion(), []);

  useFrame(({ clock }) => {
    const f = getState().frame;
    if (!f) return;
    const flying = f.motor.command.flying;
    for (const p of parts) {
      const pose = f.body.geoms[p.name];
      if (!pose) continue;
      const [pos, quat] = pose;
      p.object.position.set(pos[0], pos[1], pos[2]);
      q.set(quat[1], quat[2], quat[3], quat[0]);
      p.object.quaternion.copy(q);
      if (p.name.startsWith("wing")) p.object.visible = !flying;
    }
    const th = f.body.geoms["thorax"];
    if (th) {
      blur.position.set(th[0][0], th[0][1], th[0][2]);
      blur.quaternion.set(th[1][1], th[1][2], th[1][3], th[1][0]);
    }
    blur.visible = flying;
    if (flying) {
      // flicker the stroke to suggest ~200 Hz wingbeats
      const k = 0.85 + 0.15 * Math.sin(clock.elapsedTime * 90);
      blur.children.forEach((c) => c.scale.set(k, k, 1));
    }
  });

  return (
    <group ref={root}>
      <primitive object={scene} />
      <primitive object={blur} />
    </group>
  );
}
