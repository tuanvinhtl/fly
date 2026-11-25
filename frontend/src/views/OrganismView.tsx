import { useSim } from "../lib/store";
import { fmtTime, MOTOR_LABEL } from "../lib/format";
import { Badge, Card, Empty, Stat } from "../components/ui";
import { LineChart } from "../components/Chart";

export default function OrganismView() {
  const { frame: f, history: h, events } = useSim();
  if (!f) return <Empty>Waiting for simulation…</Empty>;
  const o = f.organism;
  const behaviour = events.filter((e) => e.category === "motor" || e.category === "organism" || e.name === "collision").slice(-40).reverse();
  const motorKeys = Object.keys(MOTOR_LABEL);
  const colors = ["#34d399", "#38bdf8", "#60a5fa", "#fbbf24", "#fb7185", "#a78bfa"];
  return (
    <div className="h-full overflow-y-auto p-3 grid grid-cols-3 gap-3 content-start">
      <Card title="Identity" className="col-span-1">
        <div className="grid grid-cols-2 gap-3">
          <Stat label="organism id" value={o.id} />
          <Stat label="species" value={<i className="text-sm">D. melanogaster</i>} />
          <Stat label="age (simulated)" value={fmtTime(o.age)} />
          <Stat label="status" value={o.alive ? (o.feeding ? "feeding" : "alive") : `dead · ${o.death_cause}`} tone={o.alive ? "good" : "bad"} />
        </div>
      </Card>
      <Card title="Physiology" actions={<Badge tone="approximation">model</Badge>} className="col-span-1">
        <div className="grid grid-cols-3 gap-3">
          <Stat label="energy" value={o.energy.toFixed(1)} tone={o.energy > 30 ? "good" : "warn"} />
          <Stat label="health" value={o.health.toFixed(1)} tone={o.health > 50 ? "accent" : "bad"} />
          <Stat label="hunger" value={(o.hunger * 100).toFixed(0)} unit="%" />
          <Stat label="food eaten" value={o.food_eaten.toFixed(1)} />
          <Stat label="locomotor drive" value={o.drive_hz.toFixed(0)} unit="Hz" />
        </div>
        <p className="text-[11px] text-faint mt-3">
          Drive is Poisson input to DNp09 scaled by hunger — a stand-in for internal state because the connectome model has no spontaneous activity.
        </p>
      </Card>
      <Card title="Kinematics" className="col-span-1">
        <div className="grid grid-cols-2 gap-3">
          <Stat label="position x, y" value={`${f.body.position[0].toFixed(1)}, ${f.body.position[1].toFixed(1)}`} unit="mm" />
          <Stat label="heading" value={((f.body.heading * 180) / Math.PI).toFixed(0)} unit="°" />
          <Stat label="speed" value={f.body.speed.toFixed(1)} unit="mm/s" />
          <Stat label="distance walked" value={o.distance.toFixed(1)} unit="mm" />
        </div>
      </Card>
      <Card title="Energy & health" className="col-span-2">
        <LineChart x={h.t} yMin={0} yMax={100} series={[{ label: "energy", values: h.energy, color: "#34d399" }, { label: "health", values: h.health, color: "#38bdf8" }]} />
      </Card>
      <Card title="Speed" className="col-span-1">
        <LineChart x={h.t} yMin={0} series={[{ label: "mm/s", values: h.speed, color: "#2dd4bf" }]} />
      </Card>
      <Card title="Motor neuron rates (left + right mean, Hz)" className="col-span-2">
        <LineChart
          x={h.t}
          yMin={0}
          series={motorKeys.map((k, i) => ({
            label: MOTOR_LABEL[k],
            values: (h.motor[`${k}_left`] ?? []).map((v, j) => (v + (h.motor[`${k}_right`]?.[j] ?? 0)) / 2),
            color: colors[i],
          }))}
          height={150}
        />
      </Card>
      <Card title="Behavioural history" className="col-span-1 row-span-2" bodyClass="max-h-96 overflow-y-auto">
        {behaviour.length === 0 ? (
          <Empty>No behaviours yet.</Empty>
        ) : (
          <ul className="text-xs flex flex-col gap-1">
            {behaviour.map((e, i) => (
              <li key={i} className="flex justify-between gap-2">
                <span className="num text-faint w-14 shrink-0">{e.t.toFixed(2)}s</span>
                <span className="flex-1 truncate">{e.name}</span>
              </li>
            ))}
          </ul>
        )}
      </Card>
      <Card title="VNC motor command" className="col-span-2">
        <LineChart
          x={h.t}
          yMin={-1}
          yMax={1}
          series={[
            { label: "forward (DNp09 − MDN)", values: h.forward, color: "#34d399" },
            { label: "turn (DNa02/01 L − R)", values: h.turn, color: "#fbbf24" },
            { label: "proboscis (MN9)", values: h.proboscis, color: "#a78bfa" },
          ]}
        />
      </Card>
    </div>
  );
}
