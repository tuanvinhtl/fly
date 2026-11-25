export const fmtTime = (s: number) => {
  const m = Math.floor(s / 60);
  const r = s - m * 60;
  return `${m}:${r.toFixed(1).padStart(4, "0")}`;
};

export const prettyKey = (k: string) =>
  k
    .replace(/_left$/, " L")
    .replace(/_right$/, " R")
    .replace(/_center$/, " C")
    .replace(/^dn_/, "DN ")
    .replace(/^mn_/, "MN ")
    .replace(/_/g, " ");

export const MOTOR_LABEL: Record<string, string> = {
  dn_forward: "DNp09 · forward",
  dn_steer_high: "DNa02 · steer",
  dn_steer_low: "DNa01 · steer",
  dn_backward: "MDN · backward",
  dn_escape: "DNp01 · escape",
  mn_proboscis: "MN9 · proboscis",
};

export const CATEGORY_COLOR: Record<string, string> = {
  sensory: "#38bdf8",
  neural: "#a78bfa",
  motor: "#34d399",
  movement: "#2dd4bf",
  environment: "#fbbf24",
  organism: "#fb7185",
  experiment: "#94a3b8",
};
