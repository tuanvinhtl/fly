export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number]; // w x y z (MuJoCo)

export interface Organism {
  id: string;
  species: string;
  energy: number;
  health: number;
  age: number;
  alive: boolean;
  food_eaten: number;
  distance: number;
  feeding: boolean;
  death_cause: string | null;
  hunger: number;
  drive_hz: number;
}

export interface SimEvent {
  t: number;
  tick: number;
  category: "sensory" | "neural" | "motor" | "movement" | "environment" | "organism" | "experiment";
  name: string;
  data: Record<string, unknown>;
}

export interface Frame {
  type: "frame";
  seq: number;
  t: number;
  tick: number;
  running: boolean;
  speed: number;
  realtime_factor: number;
  experiment_id: number | null;
  world_seq: number;
  error: string | null;
  organism: Organism;
  body: { position: Vec3; heading: number; speed: number; geoms: Record<string, [Vec3, Quat]> };
  motor: { rates: Record<string, number>; command: { forward: number; turn: number; escape: boolean; proboscis: number } };
  sensory: { rates: Record<string, number>; physical: Record<string, number>; external: Record<string, number> };
  brain: { clusters: number[]; active_idx: number[]; active_cnt: number[]; spikes_tick: number; silenced: number };
  learning: { habituation: boolean; plasticity: boolean };
  events: SimEvent[];
}

export interface Food { id: string; x: number; y: number; radius: number; sugar: number; odor: number; odor_sigma: number; amount: number }
export interface Obstacle { id: string; x: number; y: number; sx: number; sy: number; height: number }
export interface Light { id: string; x: number; y: number; intensity: number; d0: number }
export interface Zone { id: string; kind: "temperature" | "toxic"; x: number; y: number; radius: number; temperature: number; bitter: number }
export interface World {
  arena_half: number;
  ambient_temperature: number;
  ambient_light: number;
  food: Food[];
  obstacles: Obstacle[];
  lights: Light[];
  zones: Zone[];
}

export interface Population { key: string; label: string; side: string; size: number; evidence: "validated" | "literature" | "approximation"; reference: string }
export interface BrainInfo {
  dataset: string;
  n_neurons: number;
  n_connections: number;
  n_synapses: number;
  preset: string;
  presets: Record<string, { name: string; description: string; deviation_from_publication: string }>;
  dt_ms: number;
  clusters: { name: string; size: number }[];
  sensory: Population[];
  motor: Population[];
}

export interface GeomSpec { name: string; type: "sphere" | "capsule" | "ellipsoid" | "box" | "cylinder"; size: number[]; rgba: number[] }

export interface Partner { idx: number; root_id: string; cell_type: string; cell_class: string; side: string; synapses: number }
export interface NeuronDetail {
  idx: number; root_id: string; super_class: string; cell_class: string; cell_sub_class: string; cell_type: string;
  top_nt: string; side: string; flow: string; position_um: (number | null)[]; n_outputs: number; n_inputs: number;
  spikes_total: number; membrane_mV: number; silenced: boolean; outputs: Partner[]; inputs: Partner[]; flywire_url: string;
}

export interface ExperimentRow {
  id: number; name: string; seed: number; preset: string; created_at: string; status: string; duration_s: number;
  summary: Record<string, number | string | boolean | null>;
}

export interface SampleRow {
  t: number; x: number; y: number; heading: number; speed: number; energy: number; health: number;
  forward: number; turn: number; proboscis: number; total_spikes: number; motor: Record<string, number>; sensory: Record<string, number>;
}
