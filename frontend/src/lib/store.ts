import { useSyncExternalStore } from "react";
import type { Frame, SimEvent, World } from "./types";

/** Seconds of live history kept for charts. */
const HISTORY_S = 60;
const MAX_EVENTS = 800;

export interface History {
  t: number[];
  energy: number[];
  health: number[];
  speed: number[];
  spikes: number[];
  forward: number[];
  turn: number[];
  proboscis: number[];
  motor: Record<string, number[]>;
  sensory: Record<string, number[]>;
  clusters: number[][];
  trail: [number, number][];
}

export interface SimState {
  connected: boolean;
  frame: Frame | null;
  world: World | null;
  events: SimEvent[];
  history: History;
  version: number;
}

function emptyHistory(): History {
  return { t: [], energy: [], health: [], speed: [], spikes: [], forward: [], turn: [], proboscis: [], motor: {}, sensory: {}, clusters: [], trail: [] };
}

let state: SimState = { connected: false, frame: null, world: null, events: [], history: emptyHistory(), version: 0 };
const listeners = new Set<() => void>();
/** Subscribers that need every frame without React re-rendering (3D views). */
const frameListeners = new Set<(f: Frame) => void>();

function emit() {
  state = { ...state, version: state.version + 1 };
  listeners.forEach((l) => l());
}

function push(arr: number[], v: number) {
  arr.push(v);
}

function ingest(f: Frame) {
  const h = state.history;
  const prevT = h.t.length ? h.t[h.t.length - 1] : -1;
  if (f.t < prevT - 1e-6) {
    // simulation was reset or restored: history no longer applies
    state.history = emptyHistory();
    state.events = [];
  }
  const hh = state.history;
  if (!hh.t.length || f.t > hh.t[hh.t.length - 1]) {
    push(hh.t, f.t);
    push(hh.energy, f.organism.energy);
    push(hh.health, f.organism.health);
    push(hh.speed, f.body.speed);
    push(hh.spikes, f.brain.spikes_tick);
    push(hh.forward, f.motor.command.forward);
    push(hh.turn, f.motor.command.turn);
    push(hh.proboscis, f.motor.command.proboscis);
    for (const [k, v] of Object.entries(f.motor.rates)) (hh.motor[k] ??= []).push(v);
    for (const [k, v] of Object.entries(f.sensory.rates)) (hh.sensory[k] ??= []).push(v);
    hh.clusters.push(f.brain.clusters);
    hh.trail.push([f.body.position[0], f.body.position[1]]);
    const cutoff = f.t - HISTORY_S;
    let drop = 0;
    while (drop < hh.t.length && hh.t[drop] < cutoff) drop++;
    if (drop > 0) {
      for (const key of ["t", "energy", "health", "speed", "spikes", "forward", "turn", "proboscis", "clusters"] as const) hh[key].splice(0, drop);
      for (const arr of Object.values(hh.motor)) arr.splice(0, drop);
      for (const arr of Object.values(hh.sensory)) arr.splice(0, drop);
    }
    if (hh.trail.length > 3000) hh.trail.splice(0, hh.trail.length - 3000);
  }
  if (f.events.length) {
    state.events = [...state.events, ...f.events].slice(-MAX_EVENTS);
  }
  state.frame = f;
  frameListeners.forEach((l) => l(f));
  emit();
}

let socket: WebSocket | null = null;
let retry: number | undefined;

export function connect() {
  if (socket) return;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  socket = ws;
  ws.onopen = () => {
    state.connected = true;
    emit();
  };
  ws.onclose = () => {
    socket = null;
    state.connected = false;
    emit();
    window.clearTimeout(retry);
    retry = window.setTimeout(connect, 1500);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data as string);
    if (msg.type === "world") {
      state.world = msg.world as World;
      emit();
    } else if (msg.type === "frame") {
      ingest(msg as Frame);
    }
  };
}

export function useSim(): SimState {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => state,
  );
}

export function onFrame(cb: (f: Frame) => void): () => void {
  frameListeners.add(cb);
  return () => frameListeners.delete(cb);
}

export function getState(): SimState {
  return state;
}
