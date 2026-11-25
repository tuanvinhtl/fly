import { useEffect, useState } from "react";
import { api } from "./api";
import type { BrainInfo, GeomSpec } from "./types";

const cache = new Map<string, Promise<unknown>>();

function useCached<T>(key: string, load: () => Promise<T>, deps: unknown[] = []): { data: T | null; error: string | null } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    if (!cache.has(key)) cache.set(key, load());
    (cache.get(key) as Promise<T>)
      .then((d) => alive && setData(d))
      .catch((e: Error) => {
        cache.delete(key);
        if (alive) setError(e.message);
      });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, ...deps]);
  return { data, error };
}

export function invalidate(key: string) {
  cache.delete(key);
}

export const useBrainInfo = (preset?: string) => useCached<BrainInfo>(`brain-info-${preset ?? ""}`, () => api.get<BrainInfo>("/api/brain/info"), [preset]);
export const useBodyModel = () => useCached<{ geoms: GeomSpec[] }>("body-model", () => api.get("/api/body/model"));
