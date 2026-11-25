async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method,
    headers: body === undefined ? undefined : { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* body was not JSON; keep statusText */
    }
    throw new Error(`${method} ${path}: ${detail}`);
  }
  return (await res.json()) as T;
}

export const api = {
  get: <T,>(p: string) => request<T>("GET", p),
  post: <T,>(p: string, b: unknown = {}) => request<T>("POST", p, b),
  patch: <T,>(p: string, b: unknown) => request<T>("PATCH", p, b),
  del: <T,>(p: string) => request<T>("DELETE", p),
};

export async function fetchLayout(): Promise<{ n: number; positions: Float32Array; clusters: Uint8Array }> {
  const buf = await (await fetch("/api/brain/layout")).arrayBuffer();
  const n = new Uint32Array(buf, 0, 1)[0];
  const positions = new Float32Array(buf.slice(4, 4 + n * 12));
  const clusters = new Uint8Array(buf.slice(4 + n * 12, 4 + n * 13));
  return { n, positions, clusters };
}
