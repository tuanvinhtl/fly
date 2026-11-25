import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");
  const backend = env.VITE_BACKEND_URL || "http://localhost:8000";
  return {
    plugins: [react(), tailwindcss()],
    server: {
      host: true,
      port: 5173,
      proxy: {
        "/api": backend,
        "/ws": { target: backend.replace(/^http/, "ws"), ws: true },
      },
    },
  };
});
