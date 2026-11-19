import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

// En desarrollo, /api se reenvía al backend; en producción el backend sirve este build
// desde el mismo origen, así que no hace falta CORS en ningún caso.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");
  return {
    plugins: [react()],
    server: {
      proxy: { "/api": env.API_PROXY_TARGET || "http://localhost:8000" },
    },
  };
});
