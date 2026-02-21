import { defineConfig } from "vite";

export default defineConfig({
  // GitHub Pages serves from /graphics_experiments/, so asset paths need this prefix
  base: "/graphics_experiments/",
  server: {
    open: true,          // Auto-open browser on `vite dev`
    port: 5199,          // Fixed dev server port
    strictPort: true,    // Fail if port 5199 is already in use (don't silently pick another)
  },
});
