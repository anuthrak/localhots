import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Two pages: the datathon map (/) and the finer-grid map (/norcal/).
export default defineConfig({
  plugins: [react()],
  build: { rollupOptions: { input: { main: "index.html", norcal: "norcal/index.html" } } },
});
