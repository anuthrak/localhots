import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Three pages: the datathon map (/), the detailed map (/norcal/) and lightning fires (/lightning/).
export default defineConfig({
  plugins: [react()],
  build: { rollupOptions: { input: { main: "index.html", norcal: "norcal/index.html", lightning: "lightning/index.html" } } },
});
