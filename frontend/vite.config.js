import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/auth": "http://api:5000",
      "/mdm": "http://api:5000",
      "/api": "http://api:5000",
      "/ingest": "http://api:5000",
    },
  },
});
