import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  server: {
    cors: false,
    proxy: {
      "/get_game_information": {
        target: "https://backend-typmvzi2ya-uc.a.run.app",
        changeOrigin: true,
        secure: true,
      },
      "/get_N_recommendations_for_user": {
        target: "https://backend-typmvzi2ya-uc.a.run.app",
        changeOrigin: true,
        secure: true,
      },
    },
  },
  plugins: [react()],
});
