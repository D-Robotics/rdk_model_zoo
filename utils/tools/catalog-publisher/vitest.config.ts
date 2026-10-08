import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
    // Every suite reads the same checked-out platform distributions; two
    // workers keep the git and filesystem traffic bounded.
    maxWorkers: 2,
    testTimeout: 120_000,
    hookTimeout: 120_000
  }
});
