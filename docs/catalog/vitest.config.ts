import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // Integration suites each load three Git-backed manifests; bound memory
    // and process contention instead of weakening their timeout assertions.
    maxWorkers: 2,
    environment: "jsdom",
    setupFiles: ["./tests/setup.ts"]
  }
});
