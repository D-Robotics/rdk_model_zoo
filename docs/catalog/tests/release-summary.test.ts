import { expect, it } from "vitest";
import { validateReleaseSummary } from "../scripts/release-summary";
import { benchmarkFixture, createModelFixture } from "./fixtures/catalog";

it("rejects stale release benchmark totals before building the website", () => {
  expect(() => validateReleaseSummary([createModelFixture()], [benchmarkFixture()], { benchmark_count: 58 }))
    .toThrow("declared 58, actual 1");
});
it("counts measurements separately from benchmark records", () => {
  const record = benchmarkFixture();
  expect(() => validateReleaseSummary([createModelFixture()], [record], {
    benchmark_count: 1, accuracy_metric_count: record.accuracy?.length ?? 0,
    performance_metric_count: record.performance?.length ?? 0
  })).not.toThrow();
});
