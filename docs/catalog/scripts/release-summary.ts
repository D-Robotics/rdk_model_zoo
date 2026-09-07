import type { BenchmarkRecord, ModelRecord } from "../src/catalog/types";

export function validateReleaseSummary(models: Array<Omit<ModelRecord, "benchmarks">>, benchmarks: BenchmarkRecord[], declared: Record<string, number> = {}): void {
  const assets = models.flatMap(model => model.assets);
  const actual: Record<string, number> = {
    sample_count: models.length,
    download_script_count: models.flatMap(model => model.download_scripts).length,
    asset_count: assets.length,
    downloadable_asset_count: assets.filter(asset => asset.url).length,
    manual_asset_count: assets.filter(asset => !asset.url).length,
    sha256_recorded_count: assets.filter(asset => asset.sha256).length,
    sha256_unrecorded_count: assets.filter(asset => !asset.sha256).length,
    benchmark_count: benchmarks.length,
    performance_metric_count: benchmarks.flatMap(record => record.performance ?? []).length,
    accuracy_metric_count: benchmarks.flatMap(record => record.accuracy ?? []).length
  };
  for (const [key, count] of Object.entries(actual)) {
    if (declared[key] !== undefined && declared[key] !== count) {
      throw new Error(`Release summary ${key}: declared ${declared[key]}, actual ${count}`);
    }
  }
}
