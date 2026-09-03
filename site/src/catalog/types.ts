export type Locale = "zh" | "en";
export type MetricUnit = "ms" | "fps" | "percent" | "ratio" | "mae" | "rmse";

export interface MetricRecord {
  metric: string;
  value: number;
  unit: MetricUnit;
  qualifier?: "exact" | "lower-bound" | "upper-bound" | "approximate";
  statistic?: "min" | "mean" | "p50" | "p95" | "max";
  scope?: string;
  concurrency?: number;
  dataset?: string;
  model_stage?: "float" | "quantized" | "compiled" | "runtime";
}

export interface BenchmarkRecord {
  id: string;
  sample_id: string;
  variant_id: string;
  display_name: string;
  asset_filename?: string;
  model_format?: string;
  precision?: string;
  input?: { shape?: number[]; layout?: string; format?: string };
  environment: {
    hardware: string;
    rdk_os?: string;
    runtime?: string;
    cpu_mode?: string;
    bpu_cores?: number;
  };
  performance?: MetricRecord[];
  accuracy?: MetricRecord[];
  source: {
    ref: string;
    path: string;
    section: string;
    provenance: "existing-repository-documentation";
  };
}

export interface ModelRecord {
  id: string;
  name: string;
  tasks: string[];
  sample_path: string;
  availability: "download" | "manual";
  download_scripts: string[];
  assets: Array<{ filename: string; format: string; url?: string; sha256?: string | null }>;
  benchmarks: BenchmarkRecord[];
}

export type CatalogSummary = Record<string, number>;

export interface Catalog {
  schema_version: number;
  release: Record<string, unknown> & { tag: string; platform: string; version: string };
  summary: CatalogSummary;
  models: ModelRecord[];
}
