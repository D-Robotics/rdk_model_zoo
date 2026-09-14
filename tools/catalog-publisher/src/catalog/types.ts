export type Locale = "zh" | "en";
export type HardwareId = "x3" | "x5" | "s100" | "s100p" | "s600";
export type MetricUnit = "ms" | "us" | "fps" | "percent" | "ratio" | "mae" | "rmse" | "MB" | "degrees" | "tokens/s";

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
  asset_filenames?: string[];
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
    repository_url?: string;
    ref: string;
    path: string;
    section: string;
    provenance: "existing-repository-documentation";
  };
}

/**
 * One published artifact of a sample. A manifest may declare `url` and `sha256`
 * as explicit `null` when the file is not downloadable or its digest is not
 * recorded, so both are nullable rather than merely optional.
 */
export interface ModelAsset {
  filename: string;
  format: string;
  url?: string | null;
  sha256?: string | null;
  role?: "model" | "dependency";
  display_name?: string;
}

export interface ModelRecord {
  id: string;
  name: string;
  tasks: string[];
  sample_path: string;
  availability: "download" | "manual";
  download_scripts: string[];
  assets: ModelAsset[];
  benchmarks: BenchmarkRecord[];
  variants?: ModelVariant[];
  /** Source records for this family on each release line. */
  platforms?: PlatformModelRecord[];
}

export interface ModelVariant {
  id: string;
  name: string;
  hardware: HardwareId;
  task: string;
  input?: BenchmarkRecord["input"];
  assets: ModelRecord["assets"];
  benchmarks: BenchmarkRecord[];
  sample_path: string;
  release_tag: string;
  /**
   * Git ref that holds `sample_path`. Platform distributions live under a
   * `platforms/<id>` prefix on the migration branch, while the frozen tags keep
   * the original repository-root layout; the pair below records which layout
   * this variant's source links must use.
   */
  source_ref?: string;
  /** Path prefix of the platform inside `source_ref`; empty for legacy layouts. */
  source_path_prefix?: string;
}

export type CatalogPlatform = "x5" | "s" | "x3";

/** Provenance of one platform distribution used to build a catalog version. */
export interface CatalogSourceRecord {
  kind: "worktree" | "tag";
  /** Git ref that holds the platform tree (`main`, or the pinned tag). */
  ref: string;
  /** Path of the platform inside `ref`. */
  path: string;
  manifest_directory: string;
  /** Release tag declared by the platform manifest. */
  release_tag: string;
  /** SHA256 of the exact manifest bytes read for this platform. */
  manifest_sha256: string;
}

export interface PlatformModelRecord extends Omit<ModelRecord, "platforms"> {
  platform: CatalogPlatform;
  release_tag: string;
}

export type CatalogSummary = Record<string, number>;

export interface Catalog {
  schema_version: number;
  release: Record<string, unknown> & {
    tag: string;
    platform: string;
    version: string;
    catalog_version?: string;
    repository?: string;
    platform_tags?: Record<string, string>;
  };
  /** Per-platform provenance; consumers use it to build layout-correct links. */
  sources?: Record<string, CatalogSourceRecord>;
  summary: CatalogSummary;
  models: ModelRecord[];
}
