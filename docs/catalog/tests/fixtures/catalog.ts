import type { BenchmarkRecord, MetricRecord, ModelRecord } from "../../src/catalog/types";

type BenchmarkOverrides = Omit<Partial<BenchmarkRecord>, "environment" | "input"> & {
  environment?: Partial<BenchmarkRecord["environment"]>;
  input?: Partial<NonNullable<BenchmarkRecord["input"]>>;
};

type ModelOverrides = Omit<Partial<ModelRecord>, "assets" | "benchmarks"> & {
  assets?: ModelRecord["assets"];
  benchmarks?: ModelRecord["benchmarks"];
};

const defaultMetricConditions: {
  input: NonNullable<BenchmarkRecord["input"]>;
  environment: BenchmarkRecord["environment"];
} = {
  input: { shape: [1, 3, 224, 224], layout: "NCHW", format: "NV12" },
  environment: { hardware: "RDK X5", runtime: "hobot_dnn", bpu_cores: 1 }
};

export function benchmarkFixture(overrides: BenchmarkOverrides = {}): BenchmarkRecord {
  const {
    environment: environmentOverrides,
    input: inputOverrides,
    ...recordOverrides
  } = overrides;

  return {
    id: "convnext-atto-x5",
    sample_id: "convnext",
    variant_id: "convnext-atto-224",
    display_name: "ConvNeXt Atto 224x224",
    asset_filename: "ConvNeXt_atto_224x224_nv12.bin",
    model_format: "bin",
    precision: "int8",
    input: { ...defaultMetricConditions.input, ...inputOverrides },
    environment: { ...defaultMetricConditions.environment, ...environmentOverrides },
    performance: [
      {
        metric: "latency",
        value: 2,
        unit: "ms",
        statistic: "mean",
        scope: "single-frame, single-thread, single-BPU-core",
        concurrency: 1
      },
      {
        metric: "throughput",
        value: 500,
        unit: "fps",
        scope: "four-thread concurrent",
        concurrency: 4
      }
    ],
    accuracy: [
      {
        metric: "top-1",
        value: 73,
        unit: "percent",
        dataset: "ImageNet-1K",
        model_stage: "quantized"
      }
    ],
    source: {
      ref: "x5-v1.0.0",
      path: "samples/vision/convnext/README.md",
      section: "## Performance Data",
      provenance: "existing-repository-documentation"
    },
    ...recordOverrides
  };
}

export function createModelFixture(overrides: ModelOverrides = {}): ModelRecord {
  const { assets, benchmarks, ...modelOverrides } = overrides;
  return {
    id: "convnext",
    name: "ConvNeXt",
    tasks: ["image-classification"],
    sample_path: "samples/vision/convnext",
    availability: "download",
    download_scripts: ["samples/vision/convnext/model/download.sh"],
    assets: assets ?? [
      {
        filename: "ConvNeXt_atto_224x224_nv12.bin",
        format: "bin",
        url: "https://archive.example.test/ConvNeXt_atto_224x224_nv12.bin",
        sha256: null
      }
    ],
    benchmarks: benchmarks ?? [benchmarkFixture()],
    ...modelOverrides
  };
}

export const modelFixture = createModelFixture();

function modelWithBenchmark(
  id: string,
  name: string,
  benchmark: BenchmarkRecord
): ModelRecord {
  return createModelFixture({ id, name, benchmarks: [benchmark] });
}

const alphaBenchmark = benchmarkFixture({
  id: "alpha-throughput",
  sample_id: "alpha",
  variant_id: "alpha-224",
  display_name: "Alpha detector",
  performance: [{
    metric: "throughput",
    value: 300,
    unit: "fps",
    scope: "four-thread concurrent",
    concurrency: 4
  }]
});

const betaBenchmark = benchmarkFixture({
  id: "beta-throughput",
  sample_id: "beta",
  variant_id: "beta-224",
  display_name: "Beta detector",
  environment: { hardware: "RDK X5" },
  performance: [{
    metric: "throughput",
    value: 100,
    unit: "fps",
    scope: "single-thread",
    concurrency: 1
  }]
});

export const incomparableModels: ModelRecord[] = [
  modelWithBenchmark("beta", "Beta", betaBenchmark),
  modelWithBenchmark("alpha", "Alpha", alphaBenchmark)
];

export const comparablePerformanceModels: ModelRecord[] = [
  modelWithBenchmark(
    "slow",
    "Slow",
    benchmarkFixture({
      id: "slow-throughput",
      sample_id: "slow",
      variant_id: "slow-224",
      performance: [{
        metric: "throughput",
        value: 100,
        unit: "fps",
        scope: "four-thread concurrent",
        concurrency: 4
      }]
    })
  ),
  modelWithBenchmark(
    "fast",
    "Fast",
    benchmarkFixture({
      id: "fast-throughput",
      sample_id: "fast",
      variant_id: "fast-224",
      performance: [{
        metric: "throughput",
        value: 300,
        unit: "fps",
        scope: "four-thread concurrent",
        concurrency: 4
      }]
    })
  )
];

export const comparableAccuracyModels: ModelRecord[] = [
  modelWithBenchmark(
    "lower-accuracy",
    "Lower accuracy",
    benchmarkFixture({
      id: "lower-accuracy-benchmark",
      sample_id: "lower-accuracy",
      variant_id: "lower-accuracy-224",
      performance: undefined,
      accuracy: [{
        metric: "top-1",
        value: 70,
        unit: "percent",
        dataset: "ImageNet-1K",
        model_stage: "quantized"
      }]
    })
  ),
  modelWithBenchmark(
    "higher-accuracy",
    "Higher accuracy",
    benchmarkFixture({
      id: "higher-accuracy-benchmark",
      sample_id: "higher-accuracy",
      variant_id: "higher-accuracy-224",
      performance: undefined,
      accuracy: [{
        metric: "top-1",
        value: 80,
        unit: "percent",
        dataset: "ImageNet-1K",
        model_stage: "quantized"
      }]
    })
  )
];

export const catalogFilterFixtures: ModelRecord[] = [
  createModelFixture(),
  createModelFixture({
    id: "detector",
    name: "Detector",
    tasks: ["object-detection"],
    assets: [{
      filename: "detector_640.onnx",
      format: "onnx",
      url: "https://archive.example.test/detector_640.onnx",
      sha256: null
    }],
    benchmarks: [benchmarkFixture({
      id: "detector-accuracy",
      sample_id: "detector",
      variant_id: "detector-640",
      model_format: "onnx",
      precision: "float32",
      performance: undefined,
      accuracy: [{
        metric: "map",
        value: 0.5,
        unit: "ratio",
        dataset: "COCO",
        model_stage: "float"
      }]
    })]
  }),
  createModelFixture({
    id: "manual-model",
    name: "Manual Model",
    availability: "manual",
    assets: [{ filename: "manual.bin", format: "bin", url: undefined, sha256: null }],
    benchmarks: []
  })
];

export const missingTargetMetricModel = createModelFixture({
  id: "latency-only",
  name: "Latency Only",
  benchmarks: [benchmarkFixture({
    id: "latency-only-benchmark",
    sample_id: "latency-only",
    variant_id: "latency-only-224",
    performance: [{
      metric: "latency",
      value: 5,
      unit: "ms",
      statistic: "mean",
      scope: "single-frame",
      concurrency: 1
    }],
    accuracy: undefined
  })]
});

export const incompleteAccuracyRecord = benchmarkFixture({
  accuracy: [{
    metric: "top-1",
    value: 70,
    unit: "percent",
    model_stage: "quantized"
  }]
});

export type { MetricRecord };
