import type { BenchmarkRecord, ModelRecord } from "../catalog/types";

/** Source-backed omissions in the historical S release; published tags stay immutable. */
export function applySCatalogErrata(
  tag: string,
  models: Array<Pick<ModelRecord, "id" | "tasks" | "assets">>,
  benchmarks: BenchmarkRecord[]
): void {
  if (tag !== "s-v1.1.2") return;
  // Archive migration: S100/S100P CLS files now use the 224 names already
  // specified by release download URLs. Legacy 640 URLs remain compatible.
  const canonicalCls = (name: string) => name.replace(
    /(yolov8|yolo11)([nsmlx])_cls_(nashe|nashm)_640x640_nv12\.hbm$/, "$1$2_cls_$3_224x224_nv12.hbm"
  );
  for (const model of models) for (const asset of model.assets) {
    asset.filename = canonicalCls(asset.filename);
    if (asset.url) asset.url = canonicalCls(asset.url);
  }
  for (const record of benchmarks) {
    if (record.asset_filename) record.asset_filename = canonicalCls(record.asset_filename);
    if (record.asset_filenames) record.asset_filenames = record.asset_filenames.map(canonicalCls);
  }

  // The Cls rows were appended to the evaluator README after its anchors were
  // first recorded, so the manifest still names `## Performance` / `## Accuracy`
  // for them — sections that never existed in that file. Repoint each record to
  // the table that actually carries its rows; the anchor names below are the
  // literal ATX headings in
  // samples/vision/ultralytics_yolo26/evaluator/README.md.
  const YOLO26_EVALUATOR_README = "samples/vision/ultralytics_yolo26/evaluator/README.md";
  const yolo26ClsSections: Array<[RegExp, string]> = [
    [/-cls-s100-performance$/, "### RDK S100 Performance Data (Performance @ NV12)"],
    [/-cls-s600-performance$/, "### RDK S600 Performance Data (Performance @ NV12)"],
    [/-cls-s100-accuracy$/, "### RDK S100 Accuracy Data (Accuracy @ NV12 - Classification)"],
    [/-cls-s600-accuracy$/, "### RDK S600 Accuracy Data (Accuracy @ NV12 - Classification)"]
  ];
  for (const record of benchmarks) {
    if (record.source.path !== YOLO26_EVALUATOR_README || !/^## (?:Performance|Accuracy)$/.test(record.source.section)) continue;
    const match = yolo26ClsSections.find(([pattern]) => pattern.test(record.id));
    if (match) record.source.section = match[1];
  }

  const depthAny = benchmarks.find(record => record.id === "depth-anything-v2-s100");
  if (depthAny && !benchmarks.some(record => record.id === "depth-anything-v2-s100-monitor")) benchmarks.push({
    ...depthAny, id: "depth-anything-v2-s100-monitor", accuracy: [],
    performance: [
      { metric: "bpu-occupancy", value: 95.4, unit: "percent", scope: "hrt_ucp_monitor reference; thread count not recorded" },
      { metric: "ion-memory", value: 300, unit: "MB", qualifier: "approximate", scope: "hrt_ucp_monitor reference; thread count not recorded" }
    ],
    source: { ...depthAny.source, section: "## Board Monitoring Metrics" }
  });
  for (const [size, hardware, value] of [["l", "s100", 11], ["l", "s100p", 8.1], ["x", "s100", 20.6], ["x", "s100p", 13.7], ["x", "s600", 10.8]] as const) {
    const base = benchmarks.find(record => record.id === `yolo26-depth-${size}-${hardware}`);
    const id = `yolo26-depth-${size}-${hardware}-readme-reference`;
    if (base && !benchmarks.some(record => record.id === id)) benchmarks.push({ ...base, id,
      display_name: `YOLO26${size} Depth lite README reference`, accuracy: [],
      performance: [{ metric: "latency", value, unit: "ms", scope: "README mixed-profile reference on bus.jpg; differs from evaluator latency table; timing boundary not recorded" }],
      source: { ...base.source, path: "samples/vision/yolo26_depth/README.md", section: "## Performance" }
    });
  }
  const calibrated: Record<string, number> = {
    "mobilenetv2-s100-ptq": 0.993383, "mobilenetv3-s100-ptq": 0.911233,
    "mobilenetv4-small-s100": 0.999892, "mobilenetv4-medium-s100": 0.999759,
    "resnet152-s100": 0.994397
  };
  for (const record of benchmarks) {
    if (record.sample_id === "yolo26_depth" && record.id.endsWith("-accuracy")) {
      for (const metric of record.accuracy ?? []) metric.dataset = "bundled bus.jpg; not full SUNRGBD evaluation";
    }
    if (calibrated[record.id] !== undefined && !record.accuracy?.some(metric => metric.metric === "calibrated-cosine-similarity")) {
      record.accuracy?.push({ metric: "calibrated-cosine-similarity", value: calibrated[record.id]!, unit: "ratio", scope: "Calibrated Cosine; toolchain report" });
    }
    if (["3dresnet-s100", "depth-anything-v2-s100"].includes(record.id)) {
      for (const metric of record.performance ?? []) if (metric.metric === "latency") metric.statistic = "mean";
    }
    if (record.id === "vit-s100-hbm" && !record.accuracy?.some(metric => metric.model_stage === "float")) {
      for (const metric of record.accuracy ?? []) if (metric.model_stage === "runtime") metric.model_stage = "quantized";
      record.accuracy?.push(
        { metric: "top-1", value: 74.54, unit: "percent", dataset: "CIFAR-10", model_stage: "float" },
        { metric: "top-5", value: 98.36, unit: "percent", dataset: "CIFAR-10", model_stage: "float" }
      );
    }
    if (record.sample_id === "siglip") {
      for (const metric of record.accuracy ?? []) {
        if (metric.scope === "pooler output zero-shot classification" && metric.model_stage === undefined) metric.model_stage = "quantized";
      }
    }
  }
  // The source reports these distributions once, not separately by board.
  const siglipDistributions: Record<string, number[]> = {
    "base-224": [0.951, 0.997, 0.980, 0.024, 0.471, 0.039],
    "base-384": [0.960, 0.997, 0.977, 0.029, 0.409, 0.050],
    "base-512": [0.956, 0.995, 0.974, 0.045, 0.507, 0.067],
    "large-256": [0.933, 0.997, 0.974, 0.018, 0.497, 0.024],
    "large-384": [0.900, 0.995, 0.965, 0.034, 0.775, 0.048],
    "so400m-224": [0.850, 0.995, 0.961, 0.028, 1.038, 0.041],
    "so400m-384": [0.859, 0.993, 0.957, 0.040, 1.093, 0.059],
    "so400m-256-i18n": [0.878, 0.996, 0.959, 0.018, 0.570, 0.030]
  };
  for (const [size, values] of Object.entries(siglipDistributions)) {
    const base = benchmarks.find(record => record.id === `siglip-${size}-s100`);
    if (!base || benchmarks.some(record => record.id === `siglip-${size}-distribution`)) continue;
    benchmarks.push({ id: `siglip-${size}-distribution`, sample_id: "siglip", variant_id: base.variant_id,
      display_name: `${base.display_name.replace(/ on RDK S100$/, "")} output distribution`,
      environment: { hardware: "RDK S100/S100P; shared source table" },
      accuracy: values.map((value, index) => ({ metric: index < 3 ? "cosine-similarity" : "mse", value, unit: "ratio", dataset: "COCO2014", scope: `last_hidden_state; 5000 images; ${["minimum", "maximum", "1% low"][index % 3]}` })),
      source: { ...base.source, section: "### last hidden state semantic consistency" }
    });
  }
  for (const hardware of ["s100", "s100p", "s600"] as const) {
    const base = benchmarks.find(record => record.id === `dinov2-${hardware}`);
    if (!base || benchmarks.some(record => record.id === `dinov2-${hardware}-board-cosine`)) continue;
    base.environment.cpu_mode = "performance governor locked";
    base.input = { shape: [1, 3, 224, 224], layout: "NCHW", format: "float32" };
    for (const metric of base.performance ?? []) metric.scope = `${metric.scope}; pure BPU forward; CPU preprocessing additional`;
    benchmarks.push({ ...base, id: `dinov2-${hardware}-board-cosine`,
      environment: { hardware: base.environment.hardware, runtime: "hbm_runtime versus ONNXRuntime float32 reference" }, performance: [],
      accuracy: [
        { metric: "cls-feature-cosine-similarity", value: hardware === "s600" ? 0.9988 : 0.9987, statistic: "min" as const },
        { metric: "cls-feature-cosine-similarity", value: 0.9989, statistic: "max" as const },
        { metric: "patch-feature-cosine-similarity", value: hardware === "s600" ? 0.9975 : 0.9977, statistic: "min" as const },
        { metric: "patch-feature-cosine-similarity", value: 0.9986, statistic: "max" as const }
      ].map(metric => ({ ...metric, unit: "ratio", model_stage: "quantized", qualifier: "exact", scope: `Board-executed versus float ONNX; observed range ${metric.statistic}` })),
      source: { ...base.source, section: "### Board-executed cosine vs float ONNX" }
    });
  }
  const dinoPtq = benchmarks.find(record => record.id === "dinov2-ptq-nash-e");
  if (dinoPtq && !dinoPtq.accuracy?.some(metric => metric.scope === "Calibrated Cosine")) {
    dinoPtq.accuracy?.push(
      { metric: "cls-feature-cosine-similarity", value: 0.9990, unit: "ratio", scope: "Calibrated Cosine" },
      { metric: "patch-feature-cosine-similarity", value: 0.9985, unit: "ratio", scope: "Calibrated Cosine" }
    );
  }
  // These SHAs are the gitlinks recorded by s-v1.1.2, not moving branches.
  if (models.some(model => model.id === "act") && !benchmarks.some(record => record.id === "act-s100-support")) {
    benchmarks.push({
      id: "act-s100-support", sample_id: "act", variant_id: "act-policy", display_name: "ACT Policy",
      environment: { hardware: "RDK S100" },
      source: { repository_url: "https://github.com/D-Robotics/rdk_LeRobot_tools", ref: "326ea043be204de25223d95c7d918efe8672dc66", path: "README.md", section: "Verified ACT deployment on RDK S100; benchmark not published", provenance: "existing-repository-documentation" }
    });
  }
  // S600 ACT support is documented alongside Pi0 in this immutable source commit.
  if (models.some(model => model.id === "act") && !benchmarks.some(record => record.id === "act-s600-support")) {
    benchmarks.push({
      id: "act-s600-support", sample_id: "act", variant_id: "act-policy", display_name: "ACT Policy",
      environment: { hardware: "RDK S600" },
      source: { repository_url: "https://github.com/D-Robotics/rdk_LeRobot_tools", ref: "a32de276bc1681a2b1531012de111eaa1c16acb6", path: "models/act/README.md", section: "ACT Policy on RDK S600", provenance: "existing-repository-documentation" }
    });
  }
  if (models.some(model => model.id === "pi0") && !benchmarks.some(record => record.id === "pi0-s600-fixed-input")) {
    benchmarks.push({
      id: "pi0-s600-fixed-input", sample_id: "pi0", variant_id: "pi0-policy", display_name: "Pi0 Policy",
      environment: { hardware: "RDK S600", runtime: "D-Robotics LLM S600 SDK 1.0.2; nash-p" },
      accuracy: [
        { metric: "mae", value: 0.4405, unit: "degrees" },
        { metric: "rmse", value: 0.5903, unit: "degrees" },
        { metric: "max-error", value: 1.6888, unit: "degrees" },
        { metric: "relative-l2", value: 0.852, unit: "percent" },
        { metric: "cosine-similarity", value: 0.999981858, unit: "ratio" }
      ].map(metric => ({ ...metric, qualifier: "exact", scope: "One fixed real input; BF16/HBM chain-integrity comparison, not task success rate" })) as BenchmarkRecord["accuracy"],
      source: { repository_url: "https://github.com/D-Robotics/rdk_LeRobot_tools", ref: "a32de276bc1681a2b1531012de111eaa1c16acb6", path: "models/pi0/README.md", section: "## Verified Baseline", provenance: "existing-repository-documentation" }
    });
  }
  if (models.some(model => model.id === "gemma4-e2b") && !benchmarks.some(record => record.id === "gemma4-e2b-s100p-demo")) {
    benchmarks.push({
      id: "gemma4-e2b-s100p-demo", sample_id: "gemma4-e2b", variant_id: "gemma4-e2b-language-model",
      asset_filename: "s100p/gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm", display_name: "Gemma4-E2B Language Model",
      environment: { hardware: "RDK S100P", runtime: "C++" },
      performance: [
        { metric: "token-generation-rate", value: 6.9, unit: "tokens/s", qualifier: "approximate", scope: "Published text-chat demo; Chinese prompt; streaming output" },
        { metric: "bpu-utilization", value: 86, unit: "percent", qualifier: "exact", scope: "Published VLM chat demo; image and Chinese prompt; pipeline utilization" }
      ],
      // Both demo captions (text chat, VLM chat) sit under this README heading.
      source: { ref: tag, path: "samples/llm/gemma4-e2b/README.md", section: "## Inference Result", provenance: "existing-repository-documentation" }
    });
  }
  // The pinned release root README categorizes both samples as robot policies.
  // Their gitlinks contain manipulation tooling, not legged locomotion demos.
  for (const model of models) {
    if (model.id === "paddle_ocr") {
      model.tasks = ["ocr-text-detection", "ocr-text-recognition"];
      for (const asset of [...model.assets]) {
        asset.display_name = asset.filename.includes("_rec_") ? "PP-OCRv6 Recognition" : "PP-OCRv6 Detection";
        if (asset.filename.startsWith("s100/")) {
          for (const board of ["s100p", "s600"]) {
            const filename = asset.filename.replace(/^s100\//, `${board}/`);
            if (!model.assets.some(candidate => candidate.filename === filename)) {
              model.assets.push({ ...asset, filename,
                url: board === "s600" ? asset.url?.replace("/rdk_s100/", "/rdk_s600/") : asset.url });
            }
          }
        }
      }
    }
    if (model.id === "diffusiondrive") {
      for (const asset of model.assets) asset.url = asset.url?.replace("/rdk_s100/diffusiondrive/", "/rdk_s100/");
      const records = benchmarks.filter(record => record.sample_id === model.id);
      for (const record of records) for (const metric of record.performance ?? []) {
        if (!metric.scope?.includes("200 frames")) metric.scope = `${metric.scope ?? ""}; 200 frames; fixed BPU core`;
      }
      const s100p = records.find(record => record.environment.hardware === "RDK S100P");
      if (s100p && !benchmarks.some(record => record.id === "diffusiondrive-s100p-five-case-means")) {
        benchmarks.push({ ...s100p, id: "diffusiondrive-s100p-five-case-means", performance: [],
          accuracy: [
            { metric: "trajectory-cosine-similarity", value: 0.999785 },
            { metric: "agent-state-cosine-similarity", value: 0.997986 },
            { metric: "bev-cosine-similarity", value: 0.998799 },
            { metric: "bev-pixel-agreement", value: 0.955664 },
            { metric: "bev-mean-iou", value: 0.819837 }
          ].map(metric => ({ ...metric, unit: "ratio", statistic: "mean", qualifier: "exact", dataset: "NAVSIM", scope: "All five packaged cases; S100P mean" }))
        });
      }
    }
    if (model.id === "bytetrack") {
      for (const asset of model.assets) asset.display_name = "ByteTrack (YOLOv5x detector)";
      const tracker = benchmarks.find(record => record.id === "bytetrack-s100");
      if (tracker) {
        tracker.input = { shape: [672, 672], format: "NV12" };
        for (const metric of tracker.performance ?? []) metric.qualifier = "approximate";
      }
    }
    if (model.id === "act" || model.id === "pi0") model.tasks = ["robot-manipulation-policy"];
    // README: two board-specific HBMs plus an external token embedding table.
    // The S100P archive deliberately lives under rdk_s100; URLs stay unchanged.
    if (model.id === "gemma4-e2b") {
      for (const asset of model.assets) {
        if (asset.filename.startsWith("common/")) asset.role = "dependency";
        else if (asset.filename.includes("_vit_")) asset.display_name = "Gemma4-E2B Vision Encoder";
        else if (asset.filename.includes("_lm_")) asset.display_name = "Gemma4-E2B Language Model";
      }
    }
  }
  const paraformer = benchmarks.find(record => record.id === "paraformer-s100");
  if (paraformer) {
    paraformer.asset_filenames = models.find(model => model.id === "paraformer")?.assets
      .filter(asset => asset.filename.startsWith("s100/") && /(?:encoder|predictor|decoder).*\.hbm$/.test(asset.filename))
      .map(asset => asset.filename);
    paraformer.input = { shape: [1, 400, 560], format: "float32 fbank+LFR features from WAV" };
    paraformer.display_name = "Paraformer-Large ASR pipeline";
    for (const metric of paraformer.performance ?? []) {
      if (!metric.scope?.includes("historical 300-utterance")) metric.scope = `${metric.scope}; historical 300-utterance HBM pipeline; WAV frontend excluded`;
    }
  }
  if (paraformer && !benchmarks.some(record => record.id === "paraformer-cpp-stages-s100")) {
    const source: BenchmarkRecord["source"] = {
      ref: "s-v1.1.2", path: "samples/speech/paraformer/evaluator/README.md",
      section: "## Completed Validation", provenance: "existing-repository-documentation"
    };
    benchmarks.push({
      ...paraformer, id: "paraformer-cpp-stages-s100", display_name: "Paraformer-Large C++ UCP",
      performance: [
        { metric: "encoder-latency", value: 33.15, unit: "ms", qualifier: "exact", scope: "C++ UCP" },
        { metric: "predictor-latency", value: 1, unit: "ms", qualifier: "exact", scope: "C++ UCP" },
        { metric: "cpu-cif-latency", value: 0.38, unit: "ms", qualifier: "exact", scope: "C++ UCP" },
        { metric: "decoder-latency", value: 6.29, unit: "ms", qualifier: "exact", scope: "C++ UCP" }
      ], accuracy: [], source
    }, {
      ...paraformer, id: "paraformer-rtf-s100", display_name: "Paraformer-Large historical HBM pipeline RTF",
      performance: [
        { metric: "rtf", value: 0.008, unit: "ratio", qualifier: "exact", scope: "Python hbm_runtime; historical 300-utterance HBM pipeline" },
        { metric: "rtf", value: 0.007, unit: "ratio", qualifier: "exact", scope: "C++ UCP; historical 300-utterance HBM pipeline" }
      ], accuracy: [], source
    });
  }
  // Read directly from the final modelOutput row of the evaluator's image.
  // Calibrated cosine is not FP32 accuracy, and this compiler report does not
  // identify a board-specific test. Keep both values outside board variants.
  if (models.some(model => model.id === "asr") && !benchmarks.some(record => record.id === "asr-toolchain-cosine-image")) {
    benchmarks.push({
      id: "asr-toolchain-cosine-image",
      sample_id: "asr",
      variant_id: "asr-model-output-toolchain",
      display_name: "ASR Wav2Vec2 modelOutput cosine similarity",
      environment: { hardware: "Toolchain report; board not specified" },
      accuracy: [
        { metric: "calibrated-cosine-similarity", value: 0.999105, unit: "ratio", qualifier: "exact", scope: "modelOutput; Calibrated Cosine" },
        { metric: "quantized-cosine-similarity", value: 0.999181, unit: "ratio", qualifier: "exact", model_stage: "quantized", scope: "modelOutput; Quantized Cosine" }
      ],
      source: {
        ref: "s-v1.1.2",
        path: "samples/speech/asr/test_data/readme_img/acc.jpg",
        section: "modelOutput: Calibrated Cosine / Quantized Cosine",
        provenance: "existing-repository-documentation"
      }
    });
  }
}
