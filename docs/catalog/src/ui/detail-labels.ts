import type { HardwareId, Locale, MetricRecord } from "../catalog/types";

export type DetailLabelKey =
  | "back"
  | "close"
  | "hardware"
  | "task"
  | "tasks"
  | "specifications"
  | "specification"
  | "input"
  | "performance"
  | "accuracy"
  | "retention"
  | "floatAccuracy"
  | "quantizedAccuracy"
  | "download"
  | "downloads"
  | "source"
  | "conditions"
  | "details"
  | "assets"
  | "modelFormat"
  | "precision"
  | "dataset"
  | "metric"
  | "value"
  | "unit"
  | "scope"
  | "statistic"
  | "concurrency"
  | "unknownConcurrency"
  | "runtime"
  | "cpuMode"
  | "bpuCores"
  | "modelStage"
  | "notMeasured"
  | "notComparable"
  | "notApplicable"
  | "notRecorded"
  | "downloadNotRecorded"
  | "manualDownload"
  | "available"
  | "checksum"
  | "noAssets"
  | "noPerformance"
  | "noAccuracy";

const labels: Record<Locale, Record<DetailLabelKey, string>> = {
  en: {
    back: "Back to catalog",
    close: "Close details",
    hardware: "Hardware",
    task: "Task",
    tasks: "Tasks",
    specifications: "Model specifications",
    specification: "Specification",
    input: "Input size",
    performance: "Platform benchmarks",
    accuracy: "Accuracy",
    retention: "Retention",
    floatAccuracy: "FP32 accuracy",
    quantizedAccuracy: "Quantized/BPU accuracy",
    download: "Download",
    downloads: "Downloads",
    source: "Source evidence",
    conditions: "Test conditions",
    details: "Details",
    assets: "Assets",
    modelFormat: "Model format",
    precision: "Precision",
    dataset: "Dataset",
    metric: "Metric",
    value: "Value",
    unit: "Unit",
    scope: "Timing scope",
    statistic: "Statistic",
    concurrency: "Concurrency",
    unknownConcurrency: "Concurrency not recorded",
    runtime: "Runtime",
    cpuMode: "CPU mode",
    bpuCores: "BPU cores",
    modelStage: "Model stage",
    notMeasured: "Not yet measured",
    notComparable: "Not comparable",
    notApplicable: "Not applicable",
    notRecorded: "Not recorded",
    downloadNotRecorded: "Download address not recorded",
    manualDownload: "Manual model required",
    available: "Available",
    checksum: "SHA-256",
    noAssets: "Model files not published for this variant",
    noPerformance: "Performance not yet measured",
    noAccuracy: "Accuracy not yet measured"
  },
  zh: {
    back: "返回模型目录",
    close: "关闭详情",
    hardware: "硬件",
    task: "任务",
    tasks: "任务",
    specifications: "模型规格",
    specification: "规格",
    input: "输入尺寸",
    performance: "平台 Benchmark",
    accuracy: "精度",
    retention: "保持率",
    floatAccuracy: "FP32 精度",
    quantizedAccuracy: "量化/BPU 精度",
    download: "下载",
    downloads: "下载文件",
    source: "来源证据",
    conditions: "测试条件",
    details: "详情",
    assets: "模型资产",
    modelFormat: "模型格式",
    precision: "精度",
    dataset: "数据集",
    metric: "指标",
    value: "数值",
    unit: "单位",
    scope: "计时范围",
    statistic: "统计口径",
    concurrency: "并发度",
    unknownConcurrency: "线程数未记录",
    runtime: "运行时",
    cpuMode: "CPU 模式",
    bpuCores: "BPU 核心数",
    modelStage: "模型阶段",
    notMeasured: "尚未实测",
    notComparable: "不可比较",
    notApplicable: "不适用",
    notRecorded: "未记录",
    downloadNotRecorded: "下载地址未记录",
    manualDownload: "需要手动提供模型",
    available: "可用",
    checksum: "SHA-256",
    noAssets: "该规格未发布模型文件",
    noPerformance: "性能尚未实测",
    noAccuracy: "精度尚未实测"
  }
};

const hardwareNames: Record<Locale, Record<HardwareId, string>> = {
  en: { x3: "X3", x5: "X5", s100: "S100", s100p: "S100P", s600: "S600" },
  zh: { x3: "X3", x5: "X5", s100: "S100", s100p: "S100P", s600: "S600" }
};

const taskNames: Record<Locale, Record<string, string>> = {
  en: {
    "image-classification": "Image classification",
    "image-text-similarity": "Image-text similarity",
    "instance-segmentation": "Instance segmentation",
    "legged-locomotion-control": "Legged locomotion control",
    "license-plate-recognition": "License plate recognition",
    "monocular-depth-estimation": "Monocular depth estimation",
    "object-detection": "Object detection",
    "ocr-text-detection": "OCR text detection",
    "ocr-text-recognition": "OCR text recognition",
    "open-vocabulary-object-detection": "Open-vocabulary object detection",
    "oriented-bounding-box-detection": "Oriented bounding box detection",
    "portrait-matting": "Portrait matting",
    "pose-estimation": "Pose estimation",
    "promptable-image-segmentation": "Promptable image segmentation",
    "semantic-segmentation": "Semantic segmentation"
  },
  zh: {
    "image-classification": "图像分类",
    "image-text-similarity": "图文相似度",
    "instance-segmentation": "实例分割",
    "legged-locomotion-control": "足式运动控制",
    "license-plate-recognition": "车牌识别",
    "monocular-depth-estimation": "单目深度估计",
    "object-detection": "目标检测",
    "ocr-text-detection": "OCR 文本检测",
    "ocr-text-recognition": "OCR 文本识别",
    "open-vocabulary-object-detection": "开放词汇目标检测",
    "oriented-bounding-box-detection": "旋转目标检测",
    "portrait-matting": "人像抠图",
    "pose-estimation": "姿态估计",
    "promptable-image-segmentation": "提示式图像分割",
    "semantic-segmentation": "语义分割"
  }
};

const metricNames: Record<Locale, Record<string, string>> = {
  en: {
    latency: "Latency",
    throughput: "Throughput",
    fps: "Throughput",
    post_process_latency: "Post-process latency",
    "top-1": "Top-1 accuracy",
    "top-5": "Top-5 accuracy",
    map: "mAP",
    miou: "mIoU",
    mae: "MAE",
    rmse: "RMSE",
    cosine_similarity: "Cosine similarity",
    retention: "Retention"
  },
  zh: {
    latency: "延迟",
    throughput: "吞吐",
    fps: "吞吐",
    post_process_latency: "后处理延迟",
    "top-1": "Top-1 精度",
    "top-5": "Top-5 精度",
    map: "mAP",
    miou: "mIoU",
    mae: "MAE",
    rmse: "RMSE",
    cosine_similarity: "余弦相似度",
    retention: "保持率"
  }
};

export function detailLabel(locale: Locale, key: DetailLabelKey): string {
  return labels[locale][key];
}

export function hardwareLabel(locale: Locale, hardware: HardwareId): string {
  return hardwareNames[locale][hardware];
}

export function taskLabel(locale: Locale, task: string): string {
  return taskNames[locale][task] ?? task;
}

export function metricLabel(locale: Locale, metric: string): string {
  const key = metric.normalize("NFKC").trim().toLocaleLowerCase();
  return metricNames[locale][key] ?? metric;
}

export function unitLabel(_locale: Locale, unit: MetricRecord["unit"]): string {
  if (unit === "percent") return "%";
  if (unit === "ratio") return "ratio";
  if (unit === "fps") return "FPS";
  return unit;
}
