import type { Locale } from "./types";

// Task IDs are data identities; labels never determine membership.
const GROUPS = [
  { id: "vision", zh: "视觉", en: "Vision", tasks: ["image-classification", "object-detection", "instance-segmentation", "semantic-segmentation", "portrait-matting", "pose-estimation", "monocular-depth-estimation", "oriented-bounding-box-detection", "oriented-object-detection", "promptable-image-segmentation", "image-embedding", "vision-embedding", "video-action-classification", "multi-object-tracking"] },
  { id: "language", zh: "语言与多模态", en: "Language & multimodal", tasks: ["image-text-similarity", "vision-language-model", "open-vocabulary-object-detection", "text-generation"] },
  { id: "text", zh: "文字识别", en: "Text recognition", tasks: ["license-plate-recognition", "ocr-text-detection", "ocr-text-recognition", "text-detection", "text-recognition"] },
  { id: "audio", zh: "语音", en: "Speech", tasks: ["keyword-spotting", "speech-recognition"] },
  { id: "robotics", zh: "机器人与空间感知", en: "Robotics & spatial", tasks: ["legged-locomotion-control", "autonomous-driving", "lane-detection", "point-cloud-segmentation"] }
];
export function groupTasks(tasks: string[], locale: Locale): Array<{ id: string; label: string; tasks: string[] }> {
  const remaining = new Set(tasks);
  const groups = GROUPS.map(group => ({ id: group.id, label: locale === "zh" ? group.zh : group.en,
    tasks: group.tasks.filter(task => remaining.delete(task)) })).filter(group => group.tasks.length);
  if (remaining.size) groups.push({ id: "other", label: locale === "zh" ? "其他任务" : "Other tasks", tasks: [...remaining].sort() });
  return groups;
}
