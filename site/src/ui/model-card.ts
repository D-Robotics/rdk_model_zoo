import type { Locale, MetricRecord, ModelRecord } from "../catalog/types";
import { t, taskTranslationKey, type TranslationKey } from "../i18n/translations";

export interface RenderedModelCard {
  element: HTMLElement;
  destroy(): void;
}

const qualifierKeys: Record<NonNullable<MetricRecord["qualifier"]>, TranslationKey> = {
  exact: "qualifier.exact",
  "lower-bound": "qualifier.lowerBound",
  "upper-bound": "qualifier.upperBound",
  approximate: "qualifier.approximate"
};

const metricKeys: Record<string, TranslationKey> = {
  latency: "metric.latency",
  throughput: "metric.throughput",
  fps: "metric.throughput",
  "top-1": "metric.top1",
  "top-5": "metric.top5",
  map: "metric.map",
  miou: "metric.miou",
  mae: "metric.mae",
  rmse: "metric.rmse",
  cosine_similarity: "metric.cosineSimilarity"
};

const unitKeys: Record<MetricRecord["unit"], TranslationKey> = {
  ms: "unit.ms",
  us: "unit.us",
  fps: "unit.fps",
  percent: "unit.percent",
  ratio: "unit.ratio",
  mae: "unit.mae",
  rmse: "unit.rmse"
};

function unique(values: Array<string | undefined>): string[] {
  return [...new Set(values.filter((value): value is string => Boolean(value)))];
}

function labeledValue(label: string, value: string): HTMLDivElement {
  const row = document.createElement("div");
  row.className = "model-fact";
  const term = document.createElement("dt");
  term.textContent = label;
  const description = document.createElement("dd");
  description.textContent = value;
  row.append(term, description);
  return row;
}

function formatMetric(metric: MetricRecord, locale: Locale): HTMLElement {
  const item = document.createElement("li");
  item.className = "metric-item";
  const labelKey = metricKeys[metric.metric.toLocaleLowerCase()];
  const label = labelKey ? t(locale, labelKey) : metric.metric;
  const number = new Intl.NumberFormat(locale === "zh" ? "zh-CN" : "en-US", {
    maximumFractionDigits: 6
  }).format(metric.value);
  item.append(`${label}: ${number} ${t(locale, unitKeys[metric.unit])}`);
  if (metric.qualifier && metric.qualifier !== "exact") {
    const qualifier = document.createElement("span");
    qualifier.className = "metric-qualifier";
    qualifier.textContent = ` (${t(locale, qualifierKeys[metric.qualifier])})`;
    item.append(qualifier);
  }
  return item;
}

function metricGroup(
  headingText: string,
  missingText: string,
  metrics: Array<{ metric: MetricRecord; hardware: string }>,
  locale: Locale,
  limit: number
): HTMLElement {
  const group = document.createElement("section");
  group.className = "model-metric-group";
  const heading = document.createElement("h4");
  heading.textContent = headingText;
  group.append(heading);
  if (metrics.length === 0) {
    const missing = document.createElement("p");
    missing.className = "missing-data";
    missing.textContent = missingText;
    group.append(missing);
    return group;
  }

  const list = document.createElement("ul");
  for (const entry of metrics.slice(0, limit)) {
    const item = formatMetric(entry.metric, locale);
    const conditions = unique([entry.hardware, entry.metric.scope]);
    if (conditions.length > 0) {
      const detail = document.createElement("small");
      detail.className = "metric-conditions";
      detail.textContent = ` — ${conditions.join(" · ")}`;
      item.append(detail);
    }
    list.append(item);
  }
  group.append(list);
  return group;
}

export function renderModelCard(
  model: ModelRecord,
  platform: string,
  locale: Locale,
  onSelect: (modelId: string) => void
): RenderedModelCard {
  const article = document.createElement("article");
  article.className = "model-card";
  article.dataset.modelId = model.id;
  const title = document.createElement("h3");
  title.textContent = model.name;

  const tasks = document.createElement("ul");
  tasks.className = "task-badges";
  tasks.setAttribute("aria-label", t(locale, "model.tasks"));
  for (const task of model.tasks) {
    const item = document.createElement("li");
    item.textContent = t(locale, taskTranslationKey(task));
    tasks.append(item);
  }

  const formats = unique([
    ...model.assets.map((asset) => asset.format),
    ...model.benchmarks.map((benchmark) => benchmark.model_format)
  ]);
  const precisions = unique(model.benchmarks.map((benchmark) => benchmark.precision));
  const variants = unique(model.benchmarks.map((benchmark) => benchmark.variant_id));
  const checksummed = model.assets.filter((asset) => asset.sha256).length;
  const facts = document.createElement("dl");
  facts.className = "model-facts";
  facts.append(
    labeledValue(t(locale, "model.platform"), platform.toUpperCase()),
    labeledValue(t(locale, "model.formats"), formats.join(", ") || t(locale, "missing.notPublished")),
    labeledValue(t(locale, "model.precisions"), precisions.join(", ") || t(locale, "missing.notPublished")),
    labeledValue(t(locale, "model.variants"), t(locale, "model.variantCount", { count: variants.length })),
    labeledValue(
      t(locale, "model.availability"),
      t(locale, model.availability === "download" ? "model.downloadable" : "model.manual")
    ),
    labeledValue(t(locale, "model.checksumCoverage"), `${checksummed}/${model.assets.length}`)
  );

  const performance = model.benchmarks.flatMap((benchmark) =>
    (benchmark.performance ?? []).map((metric) => ({ metric, hardware: benchmark.environment.hardware }))
  );
  const accuracy = model.benchmarks.flatMap((benchmark) =>
    (benchmark.accuracy ?? []).map((metric) => ({ metric, hardware: benchmark.environment.hardware }))
  );
  const metrics = document.createElement("div");
  metrics.className = "model-metrics";
  metrics.append(
    metricGroup(t(locale, "details.performance"), t(locale, "model.performanceMissing"), performance, locale, 2),
    metricGroup(t(locale, "details.accuracy"), t(locale, "model.accuracyMissing"), accuracy, locale, 1)
  );

  const actions = document.createElement("div");
  actions.className = "model-actions";
  const details = document.createElement("button");
  details.type = "button";
  details.dataset.action = "open-details";
  details.textContent = t(locale, "details.overview");
  details.setAttribute("aria-label", t(locale, "model.openDetails", { name: model.name }));
  const select = (): void => onSelect(model.id);
  details.addEventListener("click", select);
  actions.append(details);

  article.append(title, tasks, facts, metrics, actions);
  return {
    element: article,
    destroy() {
      details.removeEventListener("click", select);
    }
  };
}
