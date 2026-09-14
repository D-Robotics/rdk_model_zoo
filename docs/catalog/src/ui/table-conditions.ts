import type { BenchmarkRecord, Locale, MetricRecord } from "../catalog/types";
import { detailLabel } from "./detail-labels";
import type { DetailContext } from "./detail-types";

/**
 * Comparison tables repeat the same tool, timing and thread wording in every
 * cell. Those repeated facts describe the column (or the whole table), not a
 * single value, so they belong in one source-backed conditions block below the
 * table instead of beside every number.
 *
 * A fact is only promoted out of the cells when every populated cell of the
 * same column agrees on it. Anything that differs between observations — a
 * named stage, a runtime, a mean/p50/p95 statistic, a timing scope — stays
 * beside its own value, because without it two numbers in one cell would be
 * indistinguishable. Nothing is inferred from the model name or assumed
 * globally: a condition is reported only when a source record states it.
 */

export type ConditionKind = "statistic" | "runtime" | "scope";

/**
 * The minimum one rendered observation has to expose for condition
 * derivation. A full `MetricEntry` satisfies it; so does a bare record and
 * metric pair, which keeps the derivation testable on its own.
 */
export interface ConditionSubject {
  record: Pick<BenchmarkRecord, "environment" | "source">;
  metric: Pick<MetricRecord, "metric" | "value" | "statistic" | "scope">;
}

export interface ConditionPart {
  kind: ConditionKind;
  value: string;
}

/** Conditions that every populated cell of one column agrees on. */
export type UniformConditions = Partial<Record<ConditionKind, string>>;

export interface ConditionFact {
  label: string;
  value: string;
}

export interface ConditionsSummary {
  /** Table-wide environment facts, each stated by the source records. */
  environment: ConditionFact[];
  sourceNote?: string;
  /** Timing wording promoted out of the cells, keyed by the column it explains. */
  timing: ConditionFact[];
  /** Thread counts measured for this table. */
  threads?: ConditionFact;
  /** Source documents the promoted conditions were read from. */
  sources: Array<{ path: string; ref: string; section: string; repositoryUrl?: string }>;
}

/** The per-value conditions that distinguish one observation from another. */
export function conditionParts(entry: ConditionSubject): ConditionPart[] {
  const parts: ConditionPart[] = [];
  if (entry.metric.statistic) parts.push({ kind: "statistic", value: entry.metric.statistic });
  if (entry.record.environment.runtime) parts.push({ kind: "runtime", value: entry.record.environment.runtime });
  if (entry.metric.scope) parts.push({ kind: "scope", value: entry.metric.scope });
  return parts;
}

export function conditionText(parts: ConditionPart[]): string {
  return parts.map((part) => part.value).join(" · ");
}

/** The parts a cell still has to print because the column does not share them. */
export function visibleParts(entry: ConditionSubject, uniform: UniformConditions | undefined): ConditionPart[] {
  return conditionParts(entry).filter((part) => uniform?.[part.kind] !== part.value);
}

function uniqueStrings(values: Array<string | undefined>): string[] {
  return [...new Set(values.filter((value): value is string => Boolean(value && value.trim())))];
}

function sharedValue(values: Array<string | undefined>): string | undefined {
  if (values.some(value => !value?.trim())) return undefined;
  const unique = uniqueStrings(values);
  return unique.length === 1 ? unique[0] : undefined;
}

/**
 * Columns are identified by the cell they feed, so uniformity is judged
 * against comparable cells only: the same metric, thread count and stage.
 *
 * A wording is only promoted out of the cells when it actually repeats, so at
 * least two rows must carry it. A single-row table repeats nothing, and the
 * scope that qualifies its only value stays beside that value where a reader
 * cannot miss it.
 */
export function uniformColumnConditions(
  columns: Map<string, ConditionSubject[][]>
): Map<string, UniformConditions> {
  const result = new Map<string, UniformConditions>();
  for (const [key, cells] of columns) {
    const populated = cells.filter((entries) => entries.length > 0);
    if (populated.length < 2) continue;
    const uniform: UniformConditions = {};
    for (const kind of ["statistic", "runtime", "scope"] as const) {
      // Every populated cell must state the condition. If one observation
      // leaves it out, the column does not agree on it, and promoting the
      // other observations' wording would silently describe a value the
      // source never qualified.
      const values = populated.map((entries) => entries
        .map((entry) => conditionParts(entry).find((part) => part.kind === kind)?.value));
      if (values.some((cell) => cell.length !== 1)) continue;
      const value = sharedValue(values.flat());
      if (value !== undefined) uniform[kind] = value;
    }
    result.set(key, uniform);
  }
  return result;
}

function environmentFacts(records: BenchmarkRecord[], locale: Locale): ConditionFact[] {
  // Only conditions the table has no column of its own for. The input size and
  // format already own the Input column, so repeating them here would be noise.
  const facts: ConditionFact[] = [];
  const runtime = sharedValue(records.map((record) => record.environment.runtime));
  if (runtime) facts.push({ label: detailLabel(locale, "runtime"), value: runtime });
  const cpuMode = sharedValue(records.map((record) => record.environment.cpu_mode));
  if (cpuMode) facts.push({ label: detailLabel(locale, "cpuMode"), value: cpuMode });
  const bpuCores = sharedValue(records.map((record) => record.environment.bpu_cores === undefined
    ? undefined : String(record.environment.bpu_cores)));
  if (bpuCores) facts.push({ label: detailLabel(locale, "bpuCores"), value: bpuCores });
  return facts;
}

export function sharedAccuracyScope(records: BenchmarkRecord[]): string | undefined {
  const metrics = records.flatMap(record => record.accuracy ?? []);
  return metrics.length >= 2 ? sharedValue(metrics.map(metric => metric.scope)) : undefined;
}

function threadFact(
  counts: Array<number | undefined>,
  locale: Locale
): ConditionFact | undefined {
  if (counts.length === 0) return undefined;
  const known = counts.filter((count): count is number => count !== undefined);
  const zh = locale === "zh";
  const value = known.length === 0
    ? detailLabel(locale, "unknownConcurrency")
    : [...known.map(String), ...(known.length === counts.length ? [] : [detailLabel(locale, "notRecorded")])].join(" / ");
  return { label: detailLabel(locale, "concurrency"), value };
}

export interface TableConditionsInput {
  records: BenchmarkRecord[];
  /** Records of the same sample on every platform, used only for tool wording. */
  siblingRecords: BenchmarkRecord[];
  columns: Map<string, ConditionSubject[][]>;
  columnLabels: Map<string, string>;
  threadCounts: Array<number | undefined>;
  /** Release tag of the rendered configurations, used to resolve source links. */
  sourceRef?: string;
  locale: Locale;
}

/** Everything the conditions block needs, derived only from source records. */
export function summarizeConditions(input: TableConditionsInput): ConditionsSummary {
  const { records, siblingRecords, columns, columnLabels, threadCounts, sourceRef, locale } = input;
  const environment = environmentFacts(records, locale);

  const tableRuntime = sharedValue(records.map(record => record.environment.runtime));

  const timing: ConditionFact[] = [];
  const accuracyScope = sharedAccuracyScope(records);
  if (accuracyScope) timing.push({ label: locale === "zh" ? "精度测试范围" : "Accuracy scope", value: accuracyScope });
  for (const [key, conditions] of uniformColumnConditions(columns)) {
    const text = conditionText((["statistic", "runtime", "scope"] as const)
      .filter((kind) => conditions[kind] !== undefined)
      .filter((kind) => !(kind === "runtime" && conditions[kind] === tableRuntime))
      .map((kind) => ({ kind, value: conditions[kind]! })));
    if (text) timing.push({ label: columnLabels.get(key) ?? key, value: text });
  }

  // Evidence can live in another repository, so the link keeps the record's
  // own repository instead of assuming the catalog's.
  const sources = [...new Map(records.map((record) => [
    `${record.source.repository_url ?? ""}|${record.source.ref}|${record.source.path}`,
    {
      path: record.source.path,
      ref: record.source.ref,
      section: record.source.section,
      repositoryUrl: record.source.repository_url
    }
  ])).values()];

  // These two published evaluator READMEs were reviewed directly. Keep their
  // explanation without inferring conditions for other samples or releases.
  const reviewedYolo = ["x5-v1.1.2", "s-v1.1.2"].includes(sourceRef ?? "")
    && sources.some(source => source.path === "samples/vision/ultralytics_yolo/evaluator/README.md" && !source.repositoryUrl);
  const sourceNote = reviewedYolo ? (locale === "zh"
    ? "README 测试说明：使用 hrt_model_exec 在开发板上测试 BPU 延迟与吞吐量，单线程与多线程结果分别比较；CPU 后处理按单核单独列出。测试命令和设备状态要求见下方来源文档。"
    : "README test instructions: hrt_model_exec measures BPU latency and throughput on the board. Compare single-thread and multi-thread results separately; single-core CPU post-processing is listed separately. See the source for commands and device-state requirements.") : undefined;
  return { environment, timing, threads: threadFact(threadCounts, locale), sources, sourceNote };
}

export function hasConditions(summary: ConditionsSummary): boolean {
  return summary.environment.length > 0 || summary.timing.length > 0;
}

function factLine(fact: ConditionFact): HTMLParagraphElement {
  const item = document.createElement("p");
  item.className = "model-detail-condition";
  const key = document.createElement("strong");
  key.textContent = `${fact.label}: `;
  item.append(key, document.createTextNode(fact.value));
  return item;
}

/**
 * The shared conditions block below a comparison table. It states only what
 * the rendered records carry, so a platform that never recorded a governor or
 * an input format does not inherit another platform's wording.
 */
export function renderTableConditions(
  summary: ConditionsSummary,
  context: DetailContext
): HTMLElement {
  const section = document.createElement("section");
  section.className = "model-detail-test-conditions";
  const heading = document.createElement("h3");
  heading.textContent = detailLabel(context.locale, "conditions");
  section.append(heading);

  for (const fact of [...summary.environment, ...summary.timing]) section.append(factLine(fact));
  if (summary.threads) section.append(factLine(summary.threads));
  if (summary.sourceNote) {
    const note = document.createElement("p");
    note.className = "model-detail-condition";
    note.textContent = summary.sourceNote;
    section.append(note);
  }

  if (summary.sources.length > 0) {
    const source = document.createElement("p");
    source.className = "model-detail-condition-source";
    for (const [index, entry] of summary.sources.entries()) {
      if (index > 0) source.append(document.createTextNode(" · "));
      const link = document.createElement("a");
      const repository = (entry.repositoryUrl ?? context.repositoryUrl).replace(/\/$/, "");
      link.href = `${repository}/blob/${encodeURIComponent(entry.ref)}/${entry.path}`;
      link.textContent = entry.section || entry.path;
      link.title = `${entry.ref}: ${entry.path}`;
      source.append(link);
    }
    section.append(source);
  }
  return section;
}
