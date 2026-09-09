import type { BenchmarkRecord } from "../catalog/types";
import { detailLabel, unitLabel } from "./detail-labels";
import type { DetailContext } from "./detail-types";
import { accuracyValueText } from "./accuracy-comparison";
import { cell, metricCellText, sourceUrl, table, wrapper } from "./detail-utils";

function sourceLink(record: BenchmarkRecord, context: DetailContext): HTMLAnchorElement {
  const link = document.createElement("a");
  link.dataset.testid = "benchmark-source";
  link.href = sourceUrl(record, context.repositoryUrl);
  link.textContent = record.source.section;
  link.title = `${record.source.ref}: ${record.source.path}`;
  return link;
}

function inputDescription(input: BenchmarkRecord["input"] | undefined): string {
  const parts = [
    input?.shape?.join("×"),
    input?.layout,
    input?.format
  ].filter((value): value is string => Boolean(value));
  return parts.join(" · ");
}

function renderConditions(records: BenchmarkRecord[], context: DetailContext): HTMLElement {
  const conditions = document.createElement("section");
  conditions.className = "model-detail-conditions";
  const heading = document.createElement("h4");
  heading.textContent = detailLabel(context.locale, "conditions");
  conditions.append(heading);
  for (const [index, record] of records.entries()) {
    const block = document.createElement("p");
    block.className = "model-detail-condition";
    const environment = [
      record.environment.hardware,
      record.environment.runtime ? `${detailLabel(context.locale, "runtime")}: ${record.environment.runtime}` : undefined,
      record.environment.cpu_mode ? `${detailLabel(context.locale, "cpuMode")}: ${record.environment.cpu_mode}` : undefined,
      record.environment.bpu_cores !== undefined ? `${detailLabel(context.locale, "bpuCores")}: ${record.environment.bpu_cores}` : undefined,
      inputDescription(record.input) || undefined
    ].filter((value): value is string => Boolean(value));
    block.textContent = environment.join(" · ");
    if (index === 0) block.append(" · ");
    block.append(sourceLink(record, context));
    conditions.append(block);
  }
  return conditions;
}

function renderMetricTable(records: BenchmarkRecord[], context: DetailContext): HTMLElement | undefined {
  const metrics = table(detailLabel(context.locale, "details"));
  metrics.className = "model-detail-metrics-table";
  const head = metrics.tHead!.insertRow();
  for (const header of [
    detailLabel(context.locale, "metric"),
    detailLabel(context.locale, "value"),
    detailLabel(context.locale, "unit"),
    detailLabel(context.locale, "scope"),
    detailLabel(context.locale, "concurrency"),
    detailLabel(context.locale, "dataset"),
    detailLabel(context.locale, "modelStage")
  ]) head.append(cell(header, true));

  const body = metrics.tBodies[0]!;
  for (const record of records) {
    for (const kind of ["performance", "accuracy"] as const) {
      for (const metric of record[kind] ?? []) {
        const row = body.insertRow();
        row.dataset.metric = metric.metric;
        row.append(
          cell(metric.metric),
          // Raw evidence shows the value exactly as published; the unit lives in
          // its own column, so a ratio is never re-scaled into a percentage.
          cell(kind === "accuracy" ? accuracyValueText(metric, context.locale) : metricCellText(metric, context.locale)),
          cell(unitLabel(context.locale, metric.unit)),
          cell(metric.scope ?? detailLabel(context.locale, "notRecorded")),
          cell(metric.concurrency === undefined ? detailLabel(context.locale, "unknownConcurrency") : String(metric.concurrency)),
          cell(metric.dataset ?? detailLabel(context.locale, "notRecorded")),
          cell(metric.model_stage ?? detailLabel(context.locale, "notRecorded"))
        );
      }
    }
  }
  return body.childElementCount > 0 ? wrapper(metrics, detailLabel(context.locale, "details")) : undefined;
}

/** Conditions and raw source metrics for one expanded benchmark row. */
export function renderEvidenceDetails(records: BenchmarkRecord[], context: DetailContext): HTMLElement {
  const result = document.createElement("section");
  result.className = "model-detail-evidence-details";
  result.append(renderConditions(records, context));
  const metrics = renderMetricTable(records, context);
  if (metrics) result.append(metrics);
  if (records.length === 0) {
    const empty = document.createElement("p");
    empty.className = "model-detail-no-evidence";
    empty.textContent = detailLabel(context.locale, "notRecorded");
    result.append(empty);
  }
  return result;
}
