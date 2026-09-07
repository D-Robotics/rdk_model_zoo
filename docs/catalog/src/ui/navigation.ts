import type { CatalogQuery } from "../catalog/query";
import { normalizeHardware } from "../catalog/variants";

export function readCatalogQuery(url: URL): CatalogQuery {
  const params = url.searchParams;
  const sort = params.get("sort");
  const benchmark = params.get("benchmark");
  const list = (key: string): string[] => params.getAll(key).filter(Boolean);
  return {
    text: params.get("q") ?? "",
    platform: normalizeHardware(params.get("platform") ?? "") ?? "",
    tasks: list("catalogTask"), formats: list("format"), precisions: list("precision"),
    sort: sort === "latency" || sort === "fps" || sort === "accuracy" ? sort : "name",
    benchmark: benchmark === "performance" || benchmark === "accuracy" || benchmark === "none" ? benchmark : "all"
  };
}

export function writeCatalogQuery(url: URL, query: CatalogQuery): URL {
  const next = new URL(url);
  const params = next.searchParams;
  for (const key of ["q", "platform", "catalogTask", "format", "precision", "sort", "benchmark"]) params.delete(key);
  if (query.text) params.set("q", query.text);
  if (query.platform) params.set("platform", query.platform);
  for (const task of query.tasks) params.append("catalogTask", task);
  for (const format of query.formats) params.append("format", format);
  for (const precision of query.precisions) params.append("precision", precision);
  if (query.sort !== "name") params.set("sort", query.sort);
  if (query.benchmark !== "all") params.set("benchmark", query.benchmark);
  return next;
}
