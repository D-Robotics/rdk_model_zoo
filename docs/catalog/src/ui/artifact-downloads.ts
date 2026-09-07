import type { Locale, ModelRecord, ModelVariant } from "../catalog/types";
import { isRunnableAsset } from "../catalog/variants";
import { detailLabel } from "./detail-labels";
import type { DetailContext } from "./detail-types";
import { appendEmptyRow, cell, table, wrapper } from "./detail-utils";

export function runnableAssets(variant: ModelVariant): ModelRecord["assets"] {
  return variant.assets.filter(isRunnableAsset);
}

function compactDownloadLabel(locale: Locale, extension: string): string {
  return `${locale === "zh" ? "下载" : "Download"} .${extension}`;
}

function appendAssetList(
  container: HTMLElement,
  assets: ModelRecord["assets"],
  context: DetailContext,
  compactLinks = false
): void {
  const list = document.createElement("ul");
  list.className = "model-detail-download-list";
  for (const asset of assets) {
    const item = document.createElement("li");
    const value = asset.url ? document.createElement("a") : document.createElement("span");
    const extension = asset.filename.split(".").pop()?.toLowerCase() || asset.format;
    value.textContent = compactLinks && asset.url
      ? compactDownloadLabel(context.locale, extension)
      : asset.filename;
    value.setAttribute("aria-label", `${context.locale === "zh" ? "下载" : "Download"} ${asset.filename}`);
    value.title = asset.filename;
    if (asset.url && value instanceof HTMLAnchorElement) {
      value.href = asset.url;
      value.download = asset.filename;
      value.dataset.action = "download-model";
    }
    item.append(value);
    if (!asset.url) {
      const missing = document.createElement("small");
      missing.className = "missing-data";
      missing.textContent = ` — ${detailLabel(context.locale, "downloadNotRecorded")}`;
      item.append(missing);
    }
    list.append(item);
  }
  container.append(list);
}

export function renderDownloadCell(
  element: HTMLElement,
  assets: ModelRecord["assets"],
  context: DetailContext
): void {
  const runnable = assets.filter(isRunnableAsset);
  if (runnable.length === 0) {
    element.textContent = detailLabel(context.locale, "noAssets");
    return;
  }
  appendAssetList(element, runnable, context, true);
}

/** Render the complete file list for the expanded row of one specification. */
export function renderAssetDetails(variant: ModelVariant, context: DetailContext): HTMLElement {
  const heading = document.createElement("h4");
  heading.textContent = detailLabel(context.locale, "assets");
  const assets = runnableAssets(variant);
  const assetTable = table(detailLabel(context.locale, "assets"));
  assetTable.className = "model-detail-assets-table";
  const header = assetTable.tHead!.insertRow();
  for (const label of [
    detailLabel(context.locale, "downloads"),
    detailLabel(context.locale, "modelFormat"),
    detailLabel(context.locale, "checksum")
  ]) header.append(cell(label, true));

  for (const asset of assets) {
    const row = assetTable.tBodies[0]!.insertRow();
    const filename = document.createElement("td");
    if (asset.url) {
      const link = document.createElement("a");
      link.href = asset.url;
      link.download = asset.filename;
      link.textContent = asset.filename;
      link.title = asset.filename;
      link.setAttribute("aria-label", `${context.locale === "zh" ? "下载" : "Download"} ${asset.filename}`);
      link.dataset.action = "download-model";
      filename.append(link);
    } else {
      filename.textContent = `${asset.filename} — ${detailLabel(context.locale, "downloadNotRecorded")}`;
    }
    row.append(filename, cell(asset.format), cell(asset.sha256 ?? detailLabel(context.locale, "notRecorded")));
  }
  if (assets.length === 0) appendEmptyRow(assetTable, detailLabel(context.locale, "noAssets"), 3);
  const result = document.createElement("section");
  result.className = "model-detail-artifact-details";
  result.append(heading, wrapper(assetTable, detailLabel(context.locale, "assets")));
  return result;
}
