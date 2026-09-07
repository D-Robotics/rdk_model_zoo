import type { BenchmarkRecord, ModelRecord } from "../src/catalog/types";

/**
 * The first S manifest expanded YOLO 8/9/10 filenames without their `v` and
 * assigned nash-p URLs to rdk_s100. Correct the catalog copy using the tagged
 * samples/vision/ultralytics_yolo/model/{download_model.sh,README.md}.
 * Tags and numerical benchmark evidence remain unchanged. YOLOv9n is not in
 * the README's Published Models list; do not relabel it as a different size.
 */
export function correctSArtifacts(
  models: { models: Array<Pick<ModelRecord, "id" | "assets">> },
  evidence: { benchmarks: BenchmarkRecord[] }
): void {
  const canonicalFilename = (filename: string): string => filename.replace(/\byolo(8|9|10)(?=[a-z])/g, "yolov$1");
  for (const model of models.models) {
    if (model.id !== "ultralytics_yolo") continue;
    model.assets = model.assets
      .filter((asset) => !/(?:^|\/)yolov?9n_/.test(asset.filename))
      .map((asset) => {
        const filename = canonicalFilename(asset.filename);
        let url = asset.url ? canonicalFilename(asset.url) : undefined;
        if (url && filename.startsWith("nash-p/")) {
          url = url.replace("/rdk_s100/Ultralytics_YOLO_OE_3.7.0/", "/rdk_s600/Ultralytics_YOLO_OE_3.7.0/");
        }
        return { ...asset, filename, url };
      });
  }
  for (const record of evidence.benchmarks) {
    if (record.sample_id === "ultralytics_yolo" && record.asset_filename) {
      record.asset_filename = canonicalFilename(record.asset_filename);
    }
  }
}
