import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const webRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");

function loadDetailView() {
  const source = readFileSync(resolve(webRoot, "src/detail-view.js"), "utf8");
  const context = {
    window: { MODEL_PROPERTIES: {}, HubI18n: { locale: "zh", apply() {} } },
    document: { documentElement: { clientWidth: 1280, classList: { add() {}, remove() {} } } },
    location: { hash: "" },
    innerHeight: 900,
    AbortController,
    Event,
    URL,
  };
  context.globalThis = context;
  vm.runInNewContext(source, context, { filename: "src/detail-view.js" });
  return context.window.ModelDetail;
}

test("detail view sorts variants by logical variantKey and matches across chips with different build slugs", () => {
  const detailView = loadDetailView();
  const base = {
    catalogId: "edgenext/edgenext/cls",
    name: "EdgeNeXt",
    family: "edgenext",
    task: "图像分类",
    taskId: "image-classification",
    description: "EdgeNeXt",
    licenseName: "MIT",
    licenseUrl: "https://opensource.org/licenses/MIT",
    releaseStatus: "released",
    assets: [{ filename: "model.bin", url: "https://model-zoo-web.oss-cn-beijing.aliyuncs.com/rdk_x5/edgenext/m.bin", sha256: "a".repeat(64), sizeBytes: 100 }],
    benchmark: { performance: [], accuracy: [] },
  };
  const models = [
    { ...base, id: "en-base-x5", variantName: "EdgeNeXt Base", variantKey: "base", modelSize: "base-usi-in1k-224", releasePlatform: "X5", platforms: ["X5"], hardware: "RDK X5", sample: "edgenext/edgenext/cls/base-usi-in1k-224" },
    { ...base, id: "en-xxs-x5", variantName: "EdgeNeXt XX-Small", variantKey: "xx-small", modelSize: "xx-small-in1k-224", releasePlatform: "X5", platforms: ["X5"], hardware: "RDK X5", sample: "edgenext/edgenext/cls/xx-small-in1k-224" },
    { ...base, id: "en-xs-x5", variantName: "EdgeNeXt X-Small", variantKey: "x-small", modelSize: "x-small-in1k-224", releasePlatform: "X5", platforms: ["X5"], hardware: "RDK X5", sample: "edgenext/edgenext/cls/x-small-in1k-224" },
    { ...base, id: "en-base-s600", variantName: "EdgeNeXt Base", variantKey: "base", modelSize: "base-usi-in1k-224-s-int16-o1-20261103", releasePlatform: "S600", platforms: ["S600"], hardware: "RDK S600", sample: "edgenext/edgenext/cls/base-usi-in1k-224-s-int16-o1-20261103" },
    { ...base, id: "en-xs-s600", variantName: "EdgeNeXt X-Small", variantKey: "x-small", modelSize: "x-small-in1k-224-s-int16-o1-20261103", releasePlatform: "S600", platforms: ["S600"], hardware: "RDK S600", sample: "edgenext/edgenext/cls/x-small-in1k-224-s-int16-o1-20261103" },
  ];
  const html = detailView.render({ model: models[2], models, data: { models } });

  // Dropdown options inside benchmark-variant listbox must be ordered XX-Small -> X-Small -> Base
  const labels = [...html.matchAll(/role="option"[^>]*data-value="(en-[^"]+)"[^>]*data-label="([^"]+)"/g)].map(m => m[2]);
  assert.deepEqual(labels, ["EdgeNeXt XX-Small", "EdgeNeXt X-Small", "EdgeNeXt Base"]);

  // Sidebar S600 platform switch button from X5 X-Small must target S600 X-Small (en-xs-s600), not Base!
  assert.match(html, /data-platform-target="en-xs-s600"[^>]*aria-label="切换到 RDK S600"/);
});
