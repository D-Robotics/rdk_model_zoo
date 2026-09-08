import { expect, it } from "vitest";
import { resolve } from "node:path";
import { buildMultiplatformCatalog } from "../scripts/multiplatform-catalog";
import { mountCatalog } from "../src/app";

it("opens a YOLOv8 deep link using the complete production catalog", async () => {
  const catalog = await buildMultiplatformCatalog(resolve(process.cwd(), "../.."));
  window.history.replaceState({}, "", "/rdk_model_zoo/?model=yolov8&hardware=x5&task=object-detection");
  document.body.innerHTML = '<main id="app"></main>';
  const app = mountCatalog(document.querySelector<HTMLElement>("#app")!, catalog, { locale: "zh" });
  expect(document.querySelector(".model-details h1")?.textContent).toBe("YOLOv8");
  expect(document.querySelector<HTMLElement>(".catalog-directory")?.hidden).toBe(true);
  expect(document.querySelector('.model-details [data-hardware="x5"][aria-selected="true"]')).not.toBeNull();
  app.destroy();
});
