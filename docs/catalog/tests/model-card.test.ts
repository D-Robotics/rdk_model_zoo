import { beforeEach, describe, expect, it, vi } from "vitest";
import type { HardwareId, ModelRecord } from "../src/catalog/types";
import { buildModelCardViewModel } from "../src/catalog/card-view-model";
import { renderModelCard } from "../src/ui/model-card";
import { benchmarkFixture, createModelFixture } from "./fixtures/catalog";

function modelWithVariants(): ModelRecord {
  return createModelFixture({
    id: "yolov8",
    name: "YOLOv8",
    tasks: ["object-detection"],
    assets: [],
    benchmarks: [
      benchmarkFixture({
        id: "yolov8n-x5",
        variant_id: "yolov8n-detect-640",
        display_name: "YOLOv8n on RDK X5",
        environment: { hardware: "RDK X5" }
      }),
      benchmarkFixture({
        id: "yolov8s-x5",
        variant_id: "yolov8s-detect-640",
        display_name: "YOLOv8s on RDK X5",
        environment: { hardware: "RDK X5" }
      }),
      benchmarkFixture({
        id: "yolov8n-s100",
        variant_id: "yolov8n-detect-640",
        display_name: "YOLOv8n on RDK S100",
        environment: { hardware: "RDK S100" }
      })
    ]
  });
}

describe("model family card view model", () => {
  beforeEach(() => {
    document.body.innerHTML = "";
    window.history.replaceState({}, "", "/?platform=x5");
  });

  it("keeps family specifications together and scopes them to the selected hardware", () => {
    const model = modelWithVariants();
    const all = buildModelCardViewModel(model);
    const x5 = buildModelCardViewModel(model, "x5");
    const s100 = buildModelCardViewModel(model, "RDK S100");

    expect(all.specifications).toEqual(["YOLOv8n", "YOLOv8s"]);
    expect(all.hardware).toEqual(["x5", "s100"]);
    expect(x5.specifications).toEqual(["YOLOv8n", "YOLOv8s"]);
    expect(x5.variantCount).toBe(2);
    expect(s100.specifications).toEqual(["YOLOv8n"]);
    expect(s100.platform).toBe("s100");
  });


});

describe("model family card", () => {
  beforeEach(() => {
    document.body.innerHTML = "";
    window.history.replaceState({}, "", "/?platform=x5");
  });

  it("renders compact text, scoped specifications, hardware controls, and detail action", () => {
    const selected = vi.fn<(modelId: string, hardware?: HardwareId) => void>();
    const card = renderModelCard(modelWithVariants(), "x5", "en", selected);
    document.body.append(card.element);

    expect(card.element.querySelector(".model-card-visual, img, svg")).toBeNull();
    expect(card.element.querySelector(".card-specifications")?.textContent).toBe("YOLOv8n · YOLOv8s");
    expect(card.element.querySelectorAll(".hardware-badges [data-hardware]")).toHaveLength(2);
    expect(card.element.querySelector('[data-action="open-details"]')).not.toBeNull();
  });

  it("keeps native title modifier clicks and removes its own listeners on destroy", () => {
    const selected = vi.fn<(modelId: string, hardware?: HardwareId) => void>();
    const card = renderModelCard(modelWithVariants(), "", "en", selected);
    document.body.append(card.element);
    const link = card.element.querySelector<HTMLAnchorElement>("h3 a")!;
    // Prevent jsdom's unimplemented navigation while preserving the modified
    // click path that the component intentionally leaves to the browser.
    link.addEventListener("click", (event) => event.preventDefault(), { capture: true, once: true });
    const modified = new MouseEvent("click", { bubbles: true, cancelable: true, button: 0, ctrlKey: true });
    link.dispatchEvent(modified);
    expect(selected).not.toHaveBeenCalled();

    card.element.querySelector<HTMLButtonElement>('[data-action="open-details"]')!.click();
    expect(selected).toHaveBeenCalledTimes(1);
    card.destroy();
    card.element.querySelector<HTMLButtonElement>('[data-action="open-details"]')!.click();
    expect(selected).toHaveBeenCalledTimes(1);
  });
});
