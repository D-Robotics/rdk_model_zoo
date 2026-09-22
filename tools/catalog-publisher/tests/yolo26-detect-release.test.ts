import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { repositoryCatalog, repositoryRoot } from "./helpers/repository";

interface ReleaseAsset {
  platform: "x5" | "s100" | "s100p" | "s600";
  size: string;
  filename: string;
  url: string;
  sha256: string;
}

// Share fixed publication evidence with the downloader, not live manifest values.
const release: ReleaseAsset[] = JSON.parse(readFileSync(resolve(repositoryRoot,
  "samples/vision/ultralytics_yolo/tests/fixtures/yolo26_detect_release.json"), "utf8"));

describe("YOLO26 Detect published release", () => {
  it("pins five sizes on each of the four targets", () => {
    expect(release).toHaveLength(20);
    expect(release.map((asset) => `${asset.platform}:${asset.size}`).sort()).toEqual(
      ["x5", "s100", "s100p", "s600"].flatMap((platform) =>
        [..."nsmlx"].map((size) => `${platform}:${size}`)).sort()
    );
  });

  it.each(release)("publishes the exact $platform/$size URL and SHA256", async (asset) => {
    const catalog = await repositoryCatalog();
    const family = catalog.models.find((model) => model.id === "yolov26");
    const group = asset.platform === "x5" ? "x5" : "s";
    const expected = {
      filename: asset.filename,
      url: asset.url,
      sha256: asset.sha256
    };
    expect(family).toBeDefined();
    expect(family?.assets).toContainEqual(expect.objectContaining(expected));
    const platform = family?.platforms?.find((entry) => entry.platform === group);
    expect(platform?.assets).toContainEqual(expect.objectContaining(expected));
    const variants = family?.variants?.filter((variant) =>
      variant.hardware === asset.platform
      && variant.task === "object-detection"
      && variant.assets.some((candidate) => candidate.filename === asset.filename)
    );
    expect(variants).toHaveLength(1);
    expect(variants?.[0]?.assets).toContainEqual(expect.objectContaining(expected));
  });
});
