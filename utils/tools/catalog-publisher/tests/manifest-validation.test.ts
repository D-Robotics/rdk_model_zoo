// @vitest-environment node
import { describe, expect, it } from "vitest";
import { hasExactMarkdownAtxHeading, validateNormalizedCatalog, validatePublishedManifests } from "../src/pipeline/manifest-validation";
import type { PlatformDocuments } from "../src/pipeline/manifest-validation";
import { fixtureRelease, repositoryRoot } from "./helpers/repository";

const repositoryUrl = "https://github.com/D-Robotics/rdk_model_zoo";

function publish(variant: string) {
  return fixtureRelease(variant);
}

describe("published manifest validation", () => {
  it("accepts a manifest pair whose evidence and references all resolve", async () => {
    const { source, documents } = await publish("valid");
    await validatePublishedManifests({ repositoryRoot, source, documents, repositoryUrl });
    await validateNormalizedCatalog({ repositoryRoot, source, documents, repositoryUrl });

    expect(documents.models.release.tag).toBe("x5-v1.0.0");
    expect(documents.models.models[0]?.id).toBe("convnext");
    expect(documents.benchmarks.benchmarks[0]?.sample_id).toBe("convnext");
  });

  it("rejects source section text that is not an exact Markdown ATX heading", async () => {
    const { source, documents } = await publish("non-heading-section");
    await expect(validateNormalizedCatalog({ repositoryRoot, source, documents, repositoryUrl }))
      .rejects.toEqual(expect.objectContaining({ code: "SOURCE_SECTION_NOT_FOUND" }));
  });

  it("rejects a benchmark whose sample_id is absent from models.yaml", async () => {
    const { source, documents } = await publish("invalid-reference");
    await expect(validateNormalizedCatalog({ repositoryRoot, source, documents, repositoryUrl }))
      .rejects.toEqual(expect.objectContaining({ code: "UNKNOWN_SAMPLE" }));
  });

  it("rejects a benchmark whose source ref is a mutable branch", async () => {
    const { source, documents } = await publish("mutable-ref");
    await expect(validatePublishedManifests({ repositoryRoot, source, documents, repositoryUrl }))
      .rejects.toEqual(expect.objectContaining({ code: "INVALID_SOURCE_REF" }));
  });

  it("rejects a resolved source path that escapes the distribution", async () => {
    const { source, documents } = await publish("valid");
    documents.benchmarks.benchmarks[0]!.source.path = "../../../../etc/passwd";
    await expect(validateNormalizedCatalog({ repositoryRoot, source, documents, repositoryUrl }))
      .rejects.toEqual(expect.objectContaining({ code: "INVALID_SOURCE_PATH" }));
  });

  it("rejects a Markdown source whose blob does not exist at the cited ref", async () => {
    const { source, documents } = await publish("valid");
    documents.benchmarks.benchmarks[0]!.source.path = "samples/vision/convnext/CHANGELOG.md";
    await expect(validateNormalizedCatalog({ repositoryRoot, source, documents, repositoryUrl }))
      .rejects.toEqual(expect.objectContaining({ code: "SOURCE_NOT_FOUND" }));
  });

  it("rejects an asset reference that no model publishes", async () => {
    const { source, documents } = await publish("valid");
    documents.benchmarks.benchmarks[0]!.asset_filename = "ConvNeXt_atto_999x999_nv12.bin";
    await expect(validateNormalizedCatalog({ repositoryRoot, source, documents, repositoryUrl }))
      .rejects.toEqual(expect.objectContaining({ code: "UNKNOWN_ASSET" }));
  });

  it("requires a non-empty source path and section", async () => {
    const { source, documents } = await publish("valid");
    documents.benchmarks.benchmarks[0]!.source.section = "  ";
    await expect(validateNormalizedCatalog({ repositoryRoot, source, documents, repositoryUrl }))
      .rejects.toEqual(expect.objectContaining({ code: "INVALID_SOURCE_LOCATOR" }));
  });

  it("accepts evidence that another repository holds, which this checkout cannot read", async () => {
    const { source, documents } = await publish("valid");
    documents.benchmarks.benchmarks[0]!.source = {
      repository_url: "https://github.com/D-Robotics/rdk_LeRobot_tools",
      ref: "326ea043be204de25223d95c7d918efe8672dc66",
      path: "models/act/README.md",
      section: "ACT Policy on RDK S600",
      provenance: "existing-repository-documentation"
    };
    await expect(validateNormalizedCatalog({ repositoryRoot, source, documents, repositoryUrl })).resolves.toBeUndefined();
  });

  it("still checks a source that names this repository itself", async () => {
    const { source, documents } = await publish("valid");
    documents.benchmarks.benchmarks[0]!.source.repository_url = repositoryUrl;
    documents.benchmarks.benchmarks[0]!.source.section = "## Not A Real Heading";
    await expect(validateNormalizedCatalog({ repositoryRoot, source, documents, repositoryUrl }))
      .rejects.toEqual(expect.objectContaining({ code: "SOURCE_SECTION_NOT_FOUND" }));
  });

  it("locates non-text evidence by caption but still requires the blob to exist at the ref", async () => {
    const { source, documents } = await publish("valid");
    const record = documents.benchmarks.benchmarks[0]!;
    // A non-Markdown artifact at the cited tag: located by the caption it
    // carries, never matched as a heading, but it must exist at that revision.
    record.source.path = "samples/vision/convnext/test_data/cheetah.JPEG";
    record.source.section = "modelOutput: Calibrated Cosine / Quantized Cosine";
    await expect(validateNormalizedCatalog({ repositoryRoot, source, documents, repositoryUrl })).resolves.toBeUndefined();

    record.source.path = "samples/vision/convnext/accuracy.jpg";
    await expect(validateNormalizedCatalog({ repositoryRoot, source, documents, repositoryUrl }))
      .rejects.toEqual(expect.objectContaining({ code: "SOURCE_NOT_FOUND" }));
  });

  it("reports accuracy metrics that publish no dataset", async () => {
    const { source, documents } = await publish("valid");
    documents.benchmarks.benchmarks[0]!.accuracy = [
      { metric: "top-1", value: 77.04, unit: "percent" }
    ];
    const warnings: string[] = [];
    await validateNormalizedCatalog({
      repositoryRoot, source, documents, repositoryUrl, onWarning: (message) => warnings.push(message)
    });
    expect(warnings).toEqual([expect.stringContaining("accuracy metrics have no published dataset")]);
  });

  it("runs the same checks against every platform's own schema", async () => {
    const { source, documents } = await publish("valid");
    // A distribution is validated against the schema it ships; a platform whose
    // manifest directory publishes none cannot be accepted silently.
    const schemaLess: typeof source = { ...source, manifestDirectory: "release/valid/samples" };
    await expect(validatePublishedManifests({ repositoryRoot, source: schemaLess, documents, repositoryUrl }))
      .rejects.toEqual(expect.objectContaining({ code: "MODELS_SCHEMA" }));
  });
});

describe("exact ATX heading matching", () => {
  // Section text that merely occurs in the document must not count: prose
  // mentions, link labels, inline code, and fenced blocks are all rejected;
  // only a heading line outside a code fence matches, and it must repeat the
  // section text exactly (level included).
  const proseMentions = [
    "# Source fixture",
    "",
    "This prose mentions ## Benchmark without defining a section.",
    "",
    "[Link labeled ## Benchmark](https://example.com)",
    "",
    "Inline code also mentions `## Benchmark`.",
    "",
    "```markdown",
    "## Benchmark",
    "```"
  ].join("\n");

  it("accepts an exact heading line outside a code fence", () => {
    expect(hasExactMarkdownAtxHeading("## Benchmark\nbody", "## Benchmark")).toBe(true);
    expect(hasExactMarkdownAtxHeading("text\n\n## Benchmark  \nbody", "## Benchmark")).toBe(true);
  });

  it("rejects section text that only occurs in prose, links, code, or fences", () => {
    expect(hasExactMarkdownAtxHeading(proseMentions, "## Benchmark")).toBe(false);
    expect(hasExactMarkdownAtxHeading("## Benchmarked\n", "## Benchmark")).toBe(false);
    expect(hasExactMarkdownAtxHeading("### Benchmark\n", "## Benchmark")).toBe(false);
    expect(hasExactMarkdownAtxHeading("## benchmark\n", "## Benchmark")).toBe(false);
  });

  it("accepts a heading that appears again outside a fence that also mentions it", () => {
    const content = `${proseMentions}\n\n## Benchmark\n`;
    expect(hasExactMarkdownAtxHeading(content, "## Benchmark")).toBe(true);
  });
});
