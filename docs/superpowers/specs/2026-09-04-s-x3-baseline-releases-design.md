# RDK Model Zoo S/X3 Baseline Releases Design

**Date:** 2026-09-04

**Status:** Approved in conversation

**Target repository:** `D-Robotics/rdk_model_zoo`

## Purpose

Publish the first formal, immutable releases for the existing S and X3 delivery branches before extending the online catalog. The releases describe the source and benchmark evidence already present in each branch. They do not add a board-test gate or claim that every sample was revalidated on hardware.

## Release identities

| Line | Source branch | Version | Annotated tag | Release title |
| --- | --- | --- | --- | --- |
| RDK S | `rdk_s` | `1.0.0` | `s-v1.0.0` | `RDK Model Zoo S v1.0.0` |
| RDK X3 | `rdk_x3` | `1.0.0` | `x3-v1.0.0` | `RDK Model Zoo X3 v1.0.0` |

Each tag points to a dedicated release-preparation commit at the tip of its platform branch. Published tags are annotated and immutable.

## Required release contents

Both release commits contain:

- `VERSION`
- `CHANGELOG.md`
- `release/models.yaml`
- `release/benchmarks.yaml`
- `release/schemas/models.schema.json`
- `release/schemas/benchmarks.schema.json`
- `release/README.md`
- bilingual `docs/releases/<tag>.md`

GitHub Releases use the corresponding tagged notes and attach both manifests as `models.yaml` and `benchmarks.yaml`.

## Evidence rules

The manifests transcribe only facts that can be attributed to files in the tagged branch. Missing URLs, checksums, datasets, runtime versions, platform mappings, or benchmark conditions remain missing and are disclosed. A benchmark source uses the release tag and an exact repository path and Markdown section. Values from another platform are never copied into the target platform record.

Model assets remain outside Git unless they are already committed. `sha256` is `null` unless the branch already publishes a trusted digest. Download URLs are not fetched as a release gate.

## RDK S baseline

The S manifest inventories the local S samples and published external assets at the release commit. YOLOE is excluded from both manifests and release totals as requested; its existing source directory is left unchanged. ACT and Pi0 are recorded only if their gitlink evidence can be represented without presenting them as local, normalized samples.

S100, S100P, and S600 support is recorded per asset or benchmark only when the branch documentation or download script identifies that platform. Conflicting claims, placeholder evaluator documents, and assets without published benchmark data remain explicit limitations.

## RDK X3 baseline

The X3 release is labeled as a historical legacy baseline. Its inventory uses 15 logical model families and the concrete external assets documented by the branch. Entries are described as legacy notebooks/wrappers rather than normalized Model Zoo buildable samples. Missing X3 accuracy, incomplete benchmark conditions, unresolved FCOS distribution, and ambiguous upstream variants are preserved as limitations.

## Validation and publication

Validation covers YAML schema conformance, unique IDs, release identity, model-to-benchmark references, existing repository paths, exact source headings, summary counts, YOLOE exclusion, `git diff --check`, and tag/branch identity. It does not run model binaries, datasets, host builds, or RDK board tests.

Before publishing S/X3 Releases, the X5 Pages workflow is restricted to automatic deployment for `x5-v*` release tags. This prevents S/X3 releases from checking out tags that do not contain the X5 site. A later catalog release will consume all three platform manifests and replace this temporary platform filter.

## Deferred catalog work

After both releases are public, the online catalog will be reworked so each concrete model variant, task, version, and input shape receives its own card, while the same concrete model groups S, X3, and X5 availability and benchmark evidence. The UI will distinguish model numeric precision from evaluation accuracy.
