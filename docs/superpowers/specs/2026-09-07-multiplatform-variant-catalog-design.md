# Multi-platform Variant Catalog Design

**Date:** 2026-09-07

**Status:** Approved in conversation

## Purpose

Replace the X5-only model-family catalog with a public catalog of concrete model variants. Each card represents one exact model, task, and input configuration. The card shows its support, assets, performance measurements, and accuracy measurements for X5, S, and X3 without conflating model numeric precision with evaluation accuracy.

## Data sources

The catalog reads the immutable release manifests for `x5-v1.0.0`, `s-v1.0.0`, and `x3-v1.0.0`. It records the release tag with every platform payload and uses the existing immutable source evidence within every benchmark record. The site remains static and performs no model or board test.

## Variant identity

A card identity is the tuple:

1. canonical model family;
2. exact variant name or version;
3. task; and
4. input shape and format when recorded.

The identity registry maps only explicitly equivalent records across platform manifests. It does not infer equivalence from a common sample directory or a name prefix. Thus `YOLOv5s v7.0 Detect 640x640`, `YOLOv5su Detect 640x640`, and `YOLOv5s v2.0 Detect 640x640` are distinct cards. Records without a declared equivalence remain platform-specific cards.

## Presentation

Each card contains a platform-support row for X5, S, and X3. A supported platform links to its release tag and opens a platform-specific details section. Each details section contains independent Performance and Accuracy tables. A missing performance or accuracy table is labeled “not yet measured”; it never means data is confidential. Missing conditions and checksums are labeled “not recorded”.

Numeric model precision, such as `int8` or `quantized`, is displayed in variant metadata. Evaluation accuracy, such as Top-1 or mAP, is displayed only in the Accuracy table. The card does not duplicate accuracy as a precision property.

The summary reports concrete model-variant cards and support entries rather than source sample families. Search indexes canonical identity, aliases, tasks, filenames, and platform names. Platform, task, precision, and measurement filters operate on the aggregated card.

## Validation

The generator validates each source manifest against its tagged source tree, preserves source evidence, rejects YOLOE, rejects unsupported registry references, and rejects a registry mapping whose task or recorded input differs. Tests cover exact YOLO separation, S/X3 support aggregation, independent empty performance and accuracy states, and the EfficientFormer regression.

## Deployment

The Pages workflow builds the catalog from the three fixed tags on `rdk_x5` and deploys it through `workflow_dispatch` or an `x5-v*` Release. S and X3 releases remain immutable and do not overwrite the deployed multi-platform catalog.
