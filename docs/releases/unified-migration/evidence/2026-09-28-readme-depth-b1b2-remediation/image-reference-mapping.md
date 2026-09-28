# Image reference mapping — readme-depth B1/B2 remediation (2026-09-28)

Source pins: X5 `rdk_x5 @ac11571` (x5-v1.1.3), S `rdk_s @380e1a2` (s-v1.1.2).
Inventory basis: `docs/releases/unified-migration/evidence/2026-09-28-source-readme-image-audit/inventory.json`
(base `5311f9c4`). "en+cn" = the reference existed in both language source READMEs.
Every mapping below was made after viewing the actual image file, not from the filename.

## resnet

| Source reference | Maintained location | Disposition |
| --- | --- | --- |
| ac11571 `README.md`+`README_cn.md` `./test_data/ResNet_architecture.png` | root README `overview` → Algorithm background, `test_data/ResNet_architecture.png` | Restored via `git show` (sha256 `cebea796…`). Paper Figure 5 (basic + bottleneck block). Caption states both source names and the shared digest. |
| ac11571 `inference.png` (en+cn) | root README `performance` (new section), `test_data/inference.png` | Restored via `git show` (sha256 `16c9d04e…`). White-wolf Rank-1 screenshot; labeled historical X5 source record. |
| 380e1a2 resnet18/50/152 `resnet_architecture.png` (en+cn, 3×) | same single `test_data/ResNet_architecture.png` | Deduplicated: identical bytes in all four source locations (inventory lists the same digest); one copy + caption noting both names. |
| 380e1a2 resnet18 `result.png` | `test_data/result_resnet18_s.png`, root `expected-results` | Restored via `git show` (sha256 `9e7a4c8a…`), renamed to carry variant + origin; zebra top-1 0.9985 caption. |
| 380e1a2 resnet50 `result.png` | `test_data/result_resnet50_s.png`, root `expected-results` | Restored via `git show` (sha256 `be58ba73…`); zebra top-1 0.9956 caption. |
| 380e1a2 resnet152 `result.png` | `test_data/result_resnet152_s.png`, root `expected-results` | Restored via `git show` (sha256 `229c2b23…`); zebra top-1 0.9649 caption. |
| ac11571 + 380e1a2 `ResNet_architecture2.png` / `resnet_architecture2.png` (present in source test_data, listed only in source directory trees) | not restored | Not embedded in any source README body on either pin; no source explanation text exists to carry. Disposition recorded here instead of a silent drop. |
| X5 source Performance Data table (ResNet18 71.5/70.5, 2.95 ms, 449+) | root `performance` (new), conditions-unstated caveat + link to evaluator record | Source numbers preserved verbatim; evaluator remains the audit-authoritative copy. |

## mobilenetv1

| Source reference | Maintained location | Disposition |
| --- | --- | --- |
| ac11571 `depthwise&pointwise.png` (en+cn) | `overview` → Algorithm background (file already on disk, sha256 `48d3cb64…`) | Referenced with D_K×D_K depthwise + 1×1 pointwise explanation. |
| ac11571 `inference.png` (en+cn) | `performance` (file on disk, sha256 `6f07652b…`) | bulbul Rank-1 screenshot labeled historical X5 record. |
| 380e1a2 S READMEs | — | S source had zero image references (inventory: 0); no figure to map. S-side distinct feature text folded where applicable; nothing discarded. |

## mobilenetv2

| Source reference | Maintained location | Disposition |
| --- | --- | --- |
| ac11571 `mobilenetv2_architecture.png` (en+cn) | `overview` → Algorithm background (on disk, sha256 `7995faf5…`) | Stride-1/stride-2 inverted residual explanation. |
| ac11571 `inference.png` (en+cn) | `performance` (on disk, sha256 `7097e2e3…`) | Scottish deerhound screenshot labeled historical X5 record. |
| source-tree `seperated_conv.png` (listed in source README directory section, not embedded; on disk, sha256 matches source) | `overview` → Algorithm background, now referenced | Restored as a referenced figure with the paper-Figure-2 explanation; disposition for its prior unreferenced state recorded. |
| 380e1a2 S READMEs | — | Zero image references (inventory). S-side extra feature bullets (linear-bottleneck/params framing) covered by the restored X5 summary; no S figure exists. |

## mobilenetv3

| Source reference | Maintained location | Disposition |
| --- | --- | --- |
| ac11571 `MobileNetV3_architecture.png` (en+cn) | `overview` → Algorithm background (on disk, sha256 `bc978181…`) | SE-on-residual-path block explanation (paper Figure 4). |
| ac11571 `inference.png` (en+cn) | `performance` (on disk, sha256 `03b15192…`) | kit fox screenshot labeled historical X5 record. |
| 380e1a2 S READMEs | — | Zero image references (inventory). |

## mobilenetv4

| Source reference | Maintained location | Disposition |
| --- | --- | --- |
| ac11571 `MobileNetV4_architecture.png` (en+cn) | `overview` → Algorithm background (on disk, sha256 `944a191d…`) | UIB instantiation explanation (paper Fig. 4). |
| ac11571 `inference.png` (en+cn) | `performance` (on disk, sha256 `64930905…`) | great grey owl screenshot labeled historical X5 record. |
| 380e1a2 S READMEs | — | Zero image references (inventory). |

## efficientnet

| Source reference | Maintained location | Disposition |
| --- | --- | --- |
| ac11571 `EfficientNet_architecture.png` (en+cn) and 380e1a2 `efficientnet_architecture.png` (en+cn) | `overview` → Algorithm background, `test_data/efficientnet_architecture.png` (on disk) | Deduplicated: same bytes under two names across pins (sha256 `f0c7ccbe…`); kept the S-side on-disk filename, caption discloses both source names + digest. Paper Figure 2 compound-scaling explanation. |
| ac11571 `inference.png` (en+cn) | `performance` (on disk, sha256 `8ecf7529…`) | redshank screenshot labeled historical X5 record (input was redshank.JPEG, not Scottish_deerhound.JPEG — stated in caption). |
| 380e1a2 S READMEs | — | Only the architecture figure (mapped above); S latency table already maintained in root `performance`. |

## efficientformer

| Source reference | Maintained location | Disposition |
| --- | --- | --- |
| ac11571 `latency_profiling.png` (en+cn) | `overview` → Algorithm background (on disk, sha256 `a3439462…`) | Paper Figure 2 latency study; caption explicitly says iPhone 12/CoreML paper measurement, not RDK X5. |
| ac11571 `EfficientFormer_architecture.png` (en+cn) | `overview` → Algorithm background (on disk, sha256 `4fe4662f…`) | Paper Figure 3 MB4D/MB3D explanation. |
| ac11571 `inference.png` (en+cn) | `performance` (on disk, sha256 `7ddbce07…`) | bittern screenshot labeled historical X5 record. |
| (no S delivery of this sample) | — | — |

## efficientformerv2

| Source reference | Maintained location | Disposition |
| --- | --- | --- |
| ac11571 `EfficientFormerV2_architecture.png` (en+cn) | `overview` → Algorithm background (on disk, sha256 `0a3fd26e…`) | Paper Figure 2 panels (a)–(f) explanation. |
| ac11571 `inference.png` (en+cn) | `performance` (on disk, sha256 `907925ac…`) | goldfish screenshot labeled historical X5 record. |
| (no S delivery of this sample) | — | — |

## efficientvit

| Source reference | Maintained location | Disposition |
| --- | --- | --- |
| ac11571 `comparison_between_transformer_and_cnn.png` (en+cn) | `overview` → Algorithm background (on disk, sha256 `be1e2e39…`) | Paper Figure 2 runtime profiling; memory-bound-op explanation. |
| ac11571 `mhsa_computation.jpg` (en+cn) | `overview` → Algorithm background (on disk, sha256 `4dda6352…`) | Source alt text said "MHSA Computation"; the actual figure is the paper's MHSA-proportion accuracy study (Figure 3). Caption describes the real content per the no-blind-title-inheritance rule; filename unchanged. |
| ac11571 `efficientvit_msra_architecture.png` (en+cn) | `overview` → Algorithm background (on disk, sha256 `403d1c63…`) | Paper Figure 6 (a)(b)(c) explanation. |
| ac11571 `inference.png` (en+cn) | `performance` (on disk, sha256 `2a23e138…`) | hook screenshot labeled historical X5 record. |
| (no S delivery of this sample) | — | — |

## Prose/content mapping (beyond images)

- All nine: source "Algorithm Overview / Features" bullets restored verbatim-in-meaning into `overview` → `### Algorithm background` / `### 算法背景`, with paper + reference-implementation links.
- resnet: per-variant notes restored from S resnet18 ("lightweight… quick classification validation"), resnet50 (bottleneck blocks 1x1/3x3/1x1), resnet152 (152-layer capacity).
- mobilenetv2: S-side capability framing ("multi-class classification of a single image, Top-K") is already covered by the current intro/output description; no S figure exists.
- efficientnet: S-side Lite family framing maintained; per-variant geometry 224/240/260/300/380 retained from the current support text.
- efficientformer/v2/vit: source feature bullets restored; source performance tables already present and unchanged.
