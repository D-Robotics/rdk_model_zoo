# Image and explanation mapping — DOC-DEPTH special package

Author: Claude Code + GLM. Scope: `samples/vision/{ultralytics_yolo, fcos, bytetrack}`,
`samples/speech/kws` bilingual READMEs plus necessary fixed-source images, per the
[DOC-DEPTH-R1 dispatch](../../2026-09-28-source-readme-image-independent-review.md) and
[KWS-N1](../../2026-09-28-kws-independent-review.md). Source pins: X5 `ac11571`,
S `380e1a2`. All SHAs below were recomputed locally; images were only ever taken via
`git show <pin>:<path>` — no network, no board, no toolchain run.

Inventory basis: `evidence/2026-09-28-source-readme-image-audit/inventory.json`
rows for these four destinations.

## 1. samples/vision/ultralytics_yolo (dest for X5 `ultralytics_yolo` + S `ultralytics_yolo`/`ultralytics_yolo26`)

| Source reference (pin, file) | sha256 (prefix) | Disposition |
| --- | --- | --- |
| S `380e1a2` `ultralytics_yolo/README*.md` → `./test_data/result_detect.jpg` | `5d792a47…` | File already in tree (byte-identical with S pin, verified). **Restored reference + caption** in root `expected-results` (both languages): labelled as the S delivery's historical detection illustration on the bundled bus scene, with pin + SHA. Inventory listed this as the destination's only S-side missing filename reference. |
| S `380e1a2` `ultralytics_yolo26/README*.md` → `./test_data/result_detect.jpg` | `2631c661…` | Not in tree (`identical_files: []` in inventory). **Restored as new file** `test_data/result_detect_yolo26.jpg` (byte-identical, sha re-verified) and **referenced** in root `expected-results` with caption: the S `ultralytics_yolo26` delivery's own historical illustration, restored under a distinct name because `result_detect.jpg` was already taken; YOLO26 detection runs from the unified entry today. Not byte-identical and not visually identical to the S `ultralytics_yolo` one (class-ID+score labels, different scores/boxes), so no dedup merge; both kept with distinct identities. |
| X5 `ac11571` root README×2 → (no root images embedded; conversion only) | — | X5 source root embedded no illustrations. The existing root reference `test_data/ultralytics_YOLO_Detect_demo.jpg` (sha `926ad7e3…`, byte-identical with X5 pin) was already present and reviewed; left untouched. |
| conversion dataflow figures (`ultralytics_yolo_detect_dataflow.png` `10bf6b3a…`, `ltrb2xyxy.jpg` `73ed0036…`, `…_seg_dataflow.png` `8171b814…`, `…_pose_dataflow.png` `9c1576cd…`) | — | Already restored and reviewed in the previous conversion package (DFL/YOLO26/pose-51-channel explanations). **Not touched** per dispatch. |
| In-tree but never referenced by either pinned source README: `ultralytics_YOLO_Pose_demo.jpg` (`194ffd24…`), `ultralytics_YOLO_Seg_demo.jpg` (`a80979cc…`), `ultralytics_YOLO_CLS_demo.png` (`7ea5f228…`), `zebra_cls.jpg` (`53c9f26d…`, referenced as CLI input, not as figure) | — | **Explicit non-restoration disposition.** Viewed: Pose/Seg/Detect demos are IDE screenshots of an SSH session titled `rdk_model_zoo_s_cauchy [SSH: RDK_S100_24GB]` (dated 2025-05-19, old `samples/Vision/ultralytics_YOLO_*` layout) whose terminal details (e.g. NMS threshold 0.70, old paths) do not map cleanly to either current delivery's documented defaults; the pinned source READMEs give no caption/provenance for them. Embedding them would imply unverifiable board provenance, so they stay unreferenced (files remain in tree). Recorded here so the absence is a decision, not an oversight. |

Root `overview` also gained one source-attributed sentence (both fixed delivery
READMEs describe Ultralytics YOLO as a real-time vision model family covering the
four tasks; YOLO26 maintained as one family of this entry) — restoring the source
"Algorithm Overview" family framing that the maintained root had compressed away.

## 2. samples/vision/fcos (X5 `ac11571`)

| Source reference | sha256 | Disposition |
| --- | --- | --- |
| X5 `conversion/README*.md` → `./fcos_efficientnetb0_512x512_nv12.png` | `69c7b4bc…` | Files already in tree (verified identical); **references + captions restored** in conversion `toolchain-targets` (both languages) with dataflow explanation. Viewed: NV12 input → BPU `NV12TOYUV444` → `YUV444,NHWC,INT8` → `torch-jit-export_subgraph_0` (BPU) → 15 INT32 outputs; B0 levels 64×64→4×4. |
| X5 `conversion/README*.md` → `./fcos_efficientnetb2_768x768_nv12.png` | `159bf34f…` | Same treatment; B2 levels 96×96→6×6 (read from the graph itself). |
| X5 `conversion/README*.md` → `./fcos_efficientnetb3_896x896_nv12.png` | `8501a002…` | Same treatment; B3 levels 112×112→7×7. |
| X5 source conversion "Output Protocol" prose (5 cls + 5 box + 5 center-ness outputs; runtime reorders by fixed shapes) | — | **Restored** in the same conversion subsection, tied to the graphs; decode semantics remain in `runtime/python/README*.md` (linked, not duplicated: confidence formula `sqrt(sigmoid(cls_max)*sigmoid(center))` stays only there). |
| X5 root README → `./test_data/demo_rdkx5_fcos_detect.jpg` | — | Already referenced in maintained root `expected-results` with historical-source label. Untouched. |
| X5 root "Algorithm Overview" text | — | Maintained root overview already states the one-stage anchor-free formulation, five levels, paper + official implementation links (source-equivalent or deeper). No rewrite without cause; recorded as covered. |

## 3. samples/vision/bytetrack (S `380e1a2`)

> **Correction (DOC-SPECIAL-R2, 2026-09-28):** an earlier revision of this
> mapping described the two PNGs with swapped visual identities — it called
> `image1.png` the three-row (a)/(b)/(c) motivation figure and `image.png` a
> three-frame strip. Re-viewing both working-tree files individually (each
> byte-identical with the S pin) shows the opposite. The SHA-256 attributions
> below were always correct; only the visual descriptions were. The captions in
> the root READMEs and this mapping are corrected; the source-embedded file
> (`image1.png`) is unchanged and no filename swap was made.

| Source reference | sha256 | Disposition |
| --- | --- | --- |
| S root `README*.md` → `./test_data/readme_img/image1.png` | `fdab9b40…` (1,265,627 bytes) | In tree, unreferenced. **Restored reference + explanation** in root `overview` (both languages), described by actual content after DOC-SPECIAL-R2: **one horizontal strip of three street frames** with colored detection boxes and confidence values (left frame e.g. 0.94/0.92/0.83; middle frame down to 0.43 beside a red triangle; triangle markers yellow in the outer frames, red in the middle), plus the source's four-step association flow restored as text bullets. This is the only PNG the pinned source README embeds. |
| S tree `test_data/readme_img/image.png` (present in the fixed source tree; **not** embedded in any pinned source README) | `032728fb…` (3,510,950 bytes) | **Now embedded in the maintained root `overview` with an explicit provenance label** (both languages): viewed — the three-row illustration with `Frame t1/t2/t3` headers and row captions "(a) detection boxes", "(b) tracklets by associating high score detection boxes", "(c) tracklets by associating every detection box"; in the top row the smaller tracked person reads 0.8 (t1), then 0.4 (t2) and 0.1 (t3) — the 0.9 boxes belong to the taller foreground person (score attribution corrected in the final narrow pass; the initial DOC-SPECIAL-R2 wording "0.9 → 0.4 → 0.1" conflated two boxes) — and row (c) re-associates that person's dashed low-score detections annotated 0.4 and 0.1. Caption states verbatim that the figure is bundled in the source tree but was not embedded by the pinned source README. Earlier revision wrongly described this file as the strip and left it unreferenced; corrected per DOC-SPECIAL-R2 and the final narrow pass. |
| S `evaluator/README*.md` → `../test_data/readme_img/MOT17-01-SDP.gif` | `6b7a613f…` | In tree, unreferenced. **Restored** under evaluator `reference-results` as "Reference tracking effects (historical)" with pin + SHA caption; explicitly not a board rerun. |
| S `evaluator/README*.md` → `../test_data/readme_img/MOT17-07-SDP.gif` | `ff99c85a…` | Same treatment. |
| S root "Inference Result" tuning guidance; S evaluator "Parameter Tuning" + multi-class note | — | **Restored as applicability/tuning text** (root `expected-results`; evaluator `reference-results` subsection), rewritten under DOC-SPECIAL-R1 to match the tracker code: `--score-thres 0.25` filters before the tracker; `--track-thresh 0.3` partitions first/second association (0.1 < score < track-thresh) with new tracks at `track_thresh + 0.1`; `--match-thresh 0.8` is the maximum accepted first-association cost (1 − IoU fused with detection score); `--track-buffer 60` scales by `frame_rate / 30`. Tuning directions, not recalibrated thresholds. |

## 4. samples/speech/kws (S `380e1a2`) — KWS-N1

No images exist in the source (inventory: 0 references); the finding is prose depth.
Restored an in-place `### Algorithm and pipeline (MDTC)` / `### 算法与流程（MDTC）`
explanation inside the fixed `overview` section of both root READMEs:

- MDTC (Multi-Scale Dynamic Temporal Convolution) and the PaddlePaddle + PaddleAudio
  framework context, attributed to the fixed source ("the fixed source describes…").
- Source feature bullets (multi-scale convolution, dynamic convolution, edge-friendly,
  high accuracy) summarized **as source descriptions**, with the explicit boundary that
  the fixed source ships a compiled artifact without training code, so dynamic-weight
  behavior and accuracy wording are not re-asserted as measured facts here.
- What the temporal model consumes and how confidence arises: mono 16 kHz float32 →
  first 60000 samples (3.75 s, zero-padded) → PaddleAudio fbank `[1, 373, 80]`
  (25 ms/10 ms/80 mel) → BPU probabilities → clip confidence = max, no extra sigmoid
  → `score >= threshold` (default 0.5). Facts cross-checked against
  `runtime/python/README*.md` and the [KWS host review](../../2026-09-28-kws-independent-review.md);
  the 3.75 s correction, S100-only support and all boundaries remain untouched.
- The S-snapshot link is kept as supplementary history, no longer the only place
  carrying algorithm context (this was the KWS-N1 defect).

## Dedup / non-restoration summary (all explicit, none silent)

1. Ultralytics: S `result_detect.jpg` vs S yolo26 `result_detect.jpg` — different
   bytes and different rendering; both kept with distinct captions/identities
   (yolo26 renamed `result_detect_yolo26.jpg` to avoid collision).
2. Ultralytics Pose/Seg/CLS demo screenshots — left unreferenced (provenance
   ambiguous, no source captions); disposition above.
3. ByteTrack `image.png` — embedded in the maintained root with an explicit
   "bundled in the source tree, not embedded by the pinned source README"
   label (correction per DOC-SPECIAL-R2; the initial revision left it
   unreferenced while also misdescribing both PNGs' content).
4. Ultralytics conversion four dataflow figures — previously restored and reviewed;
   intentionally not re-touched.
5. FCOS root algorithm text — already source-equivalent; no change.
6. No figure was duplicated across sample levels; each source figure appears once,
   at the level where the source placed it (root overview / expected-results /
   conversion / evaluator).
