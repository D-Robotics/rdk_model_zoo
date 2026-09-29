# README bilingual pair remediation — author record

Implementer: Claude Code + GLM (documentation package; direction, independent
review and GitHub sync remain with Codex). Base `d38123be`; revised after
[the re-review](2026-09-28-readme-pair-rereview.md) (base `2f128f2f`) requested
R3-A/B/C and reporting corrections. This package fixes
DOC-R1, DOC-R2 and DOC-R3 from
[the independent review](2026-09-28-readme-pair-independent-review.md) and its
[findings.json](evidence/2026-09-28-readme-pair-review/findings.json).

Scope discipline: only the six README files named below plus this report and
[the remediation evidence directory](evidence/2026-09-28-readme-pair-remediation/)
were added or changed. The concurrent MiniCPM package's working-tree changes
(`samples/llm/minicpm5-2b/...`), the repository root index and the migration
plan were not touched and are not part of this package. The reviewer report,
historical evidence, plan and ledgers are left for Codex's re-review.

## DOC-R1 — EdgeNeXt evaluator image references (fixed)

- `samples/vision/edgenext/evaluator/README.md:45` — `zebra.JPEG` →
  `test_data/Zebra.jpg`.
- `samples/vision/edgenext/evaluator/README_cn.md:40` — `bittern.JPEG` →
  `test_data/Zebra.jpg`.

The tracked bundled image `test_data/Zebra.jpg` was visually confirmed to be
a zebra photograph, so both examples keep the original input intent and the
stated success criterion ("Top-5 containing a zebra-related class"). No other
line changed; the command, model path, labels and Top-K are identical in both
languages.

## DOC-R2 — OCR English inspection filename (fixed)

- `samples/vision/paddle_ocr/conversion/README.md:197` — dropped the wrong
  `en_` prefix: `model_output/en_PP-OCRv6_det_infer-deploy_640x640_nv12.hbm`
  → `model_output/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm`.

This aligns the English inspection command with the English emitted-filename
listing (line 176), the Chinese inspection command (`README_cn.md:185`), and
`s100/paddleocr_det_configs.yaml:18`
(`output_model_file_prefix: 'PP-OCRv6_det_infer-deploy_640x640_nv12'`).
The X5 line (`en_PP-OCRv3_det_infer-deploy_640x640_nv12.bin`) keeps its
prefix, matching `x5/ptq_yamls/paddleocr_det_config.yaml:18`. Quantization
recipes, YAMLs and all other commands are unchanged.

## DOC-R3 — Ultralytics source explanation restored (fixed)

Restored the source conversion guides' illustrated explanation into the
maintained guides, adapted to the maintained contract:

- `samples/vision/ultralytics_yolo/conversion/README.md` — new section
  `<a id="dataflow"></a>` at line 142 (between export and calibration).
- `samples/vision/ultralytics_yolo/conversion/README_cn.md` — same section at
  line 132.

Content, merged from the same-path source guides
`platforms/x5/samples/vision/ultralytics_yolo/conversion/README{,_cn}.md` and
`platforms/s/samples/vision/ultralytics_yolo/conversion/README{,_cn}.md`.
Source pins are distinct per platform and were verified byte-identical to the
archived copies in this worktree: X5 = rdk_x5 @`ac11571`, S = rdk_s
@`380e1a2bf42041af54be6f34935e50197cfadff9`:

- All four existing illustrations are embedded and explained, not merely
  linked: `ultralytics_yolo_detect_dataflow.png`, `ltrb2xyxy.jpg`,
  `ultralytics_yolo_seg_dataflow.png`, `ultralytics_yolo_pose_dataflow.png`.
  Body text preserves the source's explanation depth: why deployment need not
  compute all 8400 candidates, Sigmoid monotonicity, ReduceMax,
  Threshold(TopK) with the threshold transform, GatherElements + ArgMax, the
  bbox `1×64×k×1` gather, DFL SoftMax + expected-bin formula, dist2bbox
  (ltrb2xyxy) with the stride/grid-cell formulae, the segmentation
  coefficients × prototype combination, and the 57-channel pose head with the
  COCO keypoint table.
- Source attribution: the section's preamble names rdk_x5 @ac11571 and the
  `platforms/s` revision as the origin of the illustrations and prose.
  (First-revision wording; superseded by the distinct per-platform pins under
  "Revisions applied after the re-review" below.)
- Protocol scoping: the two box diagrams are explicitly scoped to the DFL
  protocol (YOLOv5u/v8/v9/v10/11/12/13); a closing subsection states that
  YOLO26 detection has no diagram, exports direct four-channel LTRB tensors
  (`[1,Hs,Ws,4]`, not `[1,Hs,Ws,64]`), has no DFL SoftMax/16-bin expectation
  stage, and that `ltrb2xyxy.jpg` belongs to the DFL decode — DFL and
  direct-LTRB artifacts are not interchangeable.
- Maintained-contract alignment: a "Where these stages run in this sample"
  paragraph states that the exported graph stops at the per-stride NHWC
  messages and the filtering/decode stages run in the maintained Python
  post-processing (`decode_dfl`, class-wise NMS; `decode_ltrb` for YOLO26;
  `segmentation_decode`; `pose_decode`), per
  `samples/vision/ultralytics_yolo/DETECTION_CONTRACT.md` and the exporter's
  `Detect_forward` (verified identical on rdk_x5 and rdk_s). Historical
  "after dequantization" wording is re-scoped: the runtime consumes
  already-dequantized F32 outputs, rejects integer/SCALE outputs at binding,
  and implements no manual dequantization
  (`runtime/python/tensor_io.py:343-355`). No manual-dequantization
  instruction was reintroduced.
- Existing commands, fixed anchors, parameters and content are untouched; the
  new section adds no shell commands.

Transcription corrections to the source prose (disclosed for review): the S
source's `y_1 = (y+0.5+b)×Stride(i)` typo is written as `y_2`, `n_3^3` as
`n_3^2`, and "$(t_p, t_p, b_p)$" is generalized to "per side" instead of
reproducing the typo. Formulae and figures are otherwise carried as published.

### Revisions applied after the re-review (R3-A/B/C and clarity)

- R3-A — pose shapes: the prose (EN and CN) now states the maintained
  contract — one class channel (single-class person pose models) plus
  `3 × 17 = 51` keypoint channels per cell — matching `pose_decode.py`
  (binds `cls` with 1 channel and `kpts` with `3 × nkpt`), replacing the
  inherited "57 channels" wording. A correction caption sits directly under
  the preserved pose diagram, marking its `×57` and 80-class labels as stale
  historical labels (the same diagram also reshapes to 3×17). The PNG is
  retained unmodified; no raster edit.
- R3-B — axis names: the grid-formula preamble now defines the cell as
  "column $x$, row $y$", with $x$ counting along the horizontal axis and $y$
  along the vertical axis, matching the decoder's `(x, y)` anchor convention
  (`gen_anchor` tiles x across columns and repeats y down rows, centers at
  half-integers), so $(x+0.5)·Stride$ is the horizontal and
  $(y+0.5)·Stride$ the vertical coordinate.
- R3-C — NMS exception: the blanket "followed by class-wise NMS" is now
  scoped to bindings that require one, and the S-series YOLOv10 NMS-free
  exception (same decode stages with `nms='none'` fixed) is stated with a
  link to the runtime README, which documents it.
- R3-D — pose decode formula (second pass): the pose paragraph no longer
  says multiplying keypoint coordinates by the stride yields input-image
  coordinates. It now states the maintained DFL decode
  `(raw_xy × 2 + anchor − 0.5) × stride` with half-integer cell-center
  anchors (`decode_kpts`, `runtime/python/rdk_yolo_utils/postprocess.py:461`),
  that `inverse_points` with `inverse_boxes` restores original-image geometry
  and Sigmoid converts keypoint visibility logits into scores
  (`pose_decode.py:101-103`), and — one comparative sentence — that the
  YOLO26 direct-LTRB pose branch instead uses `(raw_xy + anchor) × stride`
  (`pose_decode.py:77`). The "a different keypoint count changes the 51"
  clause is removed: the published binding fixes `nkpt = 17`
  (`model_binding.py:280`), and changing the shape declaration alone is
  stated not to produce a supported variant (EN and CN equally).
- Clarity — raw maximum vs probability: the ReduceMax paragraph now states
  `Sigmoid(max logits) = max Sigmoid(logits)`; the argmax ordering agrees
  before and after Sigmoid, and the raw output value is a logit, not yet a
  probability (EN and CN equally).
- Source pins: the section preamble (both languages) names the two distinct
  pins — X5 rdk_x5 @`ac11571`, S rdk_s
  @`380e1a2bf42041af54be6f34935e50197cfadff9` — verified byte-identical to
  the archived `platforms/x5` and `platforms/s` copies with `git show` +
  `diff` in this worktree. The first revision had described both as the X5
  pin; that is corrected here and in the READMEs.

## Static verification performed (host only)

Two kinds of "checker" appear in this package; they are distinct and only the
first was executed:

- Executed — the static README/sample-contract checker
  `tools/sample_contract/check.py` (documentation lint: anchors, links,
  bilingual parameter agreement). Commands run from the repository root with
  `../rdk_model_zoo/.venv/bin/python`; outputs in
  [evidence/2026-09-28-readme-pair-remediation/](evidence/2026-09-28-readme-pair-remediation/):
  1. First revision: `check.py --sample samples/vision/edgenext --sample
     samples/vision/paddle_ocr --sample samples/vision/ultralytics_yolo`
     (`check.py --help` consulted first; no exemptions flag — the reviewed
     exemptions baseline was emptied by the 2026-09-26 YOLO README debt
     closure) → `3 samples, 0 violations, 4 skips, 0 exemptions applied`;
     the 4 skips are the documented R-STAGE-PURITY policy skips (CLI
     `main.py` layers and the ultralytics legacy shim). Full JSON:
     `2026-09-28-checker-report.json`.
  2. After the re-review revisions: the same three-sample run was repeated
     (`2026-09-28-r2-checker-report.json`), plus link/image existence
     (`2026-09-28-r2-link-image-check.txt`) and bilingual shell-block
     agreement (`2026-09-28-r2-bilingual-shell-blocks.txt`) re-runs covering
     the edited Ultralytics section. First-revision evidence files
     (`2026-09-28-link-image-check.txt`,
     `2026-09-28-bilingual-shell-blocks.txt`, `2026-09-28-doc-r1-r2.diff`)
     are retained unchanged.
  3. After the second-pass R3-D revision: the same checks were re-run again
     with `r3-` prefixed evidence files, including spot assertions that both
     languages state the `(raw_xy × 2 + anchor − 0.5) × stride` DFL pose
     formula, the fixed `nkpt = 17` wording, the YOLO26
     `(raw_xy + anchor) × stride` contrast, and that the old
     "multiply by stride" sentence is gone.
- Not executed — any model/toolchain check: no `hb_mapper checker`, no
  `hb_compile`, no `hrt_model_exec model_info`, no weight download, export,
  calibration, quantization, compilation, board run, SDK call or toolchain
  provisioning. The existing quantization READMEs are treated as trusted
  source material per the 2026-09-28 user scope. `hrt_model_exec
  model_info` and the board functional check remain described commands, not
  executed ones. No mechanical tests were added.

## Out-of-repository action disclosure

In the first revision round, outside the assigned repository file scope, the
implementer also updated its local Claude Code tooling-memory files on this
machine (host-side notes about the venv interpreter and the checker's
exemptions baseline). That content is private local tooling state: it is not
reproduced in this repository, no repository text derives from it, and the
remediation diffs above contain no memory paths or content. No further local
memory reads or writes were performed in this revision round.

## Status

DOC-R1, DOC-R2: accepted by the re-review, unchanged since. DOC-R3:
re-revision applied per R3-A/B/C, the clarity and reporting corrections, and
the second-pass R3-D pose-decode correction; awaiting independent acceptance.
This report does not change H1/H9 or batch acceptance states.
