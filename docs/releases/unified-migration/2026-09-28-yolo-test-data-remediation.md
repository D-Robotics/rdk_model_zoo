# H2-DATA-R1 (bounded): Ultralytics YOLO test_data documentation — 2026-09-28

Status: **implemented by Claude Code + GLM per the Codex review
`2026-09-28-yolo-test-data-review.md`. This is an author report only — Codex
performs the independent review and sync; H2 is not closed by this document.**

Scope: add bilingual directory-local guides for
`samples/vision/ultralytics_yolo/test_data/` and one navigation line each in
the sample's root README pair. Documentation-only product change: every
pre-existing file in the sample is byte-identical with HEAD, no command block
was altered, and no download, board run, OE/HMCT toolchain, calibration or
quantization step was executed.

## Product changes

| File | Change |
| --- | --- |
| `samples/vision/ultralytics_yolo/test_data/README.md` | New English guide |
| `samples/vision/ultralytics_yolo/test_data/README_cn.md` | New Chinese guide, same anchors |
| `samples/vision/ultralytics_yolo/README.md` | One `entry-points` bullet linking the test_data guide |
| `samples/vision/ultralytics_yolo/README_cn.md` | Same bullet in Chinese |

## Findings addressed (H2-DATA-R1)

- **Directory-local distinction of the three content kinds.** The guide opens
  by separating input images (`bus.jpg`, `zebra_cls.jpg`), runtime display
  labels (the three `.names` tables) and historical illustrations
  (`result_detect*.jpg`, four `ultralytics_YOLO_*_demo` captures), with a
  per-file role/pixels/SHA-256 table covering all eleven tracked files.
- **Image loading expectations.** `--test-img` accepts any OpenCV-readable
  three-channel BGR image; the task resizes to the model's input geometry
  (default letterbox, exceptions linked), results restore original-image
  pixels, and filenames never override model metadata.
- **Custom-image CLI inputs and output locations.** Copyable blocks reuse only
  existing parser syntax: the zero-argument default entry, the documented
  zebra classification command, the runtime guide's custom-image/custom-model
  example, and a custom `--label-file` variant of the same form. Output
  semantics are stated without invented numbers: `--img-save-path` default
  `result.jpg` relative to the caller, parent directory created, existing file
  overwritten, `[Saved]` line; classification prints and writes no image;
  `Detection Report:` / `Top-K Classification Results:` console formats shown
  as placeholder templates; exit 0 with an empty detection list is valid.
  The no-bundled-aerial-image OBB boundary is explicit.
- **Display labels vs evaluation annotations vs ImageNet synsets.** The label
  section documents per-task defaults from `main.py`, the COCO-80 order with
  its six VOC-style display synonyms (indices 3/4/57/58/60/62), the
  dict-literal ImageNet file (1000 entries, index 340 `zebra`, no trailing
  newline, display names — not `n########` synsets), and the DOTA table's
  actual 15-name model-output order (`plane` 0, `ship` 1, `storage-tank` 2, …
  `swimming-pool` 14) with the warning that it differs from
  `datasets/dotav1/dota_classes.names` at 14 of 15 positions. A custom-model
  subsection states the observed out-of-range behavior: numeric-ID fallback in
  the detect report, in OBB drawing and in classification (`Unknown(<id>)`),
  while detect/seg rendered drawing raises `IndexError` before any image is
  written. The evaluator's different `--label-file` meaning (ordered synset
  list) is called out so the display-name file is not passed there.
- **Historical figures stay historical.** The provenance section records both
  delivery pins (X5 `ac11571…`, S `380e1a2…`), the byte-identity of the demo
  captures and `zebra_cls.jpg` with the X5 pin copies, the S-side origin of
  both result illustrations (the yolo26 one renamed to avoid the filename
  collision), the recorded source-image audit's characterization of the four
  demo captures as 2025-05-19 IDE/SSH screenshots whose on-screen values do
  not describe current defaults, and the explicit non-reference disposition of
  the pose/segmentation/classification captures. No file was generated or
  re-rendered; nothing here is presented as measurement evidence for the
  unified code, and the guide states future documentation work will not
  replace them with fresh captures.
- **Root navigation.** The root pair's `entry-points` list gains the test_data
  link; nothing else in either root file changed (diff: one insertion each).

## Verification (host, static)

- Byte preservation: all eleven pre-existing `test_data` files re-hashed and
  compared with HEAD — identical
  (`evidence/2026-09-28-yolo-test-data-remediation/asset-sha256.txt`); the
  root README pair diff is exactly one added bullet each
  (`link-anchor-scope-check.json`).
- Byte-identity relations documented in the guide were reproduced with
  `filecmp` against `datasets/coco`, `datasets/imagenet`,
  `samples/vision/resnet`, and the archived
  `platforms/{x5,s}` snapshots, including the two expected-difference rows
  (`ultralytics_dota_classes.names` vs both DOTA listings)
  (`byte-identity-check.json`).
- Label facts verified by executing the sample's own
  `rdk_yolo_utils.file_io`/`visualize` on the dev host with synthetic arrays:
  counts 80/15/1000, full orders, dict-literal parsing, index 340 `zebra`,
  and the four out-of-range-ID behaviors quoted above
  (`label-and-behavior-check.json`). No board, model or download involved.
- Local links and fragments resolve in both new files; anchor sets match
  (`link-anchor-scope-check.json`); the six fenced code blocks are
  byte-identical between the languages (`code-block-parity.json`).
- `tools/sample_contract/check.py --sample samples/vision/ultralytics_yolo`:
  0 violations, 0 exemptions applied (skip set unchanged);
  `samples/vision/ultralytics_yolo` host suite 143 tests OK; shared suite
  158 tests OK (`host-checks.log`).

## Remaining limits (not claimed)

- The two delivered guides describe facts read from the working tree and the
  archived snapshots; the pins themselves were not re-fetched from any remote,
  and the local SHA-256 values are byte digests, not publisher authentication.
- No inference was run, so no console transcript is quoted as output; all
  console text in the guides is placeholder-format or previously documented
  behavior, with scores/probabilities left as `<score>`/`<probability>`.
- Board accuracy, performance, conversion and calibration remain untouched
  topics: the guides only point at the existing runtime/evaluator/conversion
  documentation.
- The three unreferenced demo captures remain unreferenced by the product
  READMEs (unchanged disposition from the recorded source-image audit); the
  new guide documents that decision instead of changing it.
