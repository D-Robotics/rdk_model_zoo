# Phase 0.5 Q2 review — inference responsibility and interface contract (2026-09-20)

Scope: plan Q2 — inference contract document, root-guidelines linkage, resnet
and paddle_ocr runtime adjustments, contract tests, host verification.

## Delivered

| Artifact | Content |
| --- | --- |
| `docs/sample-standards/inference-contract.md` | §1 four-method interface + when post_process takes context; §2 five concept types (Input/Tensors/Context/RawOutputs/Result) concretized per task, context discipline, stateful-task rules; §3 responsibility table with explicit allowances (NMS/CTC helpers, visualization in main, runner exposes quant metadata but H1 transforms live in post_process), multi-stage rules (public per-stage pre/forward/post, readable order, stage/crop error attribution), LLM/streaming exemption; §4 file organization; §5 required tests; §6 reference implementations |
| `docs/Model_Zoo_Repository_Guidelines.md` | layering table now links the contract as landed |
| `samples/vision/resnet/runtime/python/classification.py` | `ClassificationTask` docstring now concretizes the five concept types; explains why classification omits the context argument in `post_process` (no geometry consumed) per contract §1 |
| `samples/vision/paddle_ocr/runtime/python/pipeline.py` | `OCRPipeline` docstring concretizes the five concept types; **new public `forward_detection` / `forward_recognition`** (previously the forward step was reachable only through private `_call_runner`); `run_detection`/`run_recognition` now compose through the public forward methods — single call path, no duplicate logic |
| `samples/vision/resnet/tests/test_stage_contract.py` | 4 tests: predict == explicit three steps (field equality); forward returns bit-identical validated raw fixture (no softmax/dequant/scale); interleaved sizes A/B/A keep per-call `ImageTransform` context (frozen, reproducible, distinct); `__call__` delegates to predict |
| `samples/vision/paddle_ocr/tests/test_stage_contract.py` | 5 tests: predict == explicit stage composition (detection + per-crop recognition); forward_detection / forward_recognition return bit-identical validated raw (no threshold/CTC); public forward attributes errors to their stage; recognizer failure keeps crop-index attribution (`recognizer stage failed for crop 1` + cause chain) |

## Verification

All commands, interpreter, package versions, and the catalog build recorded in
[evidence/2026-09-20-phase05-q2-host-tests.json](evidence/2026-09-20-phase05-q2-host-tests.json).

- resnet 39 OK · paddle_ocr 43 OK · ultralytics_yolo 59 OK · `_shared` 17 OK
  (158 total; the 2026-09-17 record was 149 — the delta is the 9 new
  contract tests plus suite growth since that checkpoint).
- Environment enablement was required on this Mac host (prior evidence ran on
  Windows `C:\Python313`): repo-local `.venv` (gitignored), numpy/pyyaml/
  scipy, opencv-python-headless pinned `<5` (board TROS carries an OpenCV 4.x
  line; the 5.x degenerate-rect behavior differences are not board-relevant),
  and `tools/catalog-publisher` `npm ci` + `catalog:build` to regenerate
  `dist/catalog.json` (sha256 recorded in the evidence JSON).

## Pre-existing defects found by running the suites (fixed, separate commits)

1. **Wrong S100 vocabulary hash constant** (`model_binding.py`): expected
   `769e7fa7…` matched no normalization of the committed dictionary; the file
   is byte-identical to `rdk_s:380e1a2` `test_data/ppocrv6_dict.txt`
   (`b5f2bfe2…`; 18708 lines + blank + space = 18710 classes, consistent with
   the recognizer output contract). Constant corrected to the verified source
   identity. This was failing on the integration baseline itself
   (develop@9f17f2a), independent of host platform.
2. **Over-pinned degenerate-crop assertion** (`test_geometry.py`): the
   collinear sub-case encoded the Windows cv2 build's angle-sign accident;
   macOS cv2 4.14 yields the rotated twin. The rdk_s source runs the identical
   crop chain with no degenerate branch, so the port is faithful. Assertion
   now checks the cross-build invariants; **board cv2 degenerate behavior is
   an open item for board smoke** (not closed here).
3. **Unresolved tempdir comparison** (`ultralytics_yolo/tests/`): macOS
   `/var` ↔ `/private/var` symlink; test now compares resolved forms.

None of these were fixed by weakening a rule: each fix either restores the
verified constant, asserts genuine invariants, or normalizes a path
comparison. Production behavior changed only in item 1 (wrong constant →
verified constant).

## Explicit limits / not-run

- Host tests prove host-boundary behavior only; no board run in Q2 (board
  smoke remains the user's per-batch gate; S600 not-run).
- The contract's forward-purity tests are fixture-based (identity comparison)
  plus design (no file calls in the modules); AST-level enforcement lands with
  Q3's checker, not here.
- `legacy.py` compatibility shims intentionally retain old interfaces
  (contract §4); they are not migrated to the four-method surface.
- ultralytics_yolo received environment/test repairs only — its Q1–Q5
  compliance is deferred to B9 per the baseline exit note.

## Verdict

Q2 deliverables complete: contract landed, both reference samples adjusted
(resnet docstring concretization; OCR public stage-forward methods), 9 new
contract tests, all suites green with recorded evidence, two real pre-existing
defects found and fixed. Proceed to Q3 (automated checker with negative
fixtures).
