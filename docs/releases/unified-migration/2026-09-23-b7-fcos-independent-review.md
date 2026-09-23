# B7 FCOS independent review

Date: 2026-09-23  
Scope: `samples/vision/fcos/**`, fixed X5 source at
`ac115717197920355fc390bb04299b20e6436864`, `docs/release/x5/models.yaml`,
and the applicable sample/inference contracts.  This is an independent review
of the dirty working tree; no FCOS file was modified, and no model, SDK,
network, remote host, or board was used.

## Findings

### [P1] Manifest points at a missing required download entrypoint

- **Location:** `docs/release/x5/models.yaml:230-231`; the sample tree has
  `model/download.py`, `model/download.sh`, and `model/fulldownload.sh`, but no
  `samples/vision/fcos/model/download_model.sh`.
- **Evidence:** the manifest's `download_scripts` includes
  `samples/vision/fcos/model/download_model.sh`; `find samples/vision/fcos`
  found no such file. The manifest is the release-facing source of the model
  entrypoint, so a consumer following the published path gets a missing-file
  failure.
- **Impact:** the FCOS model preparation contract is broken for the manifest
  entrypoint even though the Python downloader and the two other scripts work
  under mocks.
- **Minimum fix:** add `download_model.sh` as a compatibility wrapper that
  delegates to `download.py` with the same target/variant behavior, or change
  the manifest only after the release entrypoint decision is explicit and
  verified. Keep all three variants and exact asset IDs visible.

### [P1] The documented board download path hard-codes a developer `.venv`

- **Location:** `samples/vision/fcos/model/download.sh:3-5` and
  `model/fulldownload.sh:3-5`; the root quickstart invokes these scripts as a
  board preparation step (`samples/vision/fcos/README.md:43-53`).
- **Evidence:** both scripts execute `${ROOT}/.venv/bin/python`. The sample
  prerequisites describe a board image providing `hbm_runtime`, while the
  repository `.venv` is a host development fixture and is not a board-image
  requirement. On a board checkout without that repository-local venv, the
  first command fails before `download.py` runs.
- **Impact:** the advertised model preparation path is not portable to the
  target environment and prevents the inference quickstart from reaching the
  model step.
- **Minimum fix:** invoke an explicit available interpreter such as
  `python3` (or a documented `PYTHON` override defaulting to `python3`) from
  the shell wrappers. Keep `.venv/bin/python` only in host evidence commands,
  not in the customer-facing board path.

### [P1] `--resize-type 1` produces incorrect original-image coordinates

- **Location:** `samples/vision/fcos/runtime/python/tensor_io.py:49-63` and
  `samples/vision/fcos/runtime/python/fcos.py:105-111`.
- **Evidence:** `prepare()` explicitly supports letterbox mode, records
  `(top, bottom, left, right)` padding and the resized shape in the frozen
  `ImageContext`, and the CLI exposes `--resize-type` choices `0` and `1`
  (`runtime/python/main.py:43`). `post_process()` then always applies the
  direct-resize ratios `orig_w/input_width` and `orig_h/input_height` and never
  consumes `context.resize_type` or `context.pad`. For a non-square image in
  letterbox mode, model-space y coordinates still include the top padding, so
  the returned boxes are shifted and scaled incorrectly. The fixed source has
  the same direct-ratio behavior, but the unified CLI currently advertises the
  letterbox option as a usable mode and exposes the context needed to fix it.
- **Impact:** users selecting the documented non-default geometry receive
  incorrect boxes; this is a functional detection result error rather than a
  documentation-only discrepancy.
- **Minimum fix:** either implement inverse letterbox mapping from the frozen
  context (`(coord-pad)/scale`, then clip) and add an A/B test with a
  non-square image, or remove/reject `--resize-type 1` until the target artifact
  has an explicitly verified letterbox contract. Keep the default direct-resize
  source behavior separately documented.

### [P1] FCOS evaluator is a manual recipe, not the required self-contained comparison

- **Location:** `samples/vision/fcos/evaluator/README.md:17-19,26-48,62,76` and
  the corresponding Chinese README.
- **Evidence:** the document explicitly says there is “no separate evaluator
  executable” and instructs an operator to run the fixed source in a separate
  checkout. The command captures one `model_info` text file, unified JSON and
  an output image; it does not execute the source, dump either implementation's
  fifteen raw arrays, record runtime metadata for both sides, hash source code,
  or compare raw/results. The stated outputs call source raw files “optional”.
  In addition, lines 28-29 generate the directory timestamp twice, so the
  `mkdir` path and `EVIDENCE` path can differ at a second boundary. The README
  itself acknowledges at line 76 that the procedure requires an operator.
- **Impact:** there is no executable, reproducible same-board source/unified
  evidence path satisfying the evaluator contract. A user cannot obtain the
  claimed complete raw/result/identity record by following the command block.
- **Minimum fix:** add `evaluator/compare.py` (or explicitly mark evaluator
  delivery incomplete) that gates `x5`, runs the source and unified stages on
  the same artifact/image/thresholds, captures all fifteen raw arrays before
  dequantization, metadata, decoded results, command argv, source/code/model/
  input hashes, and a comparison decision in a new output directory. Use one
  captured output-directory variable rather than evaluating the timestamp
  twice. Add a fake-runtime/static test for the host-side evidence schema;
  retain board execution as `not-run` until actually performed.

### [P2, confirmed code path / artifact metadata needed] Float outputs are accepted under an unconditional dequant contract

- **Location:** `samples/vision/fcos/runtime/python/model_binding.py:35,234-238`
  and `samples/vision/fcos/runtime/python/fcos.py:75-76`.
- **Evidence:** `FCOSContract.output_transform` is always `"dequant"`, and
  `bind_model()` permits `float16` and `float32` in `_ALLOWED_OUTPUT_DTYPES`.
  `post_process()` consequently always calls `apply_output_transform("dequant", ...)`.
  If an observed artifact reports an F32 output plus a quant descriptor, the
  helper numerically applies scale/zero-point to already-float values; if it
  reports F32 without descriptors, binding rejects it. The source audit and
  current fixture establish the integer/dequant path, but no published model
  metadata was available to prove that every released FCOS binary is integer
  output.
- **Impact:** a future or differently exposed artifact can silently produce
  wrong confidence/box values instead of being rejected as a contract mismatch.
- **Minimum fix:** bind the exact published output dtype contract and reject
  F16/F32 for this `dequant` variant, or select `raw_f32` based on reviewed
  artifact facts and test both branches explicitly. Do not leave a float path
  that is always sent through integer dequantization.

## Independent verification

Using the repository `.venv` and no model or SDK:

- `.venv/bin/python -m unittest discover -s samples/vision/fcos/tests -v` — **13 passed**.
  This includes all three manifest identities, five-by-three metadata binding,
  source numerical dequant/decode fixture comparison, raw-object preservation,
  context A/B/A, CLI help/list/dry-run, and mocked all-variant download calls.
- `.venv/bin/python tools/sample_contract/check.py --sample samples/vision/fcos
  --parser-mode import --format json` — **0 violations**, with the expected
  CLI-layer stage-purity policy skip.

These checks do not validate the four findings above: the missing manifest
script, board interpreter availability, letterbox inverse geometry, and
self-contained evaluator evidence remain outside the current tests.

## Three-dimensional conclusion

- **Scope:** source protocol and all three X5 variants are represented. Binding
  resolves fifteen outputs as five classification, five box, and five
  center-ness heads by observed shape; quantization, FCOS confidence, stride
  decode, NMS, and direct-resize geometry match the fixed source fixture.
- **Host:** the independent host suite and contract checker pass. Host success
  does not establish model-file availability, board runtime compatibility, or
  board numerical output.
- **Board/delivery:** board validation is **not-run** and delivery is **not
  ready** until the manifest/download path, letterbox behavior (or its removal),
  and evaluator evidence implementation are resolved. No board-not-run status
  is being treated as a separate failure.

## Unmeasured boundaries

No model binary was downloaded; no source or unified SDK inference was run; no
X5 identity, runtime metadata, performance, COCO accuracy, raw board tensor,
or board-side source comparison was observed. Conversion remains source-gap
only: no checkpoint/export/calibration/OE recipe or publisher SHA-256 is
available in the fixed source/manifest.
