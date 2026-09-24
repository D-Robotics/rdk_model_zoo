# LPRNet evaluator

<a id="dataset"></a>
## Dataset

There is no source accuracy dataset or label file. The reproducible input is the
bundled `../test_data/test_input.dat` (float32 `1x3x24x94`), the same pre-packed
tensor the fixed source reads; `../test_data/example.jpg` is only a visual
reference. This is a single input fixture, not a license-plate accuracy
benchmark.

<a id="environment"></a>
## Environment

`compare.py` runs both implementations itself on the board: the fixed source
wrapper from `platforms/x5/samples/vision/lprnet/runtime/python` and the unified
task from `samples/vision/lprnet/runtime/python`. It requires the X5 runtime
(imported lazily after the identity gate) and a prepared `lpr.bin` plus the
`.dat` input. It does not import the SDK on the host and does not download
anything. Historical source performance is preserved below and was not re-run.

| Model | Test frames | FPS | Average latency | BPU usage | ION memory |
|---|---:|---:|---:|---:|---:|
| `lpr.bin` | 100 | 266 FPS | 3.75 ms | 9% | 1.11 MB |

<a id="command"></a>
## Evaluation command

From the repository root, on a board with the artifact and input prepared:

```bash
python3 samples/vision/lprnet/evaluator/compare.py \
  --target x5 \
  --output-dir /tmp/lprnet-compare-$(date -u +%Y%m%dT%H%M%SZ)
```

Use `--asset-id x5:lprnet:lpr.bin --model-path <file>` to compare an externally
prepared artifact, and `--input-dat <file>` to point at another packed input.
The output directory must not already exist.

<a id="metrics"></a>
## Metrics

The evaluator records, for both sides, the prepared input tensor, the raw float32
logits and the decoded plate, and reports per-tensor shape/dtype/finite checks
with `max_abs_diff`. Inputs and the decoded plate must be exactly equal; raw
logits use `atol=1e-5`. It returns `0` only when every check passes, `1` when the
two sides differ, and `2` when the run failed. No accuracy score is reported
because no labeled dataset is included.

<a id="outputs"></a>
## Outputs

`comparison.json` binds the run to `target`, `asset_id`, model/input/code SHA-256
digests, observed runtime metadata for both sides, `argv`, `cwd`,
`started_utc`/`finished_utc` and `return_code`, and lists every saved array file
with its own digest. Each recorded input, raw logits and result array is written
to its own `.npy` file in the same directory. A failed execution still writes the
manifest with `error`, `return_code: 2` and `passed: false`.

<a id="reference-results"></a>
## Reference results

The historical source reference is the `lpr.bin` row above (100 frames); it is
not a retest. Board comparisons (2026-09-24): same-board source/unified runs
passed on one X5 8GB and one X5 4GB with the bundled `test_input.dat` — both
rc=0 with every check true and `max_abs_diff` 0.0 for the input tensor, the raw
`(1,68,18,1)` logits, and the decoded plate (evidence: [8GB
recheck](../../../../docs/releases/unified-migration/evidence/2026-09-24-b7-binding-recheck/),
[4GB
run](../../../../docs/releases/unified-migration/evidence/2026-09-24-b7-other-x5-variants/)).
The board log prints an HBRT-library/model-build minor-version mismatch warning
when loading `lpr.bin`; it is preserved verbatim in the evidence and all checks in the recorded comparisons passed. These runs are numerical parity for one input on the unified
side at board-test commit `73a6de1`, not an accuracy benchmark.

<a id="boundaries"></a>
## Boundaries

The evaluator runs both sides itself and never substitutes a hand-supplied file
for a real inference. It does not download models, prepare an accuracy dataset,
or measure performance. Same-board runs are recorded for one X5 8GB and one X5
4GB with the bundled input (2026-09-24, reference results above); any other
board or input still requires its own run, and no license-plate accuracy claim
is made from these comparisons.
