# LPRNet evaluator

<a id="dataset"></a>
## Dataset

There is no source accuracy dataset or label file. The reproducible smoke input is the bundled `../test_data/test_input.dat` (float32 `1x3x24x94`) with `../test_data/example.jpg` as a visual reference. It is a single input fixture, not a license-plate accuracy benchmark.

<a id="environment"></a>
## Environment

The evaluator itself is host-runnable with Python and NumPy and does not import `hbm_runtime`. Board generation of raw evidence requires the X5 runtime and two model runs under the same model/input conditions. The complete source historical X5 row is preserved below and was not re-run.

| Model | Test frames | FPS | Average latency | BPU usage | ION memory |
|---|---:|---:|---:|---:|---:|
| `lpr.bin` | 100 | 266 FPS | 3.75 ms | 9% | 1.11 MB |

<a id="command"></a>
## Evaluation command

First save complete raw float32 outputs from the legacy and unified board runs to unique files. The legacy source path is `platforms/x5/samples/vision/lprnet/runtime/python`; the unified path is `samples/vision/lprnet/runtime/python`. Then, from the repository root, run:

```bash
python3 samples/vision/lprnet/evaluator/compare.py \
  --legacy-raw /tmp/lprnet-run-legacy/raw.bin \
  --unified-raw /tmp/lprnet-run-unified/raw.bin \
  --shape 1 68 18 \
  --output /tmp/lprnet-run-unified/compare.json
```

The files must be complete raw float32 arrays. The command returns `0` only when shape, dtype, every raw value, and decoded plate agree; it returns `2` otherwise. This migration did not run a board comparison.

<a id="metrics"></a>
## Metrics

The primary consistency metric is exact `array_equal` for raw float32 logits, followed by exact equality of the source CTC-style decoded plate. No accuracy score is reported because no labeled dataset is included. Performance numbers above are historical source measurements with their original test conditions.

<a id="outputs"></a>
## Outputs

`compare.py` prints and optionally writes JSON containing shape, dtype, raw equality, both decoded strings, and status. Preserve both raw arrays and the report under a unique run directory; do not overwrite a prior evidence run.

<a id="reference-results"></a>
## Reference results

Historical source reference is the table above for `lpr.bin` and 100 frames. Current host and board comparison status is `not-run`.

<a id="boundaries"></a>
## Boundaries

This evaluator compares complete saved evidence; it does not download models, load an SDK, create a label benchmark, or claim board compatibility. A successful host comparison fixture would still be `supported-not-run` until the same-board source/unified run is recorded.
