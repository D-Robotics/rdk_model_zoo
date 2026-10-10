English | [简体中文](README_cn.md)

# LPRNet evaluator

<a id="dataset"></a>

## Dataset

There is no source accuracy dataset or label file. The reproducible input is the
bundled `../test_data/test_input.dat` (float32 `1x3x24x94`), the same pre-packed
tensor the source runtime reads. `../test_data/example.jpg` depicts a different
plate; the DAT reference is `渝A999U9`. This is a single input fixture, not a license-plate accuracy
benchmark.

<a id="directory"></a>
## Directory structure

```text
evaluator/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── compare.py  # Python script
├── evaluate.py  # Plate reference comparison
└── source_reference.py  # Python script
```

<a id="environment"></a>
## Environment

`compare.py` runs both implementations itself on the board: the pinned original
helper (loaded from Git history) and this sample's
task from `samples/vision/lprnet/runtime/python`. It requires the X5 runtime
(imported lazily after the identity gate) and a prepared `lpr.bin` plus the
`.dat` input. It does not import the SDK on the host and does not download
anything. The source performance record follows.

| Model | Test frames | FPS | Average latency | BPU usage | ION memory |
|---|---:|---:|---:|---:|---:|
| `lpr.bin` | 100 | 266 FPS | 3.75 ms | 9% | 1.11 MB |

<a id="command"></a>
## Evaluation command

`evaluate.py` compares the decoded plate with a reference established from the input pixels; `compare.py` compares numerical results between two implementations. The bundled `test_input.dat`, interpreted as RGB CHW scaled to [-1, 1], shows `渝A999U9`.

```bash
python3 samples/vision/lprnet/evaluator/evaluate.py \
  --target x5 --asset-id x5:lprnet:lpr.bin \
  --model-path samples/vision/lprnet/model/lpr.bin \
  --test-bin samples/vision/lprnet/test_data/test_input.dat \
  --reference-plate 渝A999U9 \
  --reference-source 'Visual transcription of decoded RGB test_input.dat pixels before inference' \
  --output outputs/lprnet-reference.json
```

The JSON records the reference and its source, decoded plate, target, asset ID and model/input SHA-256. Exact plate agreement returns 0; mismatch or a recorded model error returns 1. The output file must be new. For another DAT, supply that input’s own plate reference and source.

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

The source reference is the `lpr.bin` row above (100 frames). Board comparison
procedure: run the two implementations on the same X5 with the bundled
`test_input.dat`; passing requires every check true and `max_abs_diff` 0.0 for
the input tensor, the raw `(1,68,18,1)` logits, and the decoded plate. The
board log prints an HBRT-library/model-build minor-version mismatch warning
when loading `lpr.bin`; the warning does not affect the comparison. The result
is numerical parity for one input, not an accuracy benchmark.

<a id="boundaries"></a>
## Scope

The evaluator runs both sides itself and never substitutes a hand-supplied file
for a real inference. It does not download models, prepare an accuracy dataset,
or measure performance. Recorded runs cover one X5 8GB and one X5
4GB with the bundled input (reference results above); any other
board or input requires its own run, and no license-plate accuracy claim
is made from these comparisons.
