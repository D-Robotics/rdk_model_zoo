# MODNet evaluator

<a id="dataset"></a>
## Dataset

No source matting benchmark dataset or ground-truth alpha masks are included.
`../test_data/person.jpg` and `../test_data/bg.jpg` are one
inference/composite fixture pair, not an accuracy dataset.

<a id="environment"></a>
## Environment

`compare.py` runs both implementations itself on the board: the fixed source
wrapper from `platforms/x5/samples/vision/modnet/runtime/python` and the unified
task from `samples/vision/modnet/runtime/python`. It needs the X5 runtime
(imported lazily after the identity gate), the manually prepared
`modnet_512x512_rgb.bin` and a BGR image. It does not import the SDK on the host
and does not download anything. The complete source performance table and
conditions are preserved below as historical, not-run data.

| Model | Size | Input format | Latency (ms) | FPS |
|---|---|---|---:|---:|
| MODNet | 512x512 | Float32 NCHW RGB | 89.88 | 11.12 |
| MODNet (2 threads) | 512x512 | Float32 NCHW RGB | 130.49 | 15.27 |

Conditions: RDK X5, CPU 8xA55@1.8G, BPU 1xBayes-e@1G (10TOPS INT8);
single-thread latency used one frame, one thread, and one BPU core, while
multi-thread FPS used two concurrent threads.

<a id="command"></a>
## Evaluation command

From the repository root, on a board with the manual artifact prepared:

```bash
python3 samples/vision/modnet/evaluator/compare.py \
  --target x5 \
  --output-dir /tmp/modnet-compare-$(date -u +%Y%m%dT%H%M%SZ)
```

Use `--asset-id x5:modnet:modnet_512x512_rgb.bin --model-path <file>` to point at
the manual artifact and `--test-img <file>` for another image. The output
directory must not already exist. The ref-size is fixed at 512, matching the
deployment metadata.

<a id="metrics"></a>
## Metrics

The evaluator records, for both sides, the prepared NCHW input tensor, the raw
float32 matte and the final uint8 matte in original image geometry, and reports
per-tensor shape/dtype/finite checks with `max_abs_diff`. Inputs and the final
matte must be exactly equal; the raw matte uses `atol=1e-5`. It returns `0` only
when every check passes, `1` on a difference, and `2` when the run failed. No
dataset-level SAD/MAD/IoU is reported because the source supplies no ground
truth.

<a id="outputs"></a>
## Outputs

`comparison.json` binds the run to `target`, `asset_id`, model/image/code SHA-256
digests, observed runtime metadata for both sides, `argv`, `cwd`,
`started_utc`/`finished_utc` and `return_code`, and lists every saved array file
with its own digest. Each recorded input, raw matte and result matte is written to
its own `.npy` file. A failed run still writes the manifest with `error`,
`return_code: 2` and `passed: false`. A composite image is a separate visual
artifact and is never used as the matte comparison tensor.

<a id="reference-results"></a>
## Reference results

Historical source reference is the table above. This migration ran no board
comparison, so source/unified board and accuracy evaluation are `not-run`; the
sample remains `closed=no` and the host fixtures prove only the evidence schema.

<a id="boundaries"></a>
## Boundaries

The evaluator runs both sides itself and never compares hand-supplied matte files
in place of a real inference. It is not a portrait dataset evaluator, does not
invent quality labels, does not download the manual artifact, and does not convert
a host fixture into board verification.
