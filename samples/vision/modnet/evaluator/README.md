# MODNet evaluator

<a id="dataset"></a>
## Dataset

No source matting benchmark dataset or ground-truth alpha masks are included. `../test_data/person.jpg` and `../test_data/bg.jpg` are one inference/composite fixture pair, not an accuracy dataset.

<a id="environment"></a>
## Environment

`compare.py` runs on a host with Python, NumPy, and OpenCV and does not load `hbm_runtime`. Generating legacy/unified matte files requires the same external model and X5 board runtime on both sides. The complete source performance table and conditions are preserved below as historical, not-run data.

| Model | Size | Input format | Latency (ms) | FPS |
|---|---|---|---:|---:|
| MODNet | 512x512 | Float32 NCHW RGB | 89.88 | 11.12 |
| MODNet (2 threads) | 512x512 | Float32 NCHW RGB | 130.49 | 15.27 |

Conditions: RDK X5, CPU 8xA55@1.8G, BPU 1xBayes-e@1G (10TOPS INT8); single-thread latency used one frame, one thread, and one BPU core, while multi-thread FPS used two concurrent threads.

<a id="command"></a>
## Evaluation command

Save complete uint8 grayscale matte outputs from the source and unified board runs to unique paths, then run from the repository root:

```bash
python3 samples/vision/modnet/evaluator/compare.py \
  --legacy-matte /tmp/modnet-run-legacy/matte.png \
  --unified-matte /tmp/modnet-run-unified/matte.png \
  --atol 0 \
  --output /tmp/modnet-run-unified/compare.json
```

The command requires equal geometry and returns `0` only when the maximum uint8 absolute difference is within `--atol`; it returns `2` otherwise. It does not download, infer, or claim board validation.

<a id="metrics"></a>
## Metrics

The report contains matte shape, dtype, maximum absolute difference, mean absolute difference, and exact equality. No dataset-level SAD/MAD/IoU score is reported because the source supplies no ground truth. Performance values above retain their original source conditions and are historical.

<a id="outputs"></a>
## Outputs

`compare.py` prints and optionally saves JSON. Keep both complete input mattes and the report in a unique run directory. A composite image is a separate visual artifact and is not used as the matte comparison tensor.

<a id="reference-results"></a>
## Reference results

Historical source reference is the table above. Current host comparison fixtures pass; source/unified board and accuracy evaluation are `not-run`.

<a id="boundaries"></a>
## Boundaries

This evaluator is a self-contained saved-matte comparator. It is not a portrait dataset evaluator, does not invent quality labels, and does not convert a host fixture into board verification.
