English | [简体中文](README_cn.md)

# Evaluator — DINOv2 ViT-S/14

Recorded performance and accuracy references for the exported artifacts, with the board commands that reproduce their settings.

<a id="dataset"></a>
## Dataset

The runtime smoke path uses the repository fixtures `samples/vision/dinov2/test_data/dog.jpg` and `bus.jpg`. The PTQ report used 50 diverse real calibration images; it also states that identical values were reproduced with an independent export and a different 50-image calibration set. No evaluator dataset preparation script or fixed dataset archive is supplied. Calibration preparation is implemented by `conversion/mapper.py` and documented in [`../conversion/README.md`](../conversion/README.md).

```text
# cwd: repository root
samples/vision/dinov2/test_data/dog.jpg
samples/vision/dinov2/test_data/bus.jpg
# board benchmark inputs: same preprocessed float32 tensors as the runtime contract
```

<a id="directory"></a>
## Directory structure

```text
evaluator/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
└── evaluate.py  # Basic image relation checks
```

<a id="environment"></a>
## Environment

- Board records: RDK S100/Nash-E, S100P/Nash-M, and S600/Nash-P with `hrt_model_exec` or `hbm_runtime`.
- PTQ report: OE 3.7.0, hmct 2.6.5 / hbdk 4.7.5 on Nash-E.
- Runtime dependencies: board-image `hbm_runtime`; host utilities use Python 3.10+ with NumPy and OpenCV.
- Board image, firmware, and runtime versions are not pinned.

<a id="command"></a>
## Evaluation Command

`evaluate.py` checks CLS features and mean patch features separately. The anchor is a puppy, the independently different positive image contains two dogs, and the negative image contains a bus. Choose image relationships before inference.

```bash
# Repository root on S100; use a new output directory.
python3 samples/vision/dinov2/evaluator/evaluate.py \
  --target s100 \
  --asset-id s:dinov2:nash-e/dinov2_vits14_224_int16_nashe.hbm \
  --model-path samples/vision/dinov2/model/nash-e/dinov2_vits14_224_int16_nashe.hbm \
  --anchor samples/vision/dinov2/test_data/dog.jpg \
  --positive samples/vision/mobile_sam/test_data/dogs.jpg \
  --negative samples/vision/dinov2/test_data/bus.jpg \
  --output-dir outputs/dinov2-relations
```

Each role requires repeat cosine ≥ 0.99999, mild-transform cosine ≥ 0.95, and positive cosine minus negative cosine > 0.05. The mild transform is `clip(BGR * 0.9 + 5, 0, 255).astype(uint8)`; patch tokens are averaged before cosine comparison. `result.json` records criteria, per-role checks, metadata and input/model hashes; per-role `.npz` files contain anchor, repeat, mild, positive and negative features. Exit 0 means every role passes; exit 1 records a failed relation or model/numeric error.

The performance commands below reproduce the recorded thread/core settings when run on the matching board with its target artifact.

```bash
# cwd: samples/vision/dinov2/evaluator on the target board; artifact prepared under ../model/
# S100 / Nash-E
hrt_model_exec perf --model_file ../model/nash-e/dinov2_vits14_224_int16_nashe.hbm --thread_num 1
hrt_model_exec perf --model_file ../model/nash-e/dinov2_vits14_224_int16_nashe.hbm --thread_num 2

# S100P / Nash-M
hrt_model_exec perf --model_file ../model/nash-m/dinov2_vits14_224_int16_nashm.hbm --thread_num 1
hrt_model_exec perf --model_file ../model/nash-m/dinov2_vits14_224_int16_nashm.hbm --thread_num 2

# S600 / Nash-P
hrt_model_exec perf --model_file ../model/nash-p/dinov2_vits14_224_int16_nashp.hbm --thread_num 1
hrt_model_exec perf --model_file ../model/nash-p/dinov2_vits14_224_int16_nashp.hbm --thread_num 12 --core_id 1,2,3,4
# expect: BPU latency/throughput over 200 frames, after locking the performance governor
```

For accuracy, run the exported float ONNX with ONNXRuntime on the same preprocessed inputs, run the HBM with `hbm_runtime.HB_HBMRuntime(...).run` on the board, then compute cosine similarity separately for `cls_feat` and `patch_feat`. The source runtime CLI's two-image path is documented in [`../runtime/python/README.md`](../runtime/python/README.md).

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `thread_num` | int | `1` in the baseline record | `hrt_model_exec perf` worker count. |
| `core_id` | CSV ints | unset, except S600 12-thread record | BPU cores used for the S600 high-concurrency record: `1,2,3,4`. |
| `frames` | int | `200` in source records | Performance measurement scope. |
| `input` | tensor | `(1,3,224,224)` F32 | RGB normalized tensor produced by the runtime preprocessing contract. |

<a id="metrics"></a>
## Metrics

| Metric | Definition | Conditions |
| --- | --- | --- |
| BPU latency | Pure BPU forward latency for one model invocation. | 200 frames, performance governor locked; board and thread/core settings in the tables. CPU preprocessing is additional. |
| BPU throughput | Frames per second reported by `hrt_model_exec perf`. | Same 200-frame run; concurrency is stated per row. |
| Calibrated cosine | Cosine between calibration/toolchain output and float reference for each output. | PTQ report, Nash-E, featuremap float32 input, all-int16, default KL calibration. |
| Quantized cosine | Cosine between quantized output and float ONNX output for each output. | PTQ report, Nash-E; `cls_feat` and `patch_feat` measured separately. |
| Board cosine range | Min/max cosine range over board executions against float ONNX references. | Same input preprocessing; S100, S100P, S600 separately; source board record. |

The preprocessing is OpenCV BGR→RGB, bicubic short-side resize to 256, center crop 224, `/255`, ImageNet mean/std, and contiguous float32 NCHW. Outputs are compared before any softmax or L2 operation.

<a id="outputs"></a>
## Outputs

The evaluation writes to the `--output-dir` given in the command: `result.json` (schema `rdk-model-zoo/embedding-relations/v1`) records the target, asset ID, model/input SHA-256 hashes, the mild-transform definition, per-role metadata, criteria, cosine values and individual checks; alongside it, `cls_feat.npz` and `patch_mean.npz` each contain the five feature arrays (`anchor`, `repeat`, `mild`, `positive`, `negative`) captured for that role. Exit 0 means every role passed; exit 1 records a failed relation or a model/numeric error. ONNX accuracy comparison requires separate cosine values for cls_feat and patch_feat and has no bundled implementation; use the reproduction commands above on the board.

<a id="reference-results"></a>
## Reference Results

The following tables list every recorded row and column. Source: S platform evaluator README and the S release benchmark records.

### Performance record

| Device | Model | Input Size | BPU Task Latency / BPU Throughput |
|---|---|---|---|
| RDK S100 | dinov2_vits14_224_int16 | 1x3x224x224 | 3.73 ms / 267.44 FPS (1 thread) <br> 288.26 FPS (2 threads) |
| RDK S100P | dinov2_vits14_224_int16 | 1x3x224x224 | 3.02 ms / 329.53 FPS (1 thread) <br> 357.63 FPS (2 threads) |
| RDK S600 | dinov2_vits14_224_int16 | 1x3x224x224 | 2.25 ms / 441.64 FPS (1 thread) <br> 1898.42 FPS (12 threads, `--core_id 1,2,3,4`) |

Model parameters: 22.06 M. Latency is pure BPU forward; CPU preprocessing is additional.

### PTQ per-output cosine — Nash-E only

This quantization-quality table is only for the Nash-E toolchain report; it must not be generalized to S100P or S600.

| Output | Calibrated Cosine | Quantized Cosine |
|---|---|---|
| cls_feat | 0.9990 | 0.9989 |
| patch_feat | 0.9985 | 0.9983 |

The record reports identical values from an independent export and a different 50-image calibration set.

### Board-executed cosine vs float ONNX

| Device | cls_feat | patch_feat |
|---|---|---|
| RDK S100 | 0.9987 - 0.9989 | 0.9977 - 0.9986 |
| RDK S100P | 0.9987 - 0.9989 | 0.9977 - 0.9986 |
| RDK S600 | 0.9988 - 0.9989 | 0.9975 - 0.9986 |

<a id="boundaries"></a>
## Scope

- `evaluate.py` measures the declared image relation; reference performance and float-ONNX comparison use the commands above.
- All reference values are recorded benchmarks; run the performance commands above for current-board measurements.
- The PTQ quantization-quality table is explicitly Nash-E only. Board cosine ranges are separately attributed to each target.
- DINOv2 is documented as a vision feature encoder. Text encoding, classification labels, retrieval datasets, and C++ evaluation are outside this sample.

## License

Evaluator documentation follows the repository [LICENSE](../../../../LICENSE), Apache-2.0.
