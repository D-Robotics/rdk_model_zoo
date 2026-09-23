# YOLOWorld evaluator

<a id="dataset"></a>
<a id="environment"></a>
## Dataset and environment

The evaluator compares the fixed X5 source implementation and the unified
implementation on one **real X5 execution target**. It uses the repository
fixture `test_data/dog.jpeg` and an explicit comma-separated prompt list. Host
Python tests use injected fake runtimes; they are not board or accuracy
results. The evaluator requires the board identity gate, `hbm_runtime`, OpenCV,
NumPy, and the exact `yolo_world.bin` asset. No dependency installation or
model download is performed by this command.

<a id="command"></a>
## Command

From the repository root, prepare the model with the explicit model command,
then run into a new directory (the evaluator refuses an existing directory):

```bash
python3 samples/vision/yoloworld/evaluator/compare.py --target x5 --output-dir /absolute/new-yoloworld-evidence --asset-id x5:yoloworld:yolo_world.bin
```

Use `--model-path /absolute/yolo_world.bin` only with the exact `--asset-id`.
`--test-img`, `--vocab-file`, `--prompts`, `--score-thres`, and `--nms-thres`
select the same inputs for both implementations. The command first gates the
actual target, then runs source and unified pre/forward/post stages with the
same image, vocabulary and model. It returns 0 only for exact raw tensors,
exact class IDs, and zero-tolerance boxes/scores; a failed comparison keeps
its output directory and returns 1. Execution/setup errors return 2.

<a id="metrics"></a>
<a id="outputs"></a>
## Metrics and outputs

This is a tensor parity evaluator, not a dataset mAP or performance benchmark.
It stores `legacy_raw_*.npy`, `unified_raw_*.npy`, both result arrays,
`metadata.json`, and SHA-256 values for the model, source code, input image and
vocabulary. Metadata records target, asset identity, prompt list, thresholds
and the decision. No output is overwritten.

<a id="reference-results"></a>
<a id="boundaries"></a>
## Historical reference and boundaries

| Source record | Input/protocol | Historical performance | Status |
| --- | --- | --- | --- |
| fixed X5 YOLOWorld sample | 640 image; 32 x 512 text; 8400 score/box rows | no published latency or mAP table | preserved fact; no new measurement |

The fixed source evaluator README publishes no benchmark table. Therefore the
historical record is explicitly: `yolo_world.bin`, 640 image input, 32 text
slots of width 512, 8400 score rows, and score/NMS defaults 0.05/0.45; no
published latency or mAP number is claimed. Board execution and model
availability remain environment-dependent and are `not-run` until this command
creates evidence. The offline vocabulary is a required model companion, not a
classification-label file.
