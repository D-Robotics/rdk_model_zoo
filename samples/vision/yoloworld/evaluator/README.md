English | [简体中文](README_cn.md)

# YOLOWorld evaluator

<a id="dataset"></a>

<a id="directory"></a>
## Directory structure

```text
evaluator/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── compare.py  # Python script
└── source_reference.py  # Python script
```

<a id="environment"></a>
## Dataset and environment

The evaluator compares the pinned original X5 implementation (loaded from the
recorded Git history) with this sample's implementation on one
**real X5 execution target**. It uses the repository
fixture `test_data/dog.jpeg` and an explicit comma-separated prompt list. Host
Python tests use injected fake runtimes. The evaluator requires the board
identity gate, `hbm_runtime`, OpenCV,
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
`--test-img`, `--vocab-file`, `--prompts`, `--score-thres`, `--nms-thres`,
`--priority` and `--bpu-cores` select the same inputs and scheduling for both
implementations. The command first gates the actual target, then runs both
implementations' pre/infer/post stages with the same image, vocabulary and
model, capturing the prepared inputs, the raw score/box tensors and the final
detections from each side. It returns 0 only when every check passes, 1 when the
two sides differ, and 2 when the run failed.

<a id="metrics"></a>

<a id="outputs"></a>
## Metrics and outputs

This evaluator compares raw tensors and decoded results on the same input.
It writes `comparison.json` plus one `.npy` per recorded input, raw output and
result array. The manifest binds the run to `target`, `asset_id`, model, image,
vocabulary and code SHA-256 digests, the observed runtime metadata for both
sides, the prompt list, thresholds, `argv`, `cwd`, `started_utc`/`finished_utc`
and `return_code`, and lists every saved array file with its own digest.
Inputs and class IDs must be exactly equal; raw tensors use `atol=1e-5`, boxes
`1e-4` and scores `1e-5`. A failed run still writes the manifest with `error`,
`return_code: 2` and `passed: false`. No output is overwritten.

<a id="reference-results"></a>

<a id="boundaries"></a>
## Reference results and conditions

| Reference record | Input/protocol | Value | Source |
| --- | --- | --- | --- |
| X5 source sample protocol | 640 image; 32 x 512 text; 8400 score/box rows | no published latency or mAP table | source evaluator README |
| Board parity check | `dog` prompt, `test_data/dog.jpeg`, score/NMS 0.05/0.45 | run the comparator on a prepared X5; passing requires all checks true and `max_abs_diff` 0.0 | this evaluator |

The published protocol facts are: `yolo_world.bin`, 640 image input, 32 text
slots of width 512, 8400 score rows, and score/NMS defaults 0.05/0.45. In the
recorded board runs the log prints an HBRT-library/model-build minor-version
mismatch warning, preserved verbatim in the evidence. Any other prompt, image,
or board needs its own run of this command. The offline vocabulary is a
required model companion, not a classification-label file.
