# YOLOWorld Python Sample

This sample demonstrates open-vocabulary object detection with YOLOWorld and offline vocabulary embeddings.

## Directory Structure

```text
.
|-- main.py
|-- yoloworld_det.py
|-- README.md
|-- README_cn.md
`-- run.sh
```

## Parameter Description

| Argument | Description | Default |
| --- | --- | --- |
| `--model-path` | Path to the quantized `.bin` model file. | `../../model/yolo_world.bin` |
| `--vocab-file` | Path to offline vocabulary embeddings JSON. | `../../test_data/offline_vocabulary_embeddings.json` |
| `--test-img` | Path to the test input image. | `../../test_data/dog.jpeg` |
| `--prompts` | Comma-separated prompt words used for detection. | `dog` |
| `--img-save-path` | Path to save the output visualization image. | `../../test_data/inference.png` |
| `--priority` | Model priority in the range `0~255`. | `0` |
| `--bpu-cores` | BPU core indexes used for inference. | `0` |
| `--score-thres` | Score threshold used to filter predictions. | `0.05` |
| `--nms-thres` | NMS threshold used to suppress overlapping boxes. | `0.45` |

## Quick Run

```bash
chmod +x run.sh
./run.sh
```

## Manual Run

Prepare the default model before running `main.py` directly. You can either run
`./run.sh` once, or run `../../model/download.sh` from this directory to download
`../../model/yolo_world.bin`.

- Run with default parameters:

```bash
python3 main.py
```

- Run with explicit parameters:

```bash
python3 main.py \
    --model-path ../../model/yolo_world.bin \
    --vocab-file ../../test_data/offline_vocabulary_embeddings.json \
    --test-img ../../test_data/dog.jpeg \
    --prompts dog \
    --img-save-path ../../test_data/inference.png
```

## Interface Description

- **YOLOWorldConfig**: Encapsulates model path, vocabulary path, and thresholds.
- **YOLOWorldDetect**: Implements image/text preprocessing, BPU execution, and fixed YOLOWorld output decoding.
