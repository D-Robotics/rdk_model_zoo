# FCOS Detection Python Sample

This sample demonstrates how to run FCOS detection models on RDK X5 with `hbm_runtime`.

## Environment Dependencies

- `RDK OS >= 3.5.0`
- `hbm_runtime` is preinstalled on the board image

## Directory Structure

```text
.
├── main.py
├── fcos_det.py
├── run.sh
├── README.md
└── README_cn.md
```

## Argument Description

| Argument | Description | Default |
| --- | --- | --- |
| `--model-path` | Path to the FCOS `.bin` model | `../../model/fcos_efficientnetb0_detect_512x512_bayese_nv12.bin` |
| `--label-file` | Path to the COCO class names file | `../../../../../datasets/coco/coco_classes.names` |
| `--priority` | Runtime scheduling priority | `0` |
| `--bpu-cores` | BPU core indexes | `0` |
| `--test-img` | Path to the test input image | `../../test_data/bus.jpg` |
| `--img-save-path` | Path to save the result image | `../../test_data/result.jpg` |
| `--resize-type` | Resize strategy: 0 for direct resize, 1 for letterbox | `0` |
| `--classes-num` | Number of detection classes | `80` |
| `--conf-thres` | Confidence threshold | `0.5` |
| `--iou-thres` | IoU threshold for NMS | `0.6` |
| `--strides` | FCOS feature strides | `8,16,32,64,128` |

## Quick Run

```bash
chmod +x run.sh
./run.sh
```

## Manual Run

- Run with default arguments:
  ```bash
  python3 main.py
  ```

- Run with explicit arguments:
  ```bash
  python3 main.py \
    --model-path ../../model/fcos_efficientnetb0_detect_512x512_bayese_nv12.bin \
    --test-img ../../test_data/bus.jpg \
    --img-save-path ../../test_data/result.jpg
  ```

## Interface Description

- `FCOSConfig`: Encapsulates model path and inference parameters.
- `FCOSDetect`: Implements `set_scheduling_params`, `pre_process`, `forward`, `post_process`, and `predict`.
