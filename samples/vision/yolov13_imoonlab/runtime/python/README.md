English | [简体中文](./README_cn.md)

# YOLOv13 iMoonLab Python Runtime

This directory shows how to run the YOLOv13 Detect model with `hbm_runtime` on RDK S100 / S100P.

## Dependencies

```bash
pip3 install numpy==1.26.4 opencv-python==4.11.0.86 scipy==1.15.3
```

`hbm_runtime` is expected to be provided by the board system environment.

## Arguments

| Argument | Description | Default |
|---|---|---|
| `--model-path` | Path to the `.hbm` model | `../../model/s100/yolo13n_detect_nashe_640x640_nv12.hbm` |
| `--priority` | Model priority | `0` |
| `--bpu-cores` | List of BPU core indexes | `0` |
| `--test-img` | Test image path | `../../test_data/kite.jpg` |
| `--label-file` | Class label file path | `../../test_data/coco_classes.names` |
| `--img-save-path` | Output image path | `result.jpg` |
| `--nms-thres` | NMS threshold | `0.45` |
| `--score-thres` | Confidence threshold | `0.25` |

## Quick Run

### One-click script

```bash
cd runtime/python
bash run.sh
```

### Direct main.py entry

```bash
python3 main.py \
  --model-path ../../model/s100/yolo13n_detect_nashe_640x640_nv12.hbm \
  --test-img ../../test_data/kite.jpg \
  --label-file ../../test_data/coco_classes.names \
  --img-save-path result.jpg
```

## Runtime Flow

`main.py` only parses CLI arguments, loads the image and labels, builds `YOLOv13Config`, calls `predict()`, and saves the output image. `yolov13.py` provides `set_scheduling_params(...)`, `pre_process(...)`, `forward(...)`, `post_process(...)`, `predict(...)`, and `__call__(...)`.

## Input and Output Protocol

### Input

- `input[0]`: Y plane
- `input[1]`: UV plane

### Output

- `output[0]`: small stride classification
- `output[1]`: small stride box distribution
- `output[2]`: medium stride classification
- `output[3]`: medium stride box distribution
- `output[4]`: large stride classification
- `output[5]`: large stride box distribution

## License

This directory follows the repository top-level `LICENSE`.
