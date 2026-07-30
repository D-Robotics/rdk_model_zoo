[English](./README.md) | [简体中文](./README_cn.md)

# Model Evaluation

This directory provides reusable numerical and SUN RGB-D evaluation tools. Generated datasets, predictions, reports, and visualizations must be written outside the sample directory.

## RDK X5 Performance Data

Compilation used OpenExplorer v1.2.8 / Mapper 1.24.3 with 768x768 input, max percentile `0.9999`, O3 latency optimization, and an int16 tail-convolution output. HRT numbers cover model execution only.

| Platform | Model | Input Size | 1-thread Latency | 1-thread FPS | 2-thread Total FPS |
| --- | --- | --- | ---: | ---: | ---: |
| RDK X5 | YOLO26n Depth | 768x768 | 23.194 ms | 43.085 | 45.682 |
| RDK X5 | YOLO26s Depth | 768x768 | 36.168 ms | 27.637 | 28.615 |
| RDK X5 | YOLO26m Depth | 768x768 | 60.783 ms | 16.449 | 16.751 |
| RDK X5 | YOLO26l Depth | 768x768 | 75.336 ms | 13.272 | 13.470 |
| RDK X5 | YOLO26x Depth | 768x768 | 161.022 ms | 6.210 | 6.253 |

Board accuracy data is not published for this sample yet. Use the tools below to generate accuracy reports when validated predictions and ground truth are available.

## Prepare SUN RGB-D Inputs

Prepare deterministic deployment-letterbox and Ultralytics-validator inputs:

```bash
python3 prepare_sunrgbd.py \
  --source-root SUNRGBD_ROOT \
  --source-manifest SOURCE_MANIFEST.json \
  --output OUTPUT_DIR \
  --size 768
```

The output contains RGB CHW uint8 tensors, depth arrays, and a manifest describing preprocessing geometry.

## Single-Image Numerical Comparison

Compare a restored X5 depth map with an official floating-point result:

```bash
python3 eval_numeric.py \
  --image ../test_data/bus.jpg \
  --official OFFICIAL_DEPTH.npy \
  --x5 X5_DEPTH.npy \
  --output REPORT_DIR
```

The command writes numerical metrics, a JSON report, and comparison visualizations.

## SUN RGB-D Evaluation

Evaluate model outputs against a prepared SUN RGB-D manifest:

```bash
python3 eval_sunrgbd.py --help
```

The evaluator supports deployment-letterbox and Ultralytics-validator preprocessing protocols and reports standard monocular-depth metrics. Use the command help to select output artifacts, alignment mode, and protocol.

## Notes

- Evaluation inputs and generated outputs are intentionally excluded from the repository.
- Relative-depth predictions require alignment before comparison with metric ground truth.
- Keep the selected preprocessing protocol consistent between input preparation and evaluation.
