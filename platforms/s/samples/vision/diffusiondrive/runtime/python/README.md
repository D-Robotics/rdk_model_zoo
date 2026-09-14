English | [简体中文](./README_cn.md)

# DiffusionDrive Python Runtime

This Python sample uses `hbm_runtime` to run one four-input DiffusionDrive inference and saves decoded NPZ tensors plus a PNG visualization.

## Dependencies

The S100P/S600 board image must provide Python 3, `hbm_runtime`, NumPy, and OpenCV:

```bash
python3 -c "import hbm_runtime, numpy, cv2"
```

## Directory Structure

```text
.
|-- diffusiondrive.py           # DiffusionDriveConfig, model wrapper, postprocess, renderer
|-- main.py                     # argparse entry point
|-- run.sh                      # One-command model check and inference
|-- run_all_cases.sh            # Run every packaged test_data/case_* input
|-- README.md                   # English instructions
`-- README_cn.md                # Chinese instructions
```

## Parameters

| Parameter | Description | Default |
| --- | --- | --- |
| `--platform` | `auto`, `s100p`, or `s600` | `auto` |
| `--model-path` | Platform-specific HBM path | Auto-selected from `../../model/<platform>/` |
| `--input-npz` | Four float32 input tensors | `../../test_data/reference_inputs.npz` |
| `--output-npz` | Decoded tensor output | `./diffusiondrive_outputs.npz` |
| `--img-save-path`, `--output-image` | Visualization output; both names are accepted | `./diffusiondrive_result.png` |
| `--agent-score-thres` | Agent sigmoid threshold | `0.5` |
| `--priority` | Runtime scheduling priority | `0` |
| `--bpu-cores` | BPU core indexes | `0` |

## Quick Run

Default input and output paths:

```bash
bash run.sh
```

Explicit paths:

```bash
python3 main.py \
  --platform s100p \
  --model-path ../../model/s100p/diffusiondrive_r34_256x1024_s100p.hbm \
  --input-npz ../../test_data/reference_inputs.npz \
  --output-npz ../../test_data/my_outputs.npz \
  --img-save-path ../../test_data/my_result.png
```

The input NPZ must contain `camera`, `lidar`, `status`, and `noise`. Quantization scales are read from the selected HBM model instead of being hard-coded. `run.sh` detects the board, downloads the matching model if absent, and passes the platform to `main.py`.

Run all packaged demo cases:

```bash
bash run_all_cases.sh
```

Outputs are written under `runtime/python/results/case_*/`. Pass another directory as the first argument to change the destination.

## Code Documentation

The public integration surface is `DiffusionDriveConfig` plus the `DiffusionDrive.pre_process`, `forward`, `post_process`, and `predict` methods. See the repository [source documentation guide](../../../../../docs/source_reference/README.md).

## Notes

- Load the model once and reuse the instance in production; process startup is not BPU inference latency.
- `noise` is an explicit model input. Keep it fixed when reproducible output is required.
- The sample visualizes only the model tensors. Full NAVSIM maps, annotations, GIFs, and PDM scoring require the NAVSIM dataset on an x86 host.
