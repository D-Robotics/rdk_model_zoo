English | [简体中文](./README_cn.md)

# YOLOE-26 PF Conversion

This directory exports and compiles the five YOLOE-26 prompt-free segmentation
models (`n`, `s`, `m`, `l`, and `x`) for RDK S100 and S100P. The supported target
marches are `nash-e` (S100) and `nash-m` (S100P). S600 is not supported here.

## Contents

```text
conversion/
├── mapper.py
├── onnx_export/
│   └── export_yoloe26_seg_pf.py
├── README.md
├── README_cn.md
└── requirements.txt
```

The exporter accepts locally obtained `yoloe-26<SIZE>-seg-pf.pt` checkpoints.
Weights are deliberately not downloaded by this sample. The export writes one
static ONNX model, checkpoint-ordered `.names`, and metadata JSON per size.
The metadata records the ONNX hash and the fixed ten-output contract:

```text
[cls_8, box_8, mc_8,
 cls_16, box_16, mc_16,
 cls_32, box_32, mc_32, proto]
```

Input is one `1x3x640x640` RGB tensor. The runtime converts the corresponding
letterboxed BGR image to NV12. The letterbox uses padding value `114` and the
model input is scaled by `1/255`.

## Export ONNX

Install the Python dependencies in the environment used for export:

```bash
python3 -m pip install -r conversion/requirements.txt
```

Run one size at a time and use a new output directory for every retry:

```bash
python3 conversion/onnx_export/export_yoloe26_seg_pf.py \
  --weights /path/to/yoloe-26n-seg-pf.pt \
  --size n \
  --output-dir /path/to/exports/n \
  --test-image /path/to/test_data/office_desk.jpg \
  --threads 2
```

Repeat for `s`, `m`, `l`, and `x`. A five-model export tree can then be passed
to `mapper.py` when it contains `<size>/yoloe_26<size>_seg_pf.{onnx,json,names}`.

## Prepare and Compile HBM

`mapper.py` validates every source ONNX and metadata pair, creates one
deterministic 100-image calibration set, writes one reviewable YAML per model,
and only runs the compiler when `--compile` is explicitly supplied. It refuses
to reuse an existing output directory.

Omit `--compile` to create a config-only run for inspection:

```bash
python3 conversion/mapper.py \
  --onnx /path/to/exports \
  --metadata /path/to/exports \
  --cal-images /path/to/cal_images \
  --march nash-e \
  --output-dir /path/to/build/yoloe26_s100_preview \
  --sample-count 100
```

For an actual compile, repeat the command with a new output directory and add
`--compile`; preparation and compilation then run in the same invocation:

```bash
python3 conversion/mapper.py \
  --onnx /path/to/exports \
  --metadata /path/to/exports \
  --cal-images /path/to/cal_images \
  --march nash-e \
  --output-dir /path/to/build/yoloe26_s100_compiled \
  --sample-count 100 \
  --compile
```

Use a different new output directory and `--march nash-m` for S100P. The five
compiled models and their matching metadata/labels are written under
`<output-dir>/<size>/`; `manifest.json` and `compile_results.json` summarize the
run. The HBM names are:

```text
yoloe_26<SIZE>_seg_pf_nashe_640x640_nv12.hbm  # nash-e / S100
yoloe_26<SIZE>_seg_pf_nashm_640x640_nv12.hbm  # nash-m / S100P
```

The generated configurations use OE 3.7.0-compatible KL INT8 PTQ,
`remove_node_type: Quantize;Dequantize`, `input_no_padding: true`,
`output_no_padding: false`, `jobs: 4`, and optimization level `O2`. The output
stride may therefore be padded; use the valid tensor shape and runtime-provided
strides when reading HBM outputs.

## Docker

Run the commands inside the validated OE 3.7.0 CPU image for the S100/S100P
toolchain. Keep the export and calibration directories inside the repository
mount, or mount them separately at paths visible from `/workspace`:

```bash
REPO_DIR=/path/to/rdk_model_zoo
docker run --rm -it --shm-size=2g \
  -v "$REPO_DIR":/workspace \
  -w /workspace/samples/vision/yoloe26_seg \
  --entrypoint /bin/bash \
  registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0
```

Inside the container, use `/workspace` paths with the same `mapper.py` command.
The compiler toolchain is expected to be present in the image; this workflow
does not install packages or alter board settings.

## Validation

The exporter performs upstream static-PF equivalence checks and ONNX Runtime
output checks for every size. Compilation is independent per size and writes
`compile.log` plus a per-size result. A successful local compile means
`compiled_not_board_validated`; it does not claim board accuracy or benchmark
results. Copy each HBM together with its same-directory JSON and `.names` file.
