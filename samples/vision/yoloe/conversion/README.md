English | [简体中文](./README_cn.md)

# YOLOE Conversion

This directory describes YOLOE-11s/m/l Seg Prompt-Free conversion for RDK X5. All three sizes share the exporter and use their own weights and quantization configurations.

---

## Conversion Assets

The following files are kept as the reference conversion resources:

- `onnx_export/export_yoloe11seg_bpu.py` — ONNX export script with BPU-compatible patches
- `ptq_yamls/yoloe_11s_seg_pf_bayese_640x640_nv12.yaml` — PTQ quantization configuration

---

## Output Protocol

The YOLOE-11 Seg Prompt-Free model on X5 uses the following output protocol:

- ONNX input: `1x3x640x640`, float32 RGB/NCHW; board input: 640x640 UINT8 packed NV12.
- Output 0: classification head (stride 8)
- Output 1: bounding-box head (stride 8, DFL 16 bins)
- Output 2: mask-coefficient head (stride 8)
- Output 3: classification head (stride 16)
- Output 4: bounding-box head (stride 16, DFL 16 bins)
- Output 5: mask-coefficient head (stride 16)
- Output 6: classification head (stride 32)
- Output 7: bounding-box head (stride 32, DFL 16 bins)
- Output 8: mask-coefficient head (stride 32)
- Output 9: prototype tensor

The Python runtime in this sample uses this contract and decodes it with DFL box regression and prototype mask generation.

---

## Conversion Steps

### 1. Environment Preparation

Clone the official YOLOE repository and install dependencies:

```bash
git clone https://github.com/um-assn/yoloe.git
cd yoloe
pip install -r requirements.txt
pip install ultralytics
```

Download the matching pretrained weights:

```bash
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11s-seg-pf.pt
```

Replace `11s` with `11m` or `11l` to download the other sizes. Export ONNX in the Python/Conda environment and compile in the X5 OE Mapper environment.

### 2. Export to ONNX

The export script (`export_yoloe11seg_bpu.py`) applies two critical patches for BPU compatibility:

- Replaces Linear vocabulary layers with equivalent 1x1 Conv2d layers
- Patches the detection head forward method to output 10 tensors in NHWC layout

It also saves the 4585-class vocabulary to a `.names` file.

Run from this sample's `conversion` directory:

```bash
python3 onnx_export/export_yoloe11seg_bpu.py --weights /path/to/yoloe-11s-seg-pf.pt --imgsz 640
```

This writes `yoloe-11s-seg-pf.onnx`, `yoloe-11s-seg-pf.names` and `yoloe-11s-seg-pf.export.json` beside the checkpoint. Change the weight path for m/l.

### 3. Prepare Calibration Data

Prepare 640x640 RGB/NCHW float32 calibration arrays in the range 0..255. Match the runtime letterbox and gray padding value 127; normalization is applied by `scale_value` in the YAML.

```bash
# Place calibration images in ./calibration_data_rgb_f32_640/
```

### 4. PTQ Conversion

Set `onnx_model`, `cal_data_dir` and `working_dir` to the actual paths in the YAML, then run `hb_mapper`:

```bash
hb_mapper makertbin --model-type onnx --config ptq_yamls/yoloe_11s_seg_pf_bayese_640x640_nv12.yaml
```

For m/l, copy the reference YAML and update the model path, working directory and `output_model_file_prefix` to `yoloe_11m_seg_pf_bayese_640x640_nv12` or `yoloe_11l_seg_pf_bayese_640x640_nv12`. For 11l, also add `/model.10/m/m.1/attn/Softmax` to `node_info` with the same settings as the first Softmax: `'ON': BPU`, `InputType: int16`, `OutputType: int16`. Check node names against the exported graph.

### 5. Validation

Visualize the compiled model:

```bash
hb_perf /path/to/yoloe_11s_seg_pf_bayese_640x640_nv12.bin
```

Check model inputs and outputs:

```bash
hrt_model_exec model_info --model_file /path/to/yoloe_11s_seg_pf_bayese_640x640_nv12.bin
```

---

## Notes

- This document focuses on the conversion steps required for YOLOE on RDK X5.
- The runtime uses converted `.bin` models through `hbm_runtime`.
- The Softmax node in the attention layer is configured to run on BPU with int16 input/output.
- Benchmark figures and additional reference assets are available under `test_data/` and related repository resources.
