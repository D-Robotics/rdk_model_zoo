# PP-LiteSeg-STDC1 Model Conversion

This directory provides a complete conversion workflow from PaddleSeg / ONNX to an RDK X5 `.bin` model.

## Environment

Two environments are required:

1. **ONNX export environment** — Python environment with PaddlePaddle, PaddleSeg, `paddle2onnx`, `onnx`, and `onnxsim`.
2. **OpenExplorer (OE) environment** — D-Robotics OpenExplorer v1.2.8 Docker, providing `hb_mapper`, `hb_perf`, and related tools.

All `hb_mapper` and `hb_perf` commands must run inside the OpenExplorer Docker container.

### 1.1 ONNX Export Environment Setup

Install dependencies in your local Python environment (Python 3.8–3.10 recommended):

```bash
pip install paddlepaddle==3.0.0 paddle2onnx onnx onnxsim

# Install PaddleSeg from Gitee mirror (GitHub may be blocked in some regions)
git clone --depth=1 https://gitee.com/paddlepaddle/PaddleSeg.git
cd PaddleSeg && pip install -e .
```

### 1.2 OpenExplorer Docker Setup

**Offline Docker image package:**

```bash
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
docker load -i docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
```

**Optional — OE SDK and documentation:**

```bash
# Full SDK package
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/horizon_x5_open_explorer_v1.2.8-py310_20240926.tar.gz

# Documentation (Chinese)
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/x5_doc-v1.2.8-py310-cn.zip
# Documentation (English)
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/x5_doc-v1.2.8-py310-en.zip
```

### 1.3 Start the OE Docker Container

Mount the repo root into the container and launch an interactive shell:

```bash
docker run -it --rm \
  -v $(pwd):/open_explorer \
  -w /open_explorer \
  openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 \
  /bin/bash
```

Verify the toolchain inside the container:

```bash
hb_mapper --version
hb_perf --version
```

## 1. Export ONNX

Run the export script from this directory:

```bash
cd samples/vision/pp_liteseg/conversion
bash onnx_export/export_pp_liteseg_stdc1_onnx.sh
```

The default script exports the PaddleSeg config:

```text
configs/pp_liteseg/pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k.yml
```

Expected ONNX output:

```text
conversion/onnx/pp_liteseg_stdc1_cityscapes_1024x512_sim.onnx
```

If you already have an ONNX model, place it at the same path or update `model_parameters.onnx_model` in `ptq_yamls/pp_liteseg_stdc1_cityscapes_1024x512_nv12.yaml`.

## 2. Prepare Calibration Data

Prepare 20 to 50 representative road-scene images. The script exports raw NCHW RGB float32 tensors without normalization. Normalization is handled by the YAML file.

```bash
cd samples/vision/pp_liteseg/conversion
python3 prepare_calibration.py \
  --src /path/to/cityscapes_or_custom_images \
  --out calibration_data_rgb_f32_1024x512 \
  --width 1024 \
  --height 512 \
  --num 50
```

Each output file should be:

```text
1 * 3 * 512 * 1024 * 4 = 6291456 bytes
```

## 3. Check ONNX Operators

Inside the OpenExplorer Docker container:

```bash
cd samples/vision/pp_liteseg/conversion
hb_mapper checker \
  --model-type onnx \
  --march bayes-e \
  --model onnx/pp_liteseg_stdc1_cityscapes_1024x512_sim.onnx
```

Read `hb_mapper_checker.log` and confirm there are no unsupported operators. If unsupported operators appear, simplify the ONNX graph or export without post-processing nodes.

## 4. Build BIN

```bash
cd samples/vision/pp_liteseg/conversion
hb_mapper makertbin \
  --config ptq_yamls/pp_liteseg_stdc1_cityscapes_1024x512_nv12.yaml \
  --model-type onnx
```

Expected output:

```text
conversion/ptq_yamls/pp_liteseg_stdc1_cityscapes_1024x512_nv12_output/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin
```

## 5. One-Command Build

After the ONNX model exists, you can run:

```bash
cd samples/vision/pp_liteseg/conversion
CAL_SRC=/path/to/calibration/images bash build_bin.sh
```

If calibration data is already prepared, omit `CAL_SRC`:

```bash
bash build_bin.sh
```

## 6. Performance Check

```bash
hb_perf ptq_yamls/pp_liteseg_stdc1_cityscapes_1024x512_nv12_output/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin
```

On the board, use:

```bash
hrt_model_exec model_info \
  --model_file pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin

hrt_model_exec perf \
  --model_file pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin \
  --core_id=0 \
  --thread_num=1 \
  --profile_path="."
```

## Runtime Protocol

- Input runtime type: `nv12`
- Input train type: `rgb`
- Input train layout: `NCHW`
- Input size: `1024x512`
- Normalization: `(pixel - mean) * scale`
- Mean: `123.675, 116.28, 103.53`
- Scale: `1/58.395, 1/57.12, 1/57.375`
- Output: segmentation logits, typically decoded with `argmax` on the class axis

## Troubleshooting

- If `checker` reports unsupported resize or argmax nodes, ensure the ONNX graph contains only the neural network and does not include post-processing.
- If cosine similarity is low, try `calibration_type: mix` or prepare more representative calibration images.
- If the output shape is unexpected, inspect the ONNX model with Netron and update runtime post-processing accordingly.
