English | [简体中文](./README_cn.md)

# MobileSAM Model Conversion and Compilation Guide

This directory provides the scripts and instructions required to export MobileSAM from its upstream PyTorch checkpoint, prepare calibration data for both the encoder and the decoder, and compile the two `.hbm` models for RDK S100 / S100P / S600.

## Directory Structure

```text
.
├── scripts/
│   ├── export_encoder_onnx.py        # Export the TinyViT image encoder ONNX
│   ├── export_decoder_onnx.py        # Export the box-prompt mask decoder ONNX
│   ├── prepare_calibration.py        # Encoder calibration: images → ImageNet-norm npy
│   ├── dump_encoder_embedding.py     # Dump one real encoder embedding (.bin)
│   ├── prepare_decoder_calibration.py# Decoder calibration: embedding + boxes → npy
│   └── quantize.py                   # hb_compile runner (encoder + decoder per march)
├── configs/
│   ├── mobile_sam_encoder_{nashe,nashm,nashp}_config.yaml
│   └── mobile_sam_decoder_512_{nashe,nashm,nashp}_config.yaml
├── README.md
└── README_cn.md
```

## Model Compilation Environment

Run conversion on an x86 Linux host with the OpenExplore Docker environment. Do not install the compiler toolchain on the board. RDK S100/S100P/S600 share the same Nash toolchain (`hb_compile`); only `--march` differs.

Toolchain documentation and download:

- OE online documentation: <https://developer.d-robotics.cc/oe_s_doc/index.html>
- RDK S100 toolchain documentation: <https://developer.d-robotics.cc/rdk_s_doc/Advanced_development/toolchain_development/algorithm_toolchain/overview?v=4.0.5&p=RDK+S100>
- RDK S600 toolchain documentation: <https://developer.d-robotics.cc/rdk_s_doc/Advanced_development/toolchain_development/algorithm_toolchain/overview?v=5.1.0&p=RDK+S600>

### 1. Install Docker

Install Docker following the official documentation, then verify it:

```bash
sudo docker --version
sudo docker run --rm hello-world
```

### 2. Obtain and Load the Offline Image

Download the OpenExplore CPU Docker image (shared across S100/S100P/S600), then load it:

```bash
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe/3.7.0/ai_toolchain_ubuntu_22_s100_s600_cpu_v3.7.0.tar
sudo docker load -i ai_toolchain_ubuntu_22_s100_s600_cpu_v3.7.0.tar
sudo docker images
```

Alternatively, pull the image online:

```bash
docker pull registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0
```

> **Note**: If the download URL above expires or becomes invalid, check the latest link on the OE online documentation, or open an issue on the documentation site to have it refreshed.

### 3. Start the Container

Mount the repository and allocate enough shared memory:

```bash
sudo docker run -it --rm \
  --network host \
  --shm-size=15g \
  -v "$(pwd)":/workspace \
  --workdir /workspace \
  <docker-image-name> /bin/bash
```

Run `sudo docker images` to obtain the loaded image name for `<docker-image-name>`.

## Conversion Workflow

MobileSAM is split into two `.hbm` models — an image encoder and a box-prompt decoder — that must be prepared in a strict order. The decoder's calibration input is the encoder's output featuremap (`image_embeddings`, shape `1×256×32×32`), not an image. That featuremap can only be produced by running the encoder, so the encoder must be exported (and, for best fidelity, run once) before any decoder calibration data can exist. Follow the steps in order.

### 1. Export ONNX

The ONNX models are not distributed. MobileSAM ships inside the `ultralytics` package, so install it and provide the official checkpoint:

```bash
cd samples/vision/mobile_sam/conversion
pip install ultralytics
# place the official mobile_sam.pt checkpoint, e.g. workspace/MobileSAM/weights/mobile_sam.pt
```

Then export the two ONNX models:

```bash
python3 scripts/export_encoder_onnx.py --weights ./workspace/MobileSAM/weights/mobile_sam.pt
python3 scripts/export_decoder_onnx.py --checkpoint ./workspace/MobileSAM/weights/mobile_sam.pt
```

Override the checkpoint path with `--weights` / `--checkpoint` if it differs from the default. Run `python3 scripts/export_encoder_onnx.py -h` for the complete argument list.

### 2. Prepare Encoder Calibration Data

Prepare 20 to 50 representative RGB images, then convert them to the encoder's NCHW float32 input tensors (ImageNet mean/std normalization):

```text
calibration_images/
├── 000001.jpg
├── 000002.jpg
└── ...
```

```bash
python3 scripts/prepare_calibration.py --src ./calibration_images --out ./calibration_data_norm_512 --num 30
```

The tensors are written to `./calibration_data_norm_512/normalized_images/*.npy`.

### 3. Obtain the Encoder Embedding

The decoder's calibration input is a real encoder embedding (shape `1×256×32×32`, float32), not an image. It cannot be prepared without first running the encoder. Choose one route:

- **Float encoder on the host** (simplest, no board required): run the exported encoder ONNX on a single image with the committed `dump_encoder_embedding.py` script, which writes the `image_embeddings` output to a raw `.bin` file:

  ```bash
  python3 scripts/dump_encoder_embedding.py \
    --image ./calibration_images/000001.jpg \
    --output ./encoder_embedding.bin
  ```

  `pip install onnxruntime` if it is not already installed.

- **Compiled encoder on the board** (best fidelity): after step 5 compiles the encoder, run it once with `hrt_model_exec` and dump the `image_embeddings` output to the same `.bin`. This calibrates the decoder against the exact quantized-encoder output distribution.

### 4. Prepare Decoder Calibration Data

Feed the single embedding to `prepare_decoder_calibration.py`, which generates `--num` derived featuremaps and matching box prompts under `./decoder_calibration/`:

```bash
python3 scripts/prepare_decoder_calibration.py \
  --embedding ./encoder_embedding.bin \
  --out ./decoder_calibration \
  --box 185 120 380 445 \
  --num 30
```

The featuremaps are written to `./decoder_calibration/image_embeddings/` and the box tensors to `./decoder_calibration/boxes/`. The box prompt is a runtime input, so the decoder requires both calibration inputs.

### 5. Compile HBM Models

If you generated the embedding with the *float* encoder (step 3), both calibration sets are ready and you can compile the encoder and decoder together:

```bash
# RDK S100 (Nash-E)
python3 scripts/quantize.py --march nash-e

# RDK S100P (Nash-M)
python3 scripts/quantize.py --march nash-m

# RDK S600 (Nash-P)
python3 scripts/quantize.py --march nash-p

# all three marches
python3 scripts/quantize.py
```

If you want the embedding to come from the *compiled* encoder, compile the encoder first, dump the embedding, prepare the decoder calibration data, then compile the decoder:

```bash
python3 scripts/quantize.py --config configs/mobile_sam_encoder_nashe_config.yaml
# run the encoder once, dump image_embeddings → encoder_embedding.bin (step 3, board route)
python3 scripts/prepare_decoder_calibration.py --embedding ./encoder_embedding.bin --out ./decoder_calibration
python3 scripts/quantize.py --config configs/mobile_sam_decoder_512_nashe_config.yaml
```

The `.hbm` files are written under `bpu_model_output_encoder_nashe/` and `bpu_model_output_decoder_nashe/`. Copy them to the model directory so `runtime/python/run.sh` and `runtime/python/main.py` can use them directly:

```bash
cp bpu_model_output_encoder_nashe/mobile_sam_image_encoder_norm_512x512_nashe.hbm ../model/nash-e/
cp bpu_model_output_decoder_nashe/mobile_sam_decoder_512_nashe.hbm ../model/nash-e/
```

Repeat for `nash-m` and `nash-p`. The output filenames already match the runtime expectation, so no rename is needed.

### 6. Script Arguments

Run `python3 <script> -h` for the complete list.

**`quantize.py`**

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--march` | Target architecture: `nash-e` (S100), `nash-m` (S100P), `nash-p` (S600). Omit to compile all three. | all |
| `--config` | Compile a single committed YAML (overrides `--march`). | none |

**`export_encoder_onnx.py` / `export_decoder_onnx.py`**

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--repo` | Path to the cloned upstream repository. | `./workspace/MobileSAM` |
| `--weights` (encoder) | Path to `mobile_sam.pt`. | `./workspace/MobileSAM/weights/mobile_sam.pt` |
| `--checkpoint` (decoder) | Path to `mobile_sam.pt`. | `./workspace/MobileSAM/weights/mobile_sam.pt` |
| `--output` | Output ONNX path. | `./mobile_sam_image_encoder_norm_512_op11.onnx` / `./mobile_sam_decoder_512_op11.onnx` |
| `--size` | Square image size. | `512` |
| `--opset` | ONNX opset version. | `11` |
| `--box` (decoder only) | Example export box `x1 y1 x2 y2`. | `185 120 380 445` |

**`prepare_calibration.py`**

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--src` | Directory containing calibration images. | required |
| `--out` | Output root (writes to `<out>/normalized_images/`). | `./calibration_data_norm_512` |
| `--num` | Maximum number of calibration tensors. | `30` |
| `--size` | Square input size. | `512` |

**`dump_encoder_embedding.py`**

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--onnx` | Encoder ONNX path. | `./mobile_sam_image_encoder_norm_512_op11.onnx` |
| `--image` | Single input image path. | required |
| `--output` | Output raw embedding `.bin` path. | `./encoder_embedding.bin` |
| `--size` | Square image size. | `512` |

**`prepare_decoder_calibration.py`**

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--embedding` | Raw float32 encoder embedding `1×256×32×32` (the `.bin`). | required |
| `--out` | Output root (writes to `<out>/image_embeddings/` and `<out>/boxes/`). | `./decoder_calibration` |
| `--num` | Number of calibration samples. | `30` |
| `--box` | Base box `x1 y1 x2 y2`. | `185 120 380 445` |

## Input and Output Protocol

Export and quantization must preserve this fixed tensor chain.

**Encoder** (`mobile_sam_image_encoder_norm_512_op11.onnx`):

- Input `normalized_images`: `1×3×512×512`, RGB, float32, ImageNet mean/std normalized.
- Output `image_embeddings`: `1×256×32×32`, float32.

**Decoder** (`mobile_sam_decoder_512_op11.onnx`):

- Input `image_embeddings`: `1×256×32×32`, float32 — the encoder output.
- Input `boxes`: `1×4`, float32 — box prompt `x1 y1 x2 y2` in resized-image coordinates.
- Output `low_res_masks` + `iou_predictions`: low-resolution mask logits (from `1×1×256×256` up to `1×1×512×512`) and IoU scores.

The box prompt is a runtime input, so the compiled decoder takes both `image_embeddings` and `boxes`. Both networks are quantized to int16 via `calibration_parameters.optimization: set_all_nodes_int16` in the committed configs.

## Compile Result Check

```bash
hrt_model_exec model_info --model_file mobile_sam_image_encoder_norm_512x512_nashe.hbm
hrt_model_exec perf --model_file mobile_sam_image_encoder_norm_512x512_nashe.hbm --thread_num 1
hrt_model_exec perf --model_file mobile_sam_decoder_512_nashe.hbm --thread_num 1
```

## Troubleshooting

- **Permission issues**: If copied files on the host have unexpected ownership, check file ownership or run `sudo chown -R`.
- **Memory or IPC errors**: Start Docker with `--shm-size=15g`.
- **Optimization-level errors**: If `O3` is not supported on Nash, use `O0`, `O1`, or `O2`.
- **"No calibration images found"**: Point `--src` at a directory containing at least one `.jpg`/`.png` image.
- **Decoder embedding reshape error**: The `--embedding` file must be a raw float32 array with exactly `1×256×32×32` (262144) values — an encoder output, not an image or an `.npy` file.
- **Encoder/decoder normalization kept consistent**: The encoder applies ImageNet normalization inside `prepare_calibration.py`; keep the same normalization in the embedding dump and runtime so the `1×256×32×32` tensor contract holds.
- **Shape mismatch**: Keep `--size 512` (image embedding size 32) for both export and calibration, or the `1×256×32×32` tensor contract breaks.

## License

Tools in this directory follow the [Apache 2.0 License](../../../../LICENSE).