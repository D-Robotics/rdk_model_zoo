English | [简体中文](./README_cn.md)

# EfficientSAM Model Evaluation

This directory documents the board-side performance evaluation of the quantized EfficientSAM encoder and decoder. The sample ships a latency/throughput benchmark only; it does not provide an accuracy harness (sparse point-prompt segmentation has no standard evaluation script in this sample).

## Environment Setup

- **Board**: RDK S100 / S100P / S600 with the RDK-S runtime installed.
- **Tool**: `hrt_model_exec`, provided by the RDK-S development kit, not pip.
- **Model**: the compiled `.hbm` files under `../model/nash-e/`, `../model/nash-m/` or `../model/nash-p/`.

## Usage

Measure each model separately with the board `hrt_model_exec perf` tool:

```bash
# Encoder (single-thread latency)
hrt_model_exec perf --model_file ../model/nash-e/efficient_sam_vitt_encoder_512x512_nashe.hbm --thread_num 1

# Encoder (multi-thread throughput)
hrt_model_exec perf --model_file ../model/nash-e/efficient_sam_vitt_encoder_512x512_nashe.hbm --thread_num 2

# Decoder
hrt_model_exec perf --model_file ../model/nash-e/efficient_sam_vitt_decoder_512_nashe.hbm --thread_num 1
```

Replace `nash-e` with `nash-m` (S100P) or `nash-p` (S600) as appropriate. The benchmark below uses 1 and 2 threads on S100/S100P, and 1 and 12 threads on S600 (multi-core via `--core_id 1,2,3,4`).

## Benchmark Results

### RDK S100 Performance Data

| Device | Model | Size <br> (Pixels) | Classes | BPU Task Latency / <br> BPU Throughput (Threads) | CPU Latency | Params <br> (M) | FLOPs <br> (G) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| S100 | EfficientSAM Encoder | 512x512 | - | 11.78 ms / 84.75 FPS (1 thread) <br> 22.93 ms / 86.95 FPS (2 threads) | - | 6.16 | 22.19 |
| S100 | EfficientSAM Decoder | 256x32x32 (embedding) | - | 3.25 ms / 306.31 FPS (1 thread) <br> 5.94 ms / 334.44 FPS (2 threads) | - | 4.06 | 0.98 |

### RDK S100P Performance Data

| Device | Model | Size <br> (Pixels) | Classes | BPU Task Latency / <br> BPU Throughput (Threads) | CPU Latency | Params <br> (M) | FLOPs <br> (G) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| S100P | EfficientSAM Encoder | 512x512 | - | 9.36 ms / 106.69 FPS (1 thread) <br> 18.20 ms / 109.52 FPS (2 threads) | - | 6.16 | 22.19 |
| S100P | EfficientSAM Decoder | 256x32x32 (embedding) | - | 2.49 ms / 399.96 FPS (1 thread) <br> 4.47 ms / 445.74 FPS (2 threads) | - | 4.06 | 0.98 |

### RDK S600 Performance Data

| Device | Model | Size <br> (Pixels) | Classes | BPU Task Latency / <br> BPU Throughput (Threads) | CPU Latency | Params <br> (M) | FLOPs <br> (G) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| S600 | EfficientSAM Encoder | 512x512 | - | 6.72 ms / 148.60 FPS (1 thread) <br> 19.58 ms / 598.44 FPS (12 threads) | - | 6.16 | 22.19 |
| S600 | EfficientSAM Decoder | 256x32x32 (embedding) | - | 1.50 ms / 662.55 FPS (1 thread) <br> 4.08 ms / 2831.10 FPS (12 threads) | - | 4.06 | 0.98 |

## Performance Test Instructions

- **Device**: The hardware device used for testing (S100, S100P, S600).
- **Model**: `EfficientSAM Encoder` is the ViT-Tiny image encoder (`efficient_sam_vitt_encoder_512x512_*.hbm`); `EfficientSAM Decoder` is the fixed-prompt mask decoder (`efficient_sam_vitt_decoder_512_*.hbm`).
- **Size (Pixels)**: The square input resolution. The encoder takes the `512×512` RGB image; the decoder takes the `256×32×32` image embedding produced by the encoder for that same `512×512` image.
- **Classes**: EfficientSAM is prompt-based and class-agnostic, so no fixed class count applies (`-`).
- **Input type**: Unlike detection samples, SAM inputs are float32 NCHW tensors — the encoder consumes an RGB float32 image (scaled by `1/255`) and the decoder consumes a float32 featuremap — not NV12.
- **BPU Task Latency / Throughput**:
  - **Single-thread Latency**: ideal single-frame latency on one thread and one BPU core, measured with `hrt_model_exec perf --thread_num 1 --model_file <model.hbm>`, from task submission to completion including cache warmup.
  - **Multi-thread Throughput**: frames per second when several threads submit to the BPU simultaneously. 2 threads on S100/S100P and 12 threads (multi-core) on S600 give the best BPU utilization; the S600 multi-core run uses `--core_id 1,2,3,4`.
  - **Test Command**: `hrt_model_exec perf --model_file <model.hbm> --thread_num N`.
- **CPU Latency (Single Core)**: CPU-side pre/post-processing (image resize and normalization for the encoder; mask upsampling for the decoder). It is optional and depends on the host pipeline, so it is not listed (see the pipeline note below).
- **Pipeline latency**: the runtime runs the encoder then the decoder sequentially, so the end-to-end mask latency is encoder latency + decoder latency + CPU overhead (the two models are measured separately to stay comparable across boards).
- **Memory management**: in streaming inference the input/output memory is allocated once and reused; the latency does not include allocation/deallocation time.
- **Params (M) & FLOPs (G)**: parameter count and computation volume of the original FP32 model at a `512×512` input. FLOPs are the MACs-derived count (`2×MACs`), consistent with the Ultralytics export-log convention used elsewhere in the Model Zoo. The decoder FLOPs are measured on its `256×32×32` embedding input.

## License

This directory is licensed under the [Apache 2.0 License](../../../../LICENSE).