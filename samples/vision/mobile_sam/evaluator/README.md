English | [简体中文](./README_cn.md)

# MobileSAM Model Evaluation

This directory documents the board-side performance evaluation of the quantized MobileSAM encoder and decoder. The sample ships a latency/throughput benchmark only; it does not provide an accuracy harness (box-prompt segmentation has no standard evaluation script in this sample).

## Environment Setup

- **Board**: RDK S100 / S100P / S600 with the RDK-S runtime installed.
- **Tool**: `hrt_model_exec`, provided by the RDK-S development kit, not pip.
- **Model**: the compiled `.hbm` files under `../model/nash-e/`, `../model/nash-m/` or `../model/nash-p/`.

## Usage

Measure each model separately with the board `hrt_model_exec perf` tool:

```bash
# Encoder (single-thread latency)
hrt_model_exec perf --model_file ../model/nash-e/mobile_sam_image_encoder_norm_512x512_nashe.hbm --thread_num 1

# Encoder (multi-thread throughput)
hrt_model_exec perf --model_file ../model/nash-e/mobile_sam_image_encoder_norm_512x512_nashe.hbm --thread_num 2

# Decoder
hrt_model_exec perf --model_file ../model/nash-e/mobile_sam_decoder_512_nashe.hbm --thread_num 1
```

Replace `nash-e` with `nash-m` (S100P) or `nash-p` (S600) as appropriate. The benchmark below uses 1 and 2 threads on S100/S100P, and 1 and 12 threads on S600 (multi-core via `--core_id 1,2,3,4`).

## Benchmark Results

### RDK S100 Performance Data

| Device | Model | Size <br> (Pixels) | Classes | BPU Task Latency / <br> BPU Throughput (Threads) | CPU Latency | Params <br> (M) | FLOPs <br> (G) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| S100 | MobileSAM Encoder | 512x512 | - | 10.97 ms / 91.03 FPS (1 thread) <br> 21.29 ms / 93.61 FPS (2 threads) | - | 6.07 | 20.78 |
| S100 | MobileSAM Decoder | 256x32x32 (embedding) + 1x4 (box) | - | 3.30 ms / 297.47 FPS (1 thread) <br> 4.42 ms / 443.42 FPS (2 threads) | - | 4.06 | 0.94 |

### RDK S100P Performance Data

| Device | Model | Size <br> (Pixels) | Classes | BPU Task Latency / <br> BPU Throughput (Threads) | CPU Latency | Params <br> (M) | FLOPs <br> (G) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| S100P | MobileSAM Encoder | 512x512 | - | 8.52 ms / 117.26 FPS (1 thread) <br> 16.50 ms / 120.79 FPS (2 threads) | - | 6.07 | 20.78 |
| S100P | MobileSAM Decoder | 256x32x32 (embedding) + 1x4 (box) | - | 2.84 ms / 345.15 FPS (1 thread) <br> 3.88 ms / 505.87 FPS (2 threads) | - | 4.06 | 0.94 |

### RDK S600 Performance Data

| Device | Model | Size <br> (Pixels) | Classes | BPU Task Latency / <br> BPU Throughput (Threads) | CPU Latency | Params <br> (M) | FLOPs <br> (G) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| S600 | MobileSAM Encoder | 512x512 | - | 5.52 ms / 180.70 FPS (1 thread) <br> 15.87 ms / 738.48 FPS (12 threads) | - | 6.07 | 20.78 |
| S600 | MobileSAM Decoder | 256x32x32 (embedding) + 1x4 (box) | - | 2.58 ms / 381.09 FPS (1 thread) <br> 6.42 ms / 1772.04 FPS (12 threads) | - | 4.06 | 0.94 |

## Performance Test Instructions

- **Device**: The hardware device used for testing (S100, S100P, S600).
- **Model**: `MobileSAM Encoder` is the TinyViT image encoder (`mobile_sam_image_encoder_norm_512x512_*.hbm`); `MobileSAM Decoder` is the box-prompt mask decoder (`mobile_sam_decoder_512_*.hbm`).
- **Size (Pixels)**: The square input resolution. The encoder takes the `512×512` RGB image; the decoder takes the `256×32×32` image embedding produced by the encoder plus a `1×4` box prompt for that same `512×512` image.
- **Classes**: MobileSAM is prompt-based and class-agnostic, so no fixed class count applies (`-`).
- **Input type**: Unlike detection samples, SAM inputs are float32 NCHW tensors — the encoder consumes an ImageNet-normalized float32 image and the decoder consumes a float32 featuremap and box tensor — not NV12.
- **BPU Task Latency / Throughput**:
  - **Single-thread Latency**: ideal single-frame latency on one thread and one BPU core, measured with `hrt_model_exec perf --thread_num 1 --model_file <model.hbm>`, from task submission to completion including cache warmup.
  - **Multi-thread Throughput**: frames per second when several threads submit to the BPU simultaneously. 2 threads on S100/S100P and 12 threads (multi-core) on S600 give the best BPU utilization; the S600 multi-core run uses `--core_id 1,2,3,4`.
  - **Test Command**: `hrt_model_exec perf --model_file <model.hbm> --thread_num N`.
- **CPU Latency (Single Core)**: CPU-side pre/post-processing (image resize and normalization for the encoder; mask upsampling for the decoder). It is optional and depends on the host pipeline, so it is not listed (see the pipeline note below).
- **Pipeline latency**: the runtime runs the encoder then the decoder sequentially, so the end-to-end mask latency is encoder latency + decoder latency + CPU overhead (the two models are measured separately to stay comparable across boards).
- **Memory management**: in streaming inference the input/output memory is allocated once and reused; the latency does not include allocation/deallocation time.
- **Params (M) & FLOPs (G)**: parameter count and computation volume of the original FP32 model at a `512×512` input. FLOPs are the MACs-derived count (`2×MACs`), consistent with the Ultralytics export-log convention used elsewhere in the Model Zoo. The decoder FLOPs are measured on its `256×32×32` embedding and `1×4` box inputs.

## License

This directory is licensed under the [Apache 2.0 License](../../../../LICENSE).