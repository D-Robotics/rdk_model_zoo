English | [简体中文](./README_cn.md)

# DINOv2 Model Evaluation Guide

This directory documents the measured performance and accuracy of the
DINOv2 ViT-S/14 int16 model on RDK S100/S100P/S600, and the commands to
reproduce them.

## Performance Data

Measured with `hrt_model_exec perf` (200 frames, performance governor locked).

| Device | Model | Input Size | BPU Task Latency / BPU Throughput |
|---|---|---|---|
| RDK S100 | dinov2_vits14_224_int16 | 1x3x224x224 | 3.73 ms / 267.44 FPS (1 thread) <br> 288.26 FPS (2 threads) |
| RDK S100P | dinov2_vits14_224_int16 | 1x3x224x224 | 3.02 ms / 329.53 FPS (1 thread) <br> 357.63 FPS (2 threads) |
| RDK S600 | dinov2_vits14_224_int16 | 1x3x224x224 | 2.25 ms / 441.64 FPS (1 thread) <br> 1898.42 FPS (12 threads, `--core_id 1,2,3,4`) |

Model parameters: 22.06 M. Latency is pure BPU forward; board-side CPU
preprocessing is additional. The canonical CPU transform is OpenCV BGR to RGB,
bicubic resize of the short side to 256 while preserving aspect ratio, center
crop 224 by 224, `/255`, ImageNet mean/std normalization, and contiguous
float32 NCHW layout.

## Performance Test Method

Lock frequencies first, then run the perf tool. These commands reproduce the
thread and core settings reported in the table:

```bash
# RDK S100 (nash-e): 1 and 2 threads
hrt_model_exec perf \
    --model_file ../model/nash-e/dinov2_vits14_224_int16_nashe.hbm \
    --thread_num 1
hrt_model_exec perf \
    --model_file ../model/nash-e/dinov2_vits14_224_int16_nashe.hbm \
    --thread_num 2

# RDK S100P (nash-m): 1 and 2 threads
hrt_model_exec perf \
    --model_file ../model/nash-m/dinov2_vits14_224_int16_nashm.hbm \
    --thread_num 1
hrt_model_exec perf \
    --model_file ../model/nash-m/dinov2_vits14_224_int16_nashm.hbm \
    --thread_num 2

# RDK S600 (nash-p): 1 thread, then 12 threads on four BPU cores
hrt_model_exec perf \
    --model_file ../model/nash-p/dinov2_vits14_224_int16_nashp.hbm \
    --thread_num 1
hrt_model_exec perf \
    --model_file ../model/nash-p/dinov2_vits14_224_int16_nashp.hbm \
    --thread_num 12 \
    --core_id 1,2,3,4
```

Frequency-locking commands are listed in the repository top-level README FAQ.

## Accuracy Data

### PTQ per-output cosine (toolchain report)

| Output | Calibrated Cosine | Quantized Cosine |
|---|---|---|
| cls_feat | 0.9990 | 0.9989 |
| patch_feat | 0.9985 | 0.9983 |

Quantized cosine is measured against the float ONNX model by the toolchain.
Identical values were reproduced with an independent export script and a
different 50-image calibration set.

### Board-executed cosine vs float ONNX

The quantized model was executed on-board via `hbm_runtime` and compared
against ONNXRuntime float32 references on the same inputs:

| Device | cls_feat | patch_feat |
|---|---|---|
| RDK S100 | 0.9987 - 0.9989 | 0.9977 - 0.9986 |
| RDK S100P | 0.9987 - 0.9989 | 0.9977 - 0.9986 |
| RDK S600 | 0.9988 - 0.9989 | 0.9975 - 0.9986 |

Reproduce with: run the exported float ONNX on the host with ONNXRuntime to
produce reference outputs, push the tensors and the `.hbm` to the board, run
`hbm_runtime.HB_HBMRuntime(...).run()` on the same inputs, and compute the
cosine similarity between the board outputs and the references.

## Result Check

Run the default demo and check that the output statistics are finite,
non-zero, and stable across repeated runs:

```bash
cd ../runtime/python
bash run.sh
```

The demo prints an output summary (shape / dtype / mean / std / min / max /
l2_norm) and the cosine similarity between the two test images. Repeated runs
with the same input must produce identical summaries.

## License

See [../../../../LICENSE](../../../../LICENSE).
