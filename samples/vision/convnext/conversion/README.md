English | [简体中文](README_cn.md)

# ConvNeXt conversion

## Reproducible Atto D2 X5 recipe

The maintained recipe `convnext_atto_d2_x5.yaml` builds a **new** artifact from
[timm/convnext_atto.d2_in1k](https://huggingface.co/timm/convnext_atto.d2_in1k/tree/435c0a8d02d5b872e90c8cde36887465982cb399)
(Apache-2.0), revision `435c0a8d02d5b872e90c8cde36887465982cb399`.
The weight SHA-256 is `0f70bc49f1896c8c15af857d2d1d418abfaf097a62b4e29a35e6e65b752f5953`.
The small Atto/Femto/Pico/Nano variants are timm extensions to the ConvNeXt
architecture from [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545).

`export.py` loads local safetensors strictly without downloading a changing
`pretrained=True` checkpoint. It exports fixed 1×3×224×224 RGB float32 ONNX
(opset 11), simplifies/checks the graph and compares logits against PyTorch
with ONNX Runtime CPU on one deterministic input. The adjacent `.export.json`
records input, output, source hashes and versions. This is a host graph check,
not ImageNet accuracy or board inference.

```bash
# Repository root, export environment: torch 2.8.0, timm 1.0.20,
# safetensors, onnx 1.17.0, onnxsim 0.4.36, onnxruntime 1.19.2.
python3 samples/vision/convnext/conversion/export.py --variant atto \
  --checkpoint /path/to/pinned/model.safetensors \
  --output /path/to/run/convnext_atto_d2_224x224.onnx
python3 samples/vision/convnext/conversion/prepare_calibration.py \
  --images /path/to/frozen/images --count 200 \
  --output /path/to/run/calibration_rgb_f32
cp samples/vision/convnext/conversion/convnext_atto_d2_x5.yaml /path/to/run/compile.yaml
# X5 OE container, cwd /path/to/run; resolve recipe paths inside this container.
hb_mapper checker --model convnext_atto_d2_224x224.onnx --model-type onnx --march bayes-e
hb_mapper makertbin --model-type onnx --config compile.yaml
hb_model_info compiled/convnext_atto_d2_bayese_224x224_nv12.bin
hb_perf compiled/convnext_atto_d2_bayese_224x224_nv12.bin
```

Use the already provisioned X5 image
`registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_20_x5_gpu:v1.2.8` on x86 Linux.
The observed tools are hb_mapper 1.24.3, hbdk 3.49.15 and HBRT 3.15.55.0.
The target is `bayes-e` (X5); no S100/S100P/S600 conversion is claimed.

## Calibration and runtime contract

`prepare_calibration.py` freezes a sorted image set and writes raw float32
RGB/NCHW tensors using `numpy.tofile` (`.rgb`, **not** `.npy`). Values remain
0–255. It uses the same Pillow bicubic shorter-edge 256 / rounded center crop
224 geometry as the native `atto-d2` runtime. Mapper fuses ImageNet mean/scale;
do not normalize twice. The adjacent manifest records source/tensor hashes,
count and producer identity. Keep calibration images/tensors private.

The recipe declares BT.601 video-range NV12 runtime input, matching OpenCV's
native conversion, and float32 1,000-class logits output. It uses default PTQ,
latency/O3 and the unique output prefix `convnext_atto_d2_bayese_224x224_nv12`.
The emitted file is registered separately as native variant `atto-d2`.
See [model artifacts](../model/README.md) and [runtime](../runtime/python/README.md).

## Acceptance and publication

The 2026-11-02 conversion passed checker/compiler and host graph comparison.
The model is **converted**, with board smoke, accuracy, Runtime 1/2 concurrency,
and C++ 1/2 streams all **not-run**. `hb_perf` and OE HTML contain compiler
estimates only; they must not populate board benchmarks. Publish the BIN and
SHA256SUMS under `rdk_model_zoo_web/rdk_x5/convnext/`; store versioned companions
under `rdk_model_zoo_web/assets/rdk_x5/convnext/<model-stem>/v1/`.

## Historical reference recipes

`ConvNeXt_atto.yaml`, `ConvNeXt_femto.yaml` and `ConvNeXt_nano.yaml` remain
verbatim `rdk_x5@ac11571` historical references. Their ONNX paths are cross-swapped
(atto → femto, femto → external atto, nano → pico), and all share the output
prefix `ConvNeXt-deploy_224x224_nv12`. They cannot identify the new Atto D2 BIN.
The four historical manifest files and their recorded measurements remain
unchanged. Do not use those historical metrics for the newly converted file.
