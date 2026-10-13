English | [简体中文](README_cn.md)

# Faster R-CNN model conversion

The PartA/PartB pairs for all four targets are published in the public archive and can be downloaded directly; see the [model guide](../model/README.md#preparation). This guide rebuilds those pairs from the TorchVision source.

<a id="source-model"></a>
## Source model

[TorchVision v0.21.0](https://github.com/pytorch/vision/tree/v0.21.0/torchvision/models/detection), `fasterrcnn_mobilenet_v3_large_320_fpn`, weights `COCO_V1`. The constructor uses `weights_backbone=None` with the COCO weights. The [official checkpoint](https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth) is downloaded into the Torch Hub cache by `export_split.py`. SHA-256: `907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5`.

<a id="directory"></a>
## Directory structure

```text
export_split.py              # Export PartA and two PartB layouts
prepare_calibration.py       # Image and ROI-feature calibration sets
recorded_calibration_urls.txt # COCO image URLs and hashes
make_configs.py              # Eight target/stage compile configs
compile_models.sh            # X5 or S-family compile
verify_pair.py               # File digest comparison
```

<a id="toolchain-targets"></a>
## Toolchains

| Stage | Image / version |
| --- | --- |
| Export and calibration | `registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0`; torch 2.6.0+cpu, torchvision 0.21.0+cpu, NumPy 1.23.0, ONNX 1.15.0, ONNX Runtime 1.19.0, Pillow 9.3.0 |
| X5 compile | `openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310`; hb_mapper 1.24.3, HBDK 3.49.15 |
| S compile | `registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0`; HBDK4 4.7.5 |

Obtain the images from the [X5 guide](https://developer.d-robotics.cc/rdk_x_doc/en/Advanced_development/toolchain_development/intermediate/environment_config) and [S guide](https://developer.d-robotics.cc/rdk_s_doc/en/Advanced_development/toolchain_development/algorithm_toolchain/overview). Run Docker on x86_64 Linux. Mount the repository at `/work`; commands below start from the host repository root.

<a id="export"></a>
## Export ONNX

Export and calibrate in the CPU container:
```bash
docker run --rm -v "$PWD":/work \
  -w /work/samples/vision/fasterrcnn/conversion \
  registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0 bash -lc '
    python3 -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu &&
    python3 -m pip install numpy==1.23.0 onnx==1.15.0 onnxruntime==1.19.0 Pillow==9.3.0 &&
    python3 export_split.py &&
    python3 prepare_calibration.py --images ../test_data/bus.jpg ../../yolov5/test_data/kite.jpg --coco-urls recorded_calibration_urls.txt --hflip repo
  '
```
Export writes `onnx/part_a_f0only.op11.onnx`, `part_a_f0only.op16.onnx`, `part_b_batch150.op11.onnx`, `part_b_packed.op16.onnx` and `export_report.json`. The deployment canvas is 320×320 with aspect-preserving resize and zero padding at the bottom/right. PartA outputs RPN logits, deltas and feat0. PartB consumes up to 150 padded RoIs.

<a id="calibration"></a>
## Calibration inputs

The command uses `bus.jpg` and `kite.jpg` from this repository, their horizontal flips, and 18 COCO images in the URL list: 22 samples. `prepare_calibration.py` checks listed image digests and writes `calib/manifest.json`.

| Directory | Data |
| --- | --- |
| `calib/parta_x5` | Headerless float32 RGB CHW, [0,255] |
| `calib/parta_s` | Normalized float32 NumPy image arrays |
| `calib/partb` | Float32 pooled ROI features, 150×256×7×7 |
| `calib/partb_packed_s` | Float32 ROI banks, 1×256×1050×7 |

For your own images, run inside the export container: `python3 prepare_calibration.py --images /work/calibration_images --hflip all`.

<a id="compile"></a>
## Compile board models

X5:
```bash
docker run --rm -v "$PWD":/work \
  -w /work/samples/vision/fasterrcnn/conversion \
  openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310 bash -lc \
  'python3 make_configs.py && bash compile_models.sh x5'
```
S100, S100P and S600:
```bash
docker run --rm -v "$PWD":/work \
  -w /work/samples/vision/fasterrcnn/conversion \
  registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0 bash -lc \
  'python3 make_configs.py && bash compile_models.sh s'
```
`make_configs.py` writes `build/<target>/<part>/config.yaml`. PartA uses NV12 input, BT.601 video range, RGB mean `123.675 116.28 103.53` and scale `0.0171247538317 0.0175070028011 0.0174291938998`. X5 uses O3/latency/one core; S uses O2/latency/one core. PartB accepts float32 feature inputs and quantizes them in the graph. X5 takes `[150,256,7,7]`; S takes `[1,256,1050,7]` with unpadded input/output.

<a id="validation"></a>
## Inspect and run the artifacts

From the repository root:
```bash
python3 samples/vision/fasterrcnn/conversion/verify_pair.py x5
```
This prints SHA-256 values and compares them with `model/frcnn_mbv3_320.json`: exit 0 means both recorded digests match, 1 means a missing file, 2 means a different digest. For a self-compiled pair, supply both runtime file paths. Copy the files to the board, then run:
```bash
python3 samples/vision/fasterrcnn/runtime/python/main.py --target x5 \
  --parta-path samples/vision/fasterrcnn/model/x5/frcnn_mbv3_320_parta.bin \
  --partb-path samples/vision/fasterrcnn/model/x5/frcnn_mbv3_320_partb.bin
```

<a id="artifacts"></a>
## Artifacts

| Target | march | PartA | PartB |
| --- | --- | --- | --- |
| x5 | bayes-e | `build/x5/parta/frcnn_mbv3_320_parta.bin` | `build/x5/partb/frcnn_mbv3_320_partb.bin` |
| s100 | nash-e | `build/s100/parta/frcnn_mbv3_320_parta.hbm` | `build/s100/partb/frcnn_mbv3_320_partb.hbm` |
| s100p | nash-m | `build/s100p/parta/frcnn_mbv3_320_parta.hbm` | `build/s100p/partb/frcnn_mbv3_320_partb.hbm` |
| s600 | nash-p | `build/s600/parta/frcnn_mbv3_320_parta.hbm` | `build/s600/partb/frcnn_mbv3_320_partb.hbm` |
Paths are relative to `conversion/`; place the pair under `model/<target>/` for board deployment.

<a id="known-gaps"></a>
## Prepare your own calibration set

Place JPEG/PNG/BMP images in a directory mounted into the container. Pass that directory with `--images`; choose `--hflip repo`, `all` or `none`. Generate all four calibration directories before running `make_configs.py` and the selected family compiler.

