[English](README.md) | 简体中文

# Faster R-CNN

<a id="overview"></a>
## 模型介绍

[TorchVision Faster R-CNN](https://github.com/pytorch/vision/tree/v0.21.0/torchvision/models/detection) 通过候选区域网络与区域分类头完成 COCO 目标检测。本示例使用 MobileNetV3-Large-320-FPN，视频输入可通过 ByteTrack 跟踪行人。骨干网络/RPN 与分类回归头在 BPU 上运行，CPU 处理候选区域选择、RoIAlign 和轨迹关联。

<a id="directory"></a>
## 目录结构

```text
fasterrcnn/
├── conversion/       # Weight export, calibration and board compilation
├── model/            # Published-pair download and install, target metadata, COCO labels
├── runtime/python/   # main.py, cli.py, detect.py, tracking.py, run.sh
├── runtime/cpp/      # Three source files, two headers, CMakeLists.txt, run.sh
└── test_data/        # bus.jpg image input
```

<a id="support-matrix"></a>
## 支持板卡

| Target | Model | Python | C++ | Runtime SDK |
| --- | --- | --- | --- | --- |
| `x5` | MobileNetV3-Large-320-FPN, PartA + PartB `.bin` | ✓ | ✓ | DNN 1.24.5 |
| `s100` | MobileNetV3-Large-320-FPN, PartA + PartB `.hbm` | ✓ | ✓ | DNN/UCP 3.13.6, HBRT 4.7.5 |
| `s100p` | MobileNetV3-Large-320-FPN, PartA + PartB `.hbm` | ✓ | ✓ | DNN/UCP 3.13.6, HBRT 4.7.5 |
| `s600` | MobileNetV3-Large-320-FPN, PartA + PartB `.hbm` | ✓ | ✓ | DNN/UCP 3.13.6, HBRT 4.7.5 |

<a id="prerequisites"></a>
## 运行环境

使用板卡系统中的 `hbm_runtime`、NumPy、OpenCV 和 SciPy。X5/S100/S100P 使用 Python 3.10，S600 使用 Python 3.12。行人跟踪还需安装两个跟踪依赖，在仓库根目录创建环境并执行下面的命令。源码构建需要 C++ 编译器和当前 NumPy 的头文件。
```bash
python3 -m venv --system-site-packages .venv-fasterrcnn
. .venv-fasterrcnn/bin/activate
python3 -m pip install Cython setuptools wheel PyYAML==6.0.3
python3 -m pip install --no-build-isolation --no-deps lap==0.5.12 cython-bbox==0.1.5
```

模型转换仅在自行编译时需要：在装有 Docker 的 x86_64 Linux 电脑上进行，使用[转换指南](conversion/README_cn.md#toolchain-targets)中的 OE 镜像。两段模型使用同一板卡目标的编译产物。

<a id="quickstart"></a>
## 快速运行

下载已发布的模型对，把仓库复制到板卡后运行 Python 推理。在可联网电脑的仓库根目录执行：
下载电脑需要 Python 3.10 或更高版本。
```bash
python3 -m venv .venv-model-download
. .venv-model-download/bin/activate
python3 -m pip install PyYAML==6.0.3
python3 samples/vision/fasterrcnn/model/download.py --target x5
```
该命令从公开档案站下载 `frcnn_mbv3_320_parta.bin` 与 `frcnn_mbv3_320_partb.bin` 到 `model/x5/`，并按 `model/frcnn_mbv3_320.json` 校验两个 SHA-256。`--target` 可接 `x5`、`s100`、`s100p`、`s600` 中的一个或多个，省略时下载全部四组。已存在的文件只做校验，不会被覆盖。`model/download.sh` 接受相同参数。

将仓库连同 `model/x5/` 复制到 X5，在板卡的仓库根目录运行：
```bash
python3 samples/vision/fasterrcnn/runtime/python/main.py --target x5
```
终端输出 JSON 检测结果，并生成 `samples/vision/fasterrcnn/test_data/result.jpg`。其他板卡先下载对应目标，再使用匹配的 `--target s100`、`--target s100p` 或 `--target s600`。

### 自行编译模型对（可选）

已发布的模型对也可以从 TorchVision 源码重建。在转换电脑的仓库根目录操作。先按 [X5 工具链说明](https://developer.d-robotics.cc/rdk_x_doc/en/Advanced_development/toolchain_development/intermediate/environment_config)与 [S 工具链说明](https://developer.d-robotics.cc/rdk_s_doc/en/Advanced_development/toolchain_development/algorithm_toolchain/overview)获取 OE 镜像，导出权重并准备校准数据：
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
编译 X5 的两段模型：
```bash
docker run --rm -v "$PWD":/work \
  -w /work/samples/vision/fasterrcnn/conversion \
  openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310 bash -lc \
  'python3 make_configs.py && bash compile_models.sh x5'
```
将产物复制到对应模型目录：
```bash
python3 samples/vision/fasterrcnn/model/prepare.py --target x5 \
  --from-build samples/vision/fasterrcnn/conversion/build
```
在 X5 板卡的仓库根目录运行自行编译的模型：
```bash
python3 samples/vision/fasterrcnn/runtime/python/main.py --target x5 \
  --parta-path samples/vision/fasterrcnn/model/x5/frcnn_mbv3_320_parta.bin \
  --partb-path samples/vision/fasterrcnn/model/x5/frcnn_mbv3_320_partb.bin
```
S100/S100P/S600 的 OE 容器、编译与安装命令见[转换指南](conversion/README_cn.md#compile)。

<a id="expected-results"></a>
## 运行结果

使用所列模型组合，`bus.jpg` 在 X5/S100/S100P 上输出 22 个检测框，在 S600 上输出 17 个。结果包含原图像素坐标 `xyxy`、置信度和 COCO 类别 ID，行人的 ID 为 1。视频通过 `--video` 和 `--records` 指定，输出标注 MP4 和逐帧轨迹记录。

模型为 MobileNetV3-Large-320-FPN，输入画布 320×320，最多 150 个候选区域；输入为 30 帧连续的 1280×720 视频画面，约 7–9 条活动轨迹，参数为 `--threads 1 --no-encode`。表中是平均耗时。`predict` 包含图像前处理、两段 BPU 模型、CPU RPN/RoIAlign、检测结果解码及行人跟踪。跟踪耗时是 `predict` 中 CPU 轨迹关联的部分。两列都不包含视频解码和编码，SDK 版本见支持表。

| Board | Python predict (ms/frame) | C++ predict (ms/frame) | Python tracker (ms/frame) | C++ tracker (ms/frame) |
| --- | ---: | ---: | ---: | ---: |
| X5 8 GB | 2168.65 | 218.74 | 6.57 | 0.157 |
| S100 | 483.81 | 44.71 | 2.10 | 0.041 |
| S100P | 387.14 | 33.79 | 1.67 | 0.034 |
| S600 64 GB | 259.27 | 32.56 | 1.51 | 0.033 |

`--threads 2` 和 `--threads 4` 调整 OpenCV 图像转换、缩放的 CPU 线程数。同一片段在三个设置下的整体推理耗时接近。

<a id="entry-points"></a>
## 使用入口

[模型准备](model/README_cn.md) · [Python 推理](runtime/python/README_cn.md) · [C++ 推理](runtime/cpp/README_cn.md) · [模型转换](conversion/README_cn.md)。

<a id="license"></a>
## 许可

示例代码遵循仓库 [LICENSE](../../../LICENSE)。TorchVision 源码采用 [BSD-3-Clause](https://github.com/pytorch/vision/blob/v0.21.0/LICENSE)，权重来自官方 COCO_V1 发布。ByteTrack 与 LAP 的来源见[跟踪器来源表](../bytetrack/runtime/python/TRACKER_SOURCE_MAP.json)及 [C++ 第三方许可](../bytetrack/runtime/cpp/THIRD_PARTY_LICENSES.md)。数据使用条件见 [COCO](https://cocodataset.org/#termsofuse)。

