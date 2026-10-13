[English](README.md) | 简体中文

# Faster R-CNN C++ runtime

<a id="overview"></a>
## 功能

使用成对板端模型执行 COCO 检测，并通过 CPU ByteTrack 进行行人跟踪。

<a id="directory"></a>
## 目录结构

```text
src/main.cpp  # 创建模型并调用 predict
src/cli.cpp   # 参数和图片/视频读写
src/detect.cpp # 检测器、板端 Runtime 和行人跟踪
inc/cli.hpp
inc/detect.hpp
CMakeLists.txt
run.sh
```

<a id="supported-boards"></a>
## 适用板卡

| Target | Model | Python | C++ | Runtime SDK |
| --- | --- | --- | --- | --- |
| `x5` | MobileNetV3-Large-320-FPN, PartA + PartB `.bin` | ✓ | ✓ | DNN 1.24.5 |
| `s100` | MobileNetV3-Large-320-FPN, PartA + PartB `.hbm` | ✓ | ✓ | DNN/UCP 3.13.6, HBRT 4.7.5 |
| `s100p` | MobileNetV3-Large-320-FPN, PartA + PartB `.hbm` | ✓ | ✓ | DNN/UCP 3.13.6, HBRT 4.7.5 |
| `s600` | MobileNetV3-Large-320-FPN, PartA + PartB `.hbm` | ✓ | ✓ | DNN/UCP 3.13.6, HBRT 4.7.5 |
X5 使用 `.bin`，S 系列使用对应板卡的 `.hbm` 组合。

<a id="dependencies"></a>
## 依赖

需要 CMake 3.16 或更高版本、C++17 编译器、OpenCV 开发头文件与库，以及板端 DNN SDK。S 系列还需要 `hbucp`。X5 的 SDK 库目录为 `/usr/lib`，S 系列为 `/usr/hobot/lib`。可通过 CMake 的 `FRCNN_HOBOT_LIB_DIR`、`FRCNN_HOBOT_INCLUDE_DIR` 和 `FRCNN_SYSTEM_HOBOT_INCLUDE_DIR` 指定 SDK 位置。CPU 跟踪器源码位于 ByteTrack Sample。

<a id="build"></a>
## 板端构建

在仓库根目录执行：
```bash
cd samples/vision/fasterrcnn/runtime/cpp
cmake -S . -B build/x5 -DFRCNN_TARGET=x5 -DCMAKE_BUILD_TYPE=Release
cmake --build build/x5 -j2
```
S100/S100P/S600 将构建目录与 `FRCNN_TARGET` 中的 `x5` 改为对应板卡，选择该板卡的 DNN/UCP 后端。

<a id="run"></a>
## 运行

在仓库根目录下载 X5 模型。下载环境使用 Python 3.10 或更高版本；其他板卡将 `x5` 换为 `s100`、`s100p` 或 `s600`。
```bash
python3 -m venv .venv-model-download
. .venv-model-download/bin/activate
python3 -m pip install PyYAML==6.0.3
python3 samples/vision/fasterrcnn/model/download.py --target x5
```
在板卡的 `samples/vision/fasterrcnn/runtime/cpp` 目录运行：
```bash
./build/x5/fasterrcnn
./build/x5/fasterrcnn --target x5 \
  --parta-path ../../model/x5/frcnn_mbv3_320_parta.bin \
  --partb-path ../../model/x5/frcnn_mbv3_320_partb.bin \
  --input ../../test_data/bus.jpg --save-path result.jpg
./build/x5/fasterrcnn --target x5 \
  --parta-path ../../model/x5/frcnn_mbv3_320_parta.bin \
  --partb-path ../../model/x5/frcnn_mbv3_320_partb.bin \
  --video ../../../bytetrack/test_data/track_test.mp4 \
  --save-path people.mp4 --records people.tsv
```
默认命令使用描述文件中的模型组合，自行编译的模型一起指定两个路径。Shell 入口可合并构建与运行：`bash run.sh --target x5 -- --list-models`。

<a id="parameters"></a>
## 参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--target` | `string` | `auto` | `auto`, `x5`, `s100`, `s100p`, `s600` |
| `--parta-path` | `path` | `None` | 第一段模型 |
| `--partb-path` | `path` | `None` | 第二段模型，与 PartA 一起指定 |
| `--input` | `path` | `samples/vision/fasterrcnn/test_data/bus.jpg` | 图片，配合 `--track` 时为视频 |
| `-i` | `path` | `samples/vision/fasterrcnn/test_data/bus.jpg` | `--input` 的简写 |
| `--video` | `path` | `None` | 视频输入，执行行人跟踪 |
| `--track` | `bool` | false | 跟踪输入视频 |
| `--reset-every` | `int` | `None` | 重新开始跟踪的正整数帧间隔 |
| `--records` | `path` | `None` | 轨迹 TSV 路径 |
| `--no-encode` | `bool` | false | 不生成标注视频 |
| `--save-path` | `path` | `samples/vision/fasterrcnn/test_data/result.jpg` | 输出路径，视频使用 `.mp4` 后缀 |
| `--threads` | `int` | `None` | 图像转换线程数，可选 1、2、4 |
| `--label-file` | `path` | `samples/vision/fasterrcnn/model/coco91_labels.txt` | COCO 类别文件 |
| `--list-models` | `bool` | false | 查看板卡与本地模型路径 |
| `--dry-run` | `bool` | false | 查看模型选择结果 |

`None` 表示未指定参数：模型路径从所选板卡目录获取，线程数沿用 OpenCV 默认值。表中默认文件路径相对仓库根目录。通过 `--help` 或 `-h` 查看命令帮助。

<a id="interface-lifecycle"></a>
## 代码调用

公开接口位于 `inc/detect.hpp`，应用链接 `fasterrcnn_core` 与 OpenCV。下面的示例从仓库根目录运行：
```cpp
#include "detect.hpp"
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <iostream>

int main() {
  fasterrcnn::FasterRCNN::Config cfg;
  cfg.target = "x5";
  cfg.parta_path = "samples/vision/fasterrcnn/model/x5/frcnn_mbv3_320_parta.bin";
  cfg.partb_path = "samples/vision/fasterrcnn/model/x5/frcnn_mbv3_320_partb.bin";
  cfg.layout = fasterrcnn::PartbLayout::kPooled;
  fasterrcnn::FasterRCNN model(cfg);
  cv::Mat image = cv::imread("samples/vision/fasterrcnn/test_data/bus.jpg");
  if (image.empty()) throw std::runtime_error("bus.jpg");
  if (!image.isContinuous()) image = image.clone();
  fasterrcnn::FasterRCNN::Input input;
  input.rows = image.rows;
  input.cols = image.cols;
  input.bgr.assign(image.data, image.data + image.total() * image.elemSize());
  const auto result = model.predict(input);
  std::cout << result.size() << " detections\n";
  return 0;
}
```
分阶段调用使用 `preprocess(input)`、`infer(prepared)` 和 `postprocess(outputs, prepared.context)`。构造模型时加载两段文件，销毁时释放 Runtime 资源。`PersonTracker` 使用期间保留检测器实例，并按顺序处理帧；每个独立视频流使用一组模型和跟踪器。

<a id="results-interpretation"></a>
## 输出

图片 JSON 包含原图像素 xyxy 坐标、置信度、COCO 类别 ID、模型哈希及图像路径。视频输出 MP4，可通过 `--records` 输出 `frame, track_id, x1, y1, x2, y2, score` 列的 TSV 记录。正常完成返回 0，输入、模型或 Runtime 错误返回 2。视频 `predict` 耗时包含检测和跟踪，跟踪耗时是其中 CPU 轨迹关联的部分。

