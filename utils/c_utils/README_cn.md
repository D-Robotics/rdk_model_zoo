[English](README.md) | 简体中文

# C++ 推理公共函数


本目录为原生 Model Zoo Sample 提供图像与张量处理、结果绘制、板卡识别和文件哈希计算。

## 目录结构

```text
c_utils/
├── inc/                   # 公共头文件
├── src/                   # 函数实现
├── platform_identity.h    # 板卡信息类型与目标解析
├── platform_identity.cc   # 读取板卡信息
└── sha256.h               # SHA-256 计算
```

| 头文件 / 实现 | 职责 |
| --- | --- |
| `file_io.hpp` / `.cpp` | 图片与标签读取 |
| `model_types.hpp` | 分类、检测和关键点结构 |
| `nn_math.hpp` / `.cpp` | Sigmoid、Softmax、归一化 |
| `preprocess.hpp` / `.cpp` | 缩放、letterbox、色彩转换与张量准备 |
| `postprocess.hpp` / `.cpp` | Top-K、反量化、解码、NMS 与坐标映射 |
| `visualize.hpp` / `.cpp` | 分类、框、掩码、关键点与文字绘制 |
| `runtime.hpp` | HB-DNN 与 HB-UCP 返回码处理 |

## 使用方式

进入 Sample 的 `runtime/cpp` 目录，执行其文档提供的 CMake 命令。各 Sample 的 CMakeLists 选择需要的公共头文件和源文件，并链接目标 SDK 与 OpenCV。具体配置见 [ResNet C++ 文档](../../samples/vision/resnet/runtime/cpp/README_cn.md)和 [YOLO C++ 文档](../../samples/vision/ultralytics_yolo/runtime/cpp/README_cn.md)。

图像与结果处理使用 `inc/` 中的头文件。板卡识别需包含 `platform_identity.h` 并编译 `platform_identity.cc`，通过 `rdk::read_native_identity()` 和 `rdk::identify_target()` 读取并解析目标。文件哈希使用 `sha256.h`。函数签名及张量说明见[源码参考文档](../../docs/source_reference/README.md)。
