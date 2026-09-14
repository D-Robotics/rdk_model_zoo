[English](./README.md) | [简体中文](./README_cn.md)

# YOLO26 Depth

本示例提供 Ultralytics YOLO26 单目深度模型在 RDK X5 上的模型转换、评测工具、板端性能数据以及 Python/C++ 推理实现。

## 算法介绍

YOLO26 Depth 根据单张 RGB 图像预测稠密相对深度。编译后的 BPU 模型输出尺寸为 `1×192×192×1` 的 calibrated log-depth；指数解码、双线性插值和 letterbox 还原在 CPU 上执行。

参考资料：[Ultralytics Depth](https://docs.ultralytics.com/tasks/depth/)

## 功能特点

- 支持 YOLO26n、YOLO26s、YOLO26m、YOLO26l 和 YOLO26x Depth。
- 提供 PyTorch 权重 → ONNX → PTQ → RDK X5 BIN 的完整转换流程。
- 提供预处理和后处理一致的 Python、C++ 推理实现。
- 提供单图数值对比和 SUN RGB-D 子集评测工具。

## 平台兼容性

| 平台 | Runtime 模型 | Python | C++ |
| --- | --- | --- | --- |
| RDK X5 | `.bin` | 支持 | 支持 |

运行时已在 RDK X5 BSP 3.5.0-beta、DNN Runtime 1.24.5 上验证。Python 推理必须使用 BSP 自带且与当前 `libdnn` 匹配的 `hbm_runtime`。

## 目录结构

```text
yolo26_depth/
├── conversion/           # ONNX 导出、校准数据准备和 PTQ 脚本
├── evaluator/            # 数值对比和 SUN RGB-D 评测工具
├── model/                # 模型下载脚本和模型元数据
├── runtime/
│   ├── cpp/              # C++ 推理实现
│   └── python/           # Python 推理实现
├── test_data/            # 示例输入图片
├── README.md
└── README_cn.md
```

## 快速运行

`model/download_model.sh` 使用正式 archive 下载地址。下载默认模型：

```bash
bash model/download_model.sh n
```

使用默认参数运行 Python 推理：

```bash
cd runtime/python
bash run.sh
```

使用默认参数运行 C++ 推理：

```bash
cd runtime/cpp
bash run.sh
```

两个脚本默认使用 YOLO26n 768 模型和 `test_data/bus.jpg`。也可以指定模型、输入图片和输出目录：

```bash
bash run.sh MODEL.bin INPUT.jpg OUTPUT_DIR
```

## 模型转换

ONNX 导出、校准数据准备、PTQ 配置和 Mapper 执行方法见 [conversion/README_cn.md](conversion/README_cn.md)。

## 模型推理

- [Python 推理](runtime/python/README_cn.md)
- [C++ 推理](runtime/cpp/README_cn.md)

Python 推理输出 `log_depth.npy`、`depth_native.npy`、`depth.png`、`overlay.png` 和 `report.json`。C++ 推理输出 `depth_native.f32`、`depth.png`、`overlay.png` 和 `report.json`。

## 模型评估与性能

RDK X5 板端性能数据、单图数值对比和 SUN RGB-D 子集评测方法见 [evaluator/README_cn.md](evaluator/README_cn.md)。当前样例尚未发布板端精度数据；在具备已验证输出后，可使用 evaluator 工具生成。

## 源码文档

按照[源码文档说明](../../../docs/source_reference/README.md)生成并查看接口参考文档。

## 注意事项

- 模型输出为相对深度，不是经过标定的绝对米制深度。
- 示例图片仅用于功能验证，不用于精度评测。
- 生成的模型、数据集、日志和评测输出不得提交到样例目录。

## 许可证

示例代码遵循仓库许可证。Ultralytics 模型和 SUN RGB-D 数据仍分别遵循其上游许可证。
