[English](./README.md) | 简体中文

# ByteTrack 模型转换说明

ByteTrack 本身不包含需要编译为 HBM 的神经网络，运行时依赖前级检测模型。本 sample 使用 YOLOv5x 行人检测模型 `yolov5x_672x672_nv12.hbm`，跟踪逻辑在 Python 中完成。

## 检测模型

Runtime 使用的检测模型为：

```text
samples/vision/bytetrack/model/s100/yolov5x_672x672_nv12.hbm
```

模型下载脚本：

```bash
cd samples/vision/bytetrack/model
bash download_model.sh s100
```

该检测模型来自 RDK S100 `ultralytics_YOLO` 模型资源，输入为 NV12 双输入：

- Y plane
- UV plane

ByteTrack 仅消费检测模型输出的行人框、置信度和类别 ID。当前示例只跟踪 COCO 类别 `person`（class id 0）。

## 转换参考

YOLO 检测模型的 ONNX 导出、PTQ 配置生成和模型编译流程请参考 Ultralytics YOLO sample 的转换说明，以及 OE 包中的示例内容。

可参考的模型族包括 YOLOv5u、YOLOv8、YOLO11、YOLO12 等检测模型。将其他检测模型接入 ByteTrack 时，需要保证 runtime 输出的检测框格式为 `(x1, y1, x2, y2, score, class_id)` 或在 wrapper 中转换为等价格式。

## OE 资源

模型转换请在 x86 Linux 主机的 RDK S100 OpenExplore 环境中完成，不建议在板端执行转换。

- OE 资源入口（docker+OE开发包）：<https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview>
- OE 工具链在线手册：<https://toolchain.d-robotics.cc/>

## License

本目录遵循 [Apache 2.0 License](../../../../LICENSE)。
