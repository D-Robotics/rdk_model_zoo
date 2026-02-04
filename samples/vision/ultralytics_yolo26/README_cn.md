# YOLO26 模型说明

[English](./README.md) | 简体中文

## 算法介绍
YOLO26 是一系列通用且高性能的实时模型。本示例提供了在 D-Robotics RDK 硬件上运行多种任务的部署例程，包括：目标检测 (Detection)、实例分割 (Segmentation)、姿态估计 (Pose)、旋转框检测 (OBB) 以及图像分类 (Classification)。

更多算法细节请参考 [Ultralytics](https://github.com/ultralytics/ultralytics) 官方资源。

## 目录结构
```bash
.
├── conversion/     # 模型转换流程说明 (ONNX -> BIN)
├── evaluator/      # 模型评估相关内容
├── model/          # 模型文件及下载脚本
├── runtime/        # 推理示例 (Python / C++)
│   └── python/     # Python 推理实现
├── test_data/      # 测试用数据（图片/标签）
└── README.md       # 当前模型总览说明
```

## 快速体验
如果您想快速体验 YOLO26 模型，可以在 RDK 开发板上直接运行推理脚本。

### Python
1. 确保开发板已安装必要的依赖。
2. 运行统一的推理入口：
```bash
cd runtime/python
python3 main.py --task detect --model-path ../../model/yolo26n_bpu_bayese_640x640_nv12.bin --test-img ../../test_data/bus.jpg
```
关于参数说明和环境配置的更多细节，请参考 [runtime/python/README.md](./runtime/python/README.md)。

## 模型转换
我们提供了已经转换好的 BPU 模型。如果您需要转换自定义模型：
1. 使用转换工具将模型导出为 ONNX 格式。
2. 使用工具链将 ONNX 转换为 `.bin` 格式。
详细步骤请参考 [conversion/README.md](./conversion/README.md)。

## 模型推理 (Runtime)
本示例为多种任务提供了标准化的推理封装：
- **Python**: 基于 `pyeasy_dnn` 后端实现（采用规范的 Config 和 Model 类结构）。
- **C++**: (即将推出)。

详细说明请查阅：
- [Python 推理说明](./runtime/python/README.md)

## 推理结果
运行示例后，推理结果将保存为图片（例如 `result.jpg`），根据任务不同展示检测框、分割掩码或人体关键点。

## 模型评估 (Evaluator)
`evaluator/` 目录包含了用于评估模型精度、性能及数值一致性的脚本。您可以直接在开发板上运行这些脚本，以获取模型在标准数据集（如 COCO, ImageNet）上的 mAP 和准确率等指标。

详细评估指南请参考：[模型评估说明](./evaluator/README_cn.md)

## License
本示例遵循 [Apache 2.0 License](../../../LICENSE)。