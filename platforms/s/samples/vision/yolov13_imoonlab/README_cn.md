[English](./README.md) | 简体中文

# YOLOv13 iMoonLab 模型说明

本目录提供 YOLOv13 iMoonLab 检测 sample 在 Model Zoo 中的完整使用说明，包括算法介绍、模型转换、模型推理、模型文件管理和评估说明。

## 算法介绍

YOLOv13 是清华大学智能媒体与认知实验室推出的新一代实时目标检测模型，核心设计包括 HyperACE、FullPAD 以及轻量级卷积替换，在 COCO 等通用目标检测任务上兼顾精度、速度和参数效率。

- 官方仓库：<https://github.com/iMoonLab/yolov13>
- 论文：<https://arxiv.org/abs/2506.17733>
- 项目主页：<https://www.gaoyue.org/>

![YOLOv13 icon](test_data/icon.png)

![YOLOv13 framework](test_data/framework.png)

## 算法功能

- 目标检测

## 算法特点

- 仅保留 YOLOv13 Detect 任务，运行接口清晰。
- Python runtime 采用 `Config + Wrapper + predict()` 结构。
- HBM 模型输入固定为 NV12 双输入：Y plane + UV plane。
- 后处理按固定 tensor 索引解析，不做输入输出结构猜测。

## 目录结构

```bash
.
├── conversion/              # 模型导出、校准与编译说明
├── evaluator/               # 精度与性能评估说明
├── model/                   # 模型下载脚本与模型说明
├── runtime/
│   └── python/              # Python 推理示例
├── test_data/               # 测试图片、标签与文档图片
├── README.md
└── README_cn.md
```

## 快速体验

进入 `runtime/python/` 目录后运行：

```bash
cd runtime/python
bash run.sh
```

默认会检查并下载 `../../model/s100/yolo13n_detect_nashe_640x640_nv12.hbm`，然后使用 `../../test_data/kite.jpg` 完成一次目标检测。

如需手动指定模型和图片，请参考 [runtime/python/README_cn.md](./runtime/python/README_cn.md)。

## 模型转换

YOLOv13 的 ONNX 导出、校准数据准备和 HBM 编译说明见 [conversion/README_cn.md](./conversion/README_cn.md)。

## 模型推理

Python 运行时使用 `hbm_runtime`，详细参数和直接入口命令见 [runtime/python/README_cn.md](./runtime/python/README_cn.md)。

## 模型评估

性能测试表、精度表和评测方法见 [evaluator/README_cn.md](./evaluator/README_cn.md)。

## 性能数据

本 sample 支持以下参考模型：

- `yolo13n_detect_nashe_640x640_nv12.hbm`
- `yolo13s_detect_nashe_640x640_nv12.hbm`
- `yolo13l_detect_nashe_640x640_nv12.hbm`
- `yolo13x_detect_nashe_640x640_nv12.hbm`

对应的性能和精度数据维护在 [evaluator/README_cn.md](./evaluator/README_cn.md)。

## License

遵循仓库顶层 `LICENSE`。
