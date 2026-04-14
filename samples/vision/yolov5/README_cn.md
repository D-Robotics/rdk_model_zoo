# 简体中文 | [English](./README.md)

# YOLOv5 模型说明

本目录给出 YOLOv5 在本 Model Zoo 中的完整使用方式，包括算法简介、模型转换、Python 运行时推理、评测说明以及参考性能数据。

---

## 算法简介

YOLOv5 是 YOLO 系列中的单阶段目标检测算法，直接在多尺度特征图上预测边界框与类别分数，在通用目标检测任务中兼顾了精度与运行效率。

### 算法功能

YOLOv5 当前支持的任务：

- 目标检测

### 原始资源

- 官方仓库：[ultralytics/yolov5](https://github.com/ultralytics/yolov5)

---

## 目录结构

```bash
.
├── conversion/                        # 模型转换说明与 yaml 配置
│   ├── yolov5_detect_bayese_640x640_nchw.yaml
│   ├── yolov5_detect_bayese_640x640_nv12.yaml
│   ├── README.md
│   └── README_cn.md
├── evaluator/                         # 精度与性能评测说明
│   ├── README.md
│   └── README_cn.md
├── model/                             # 模型文件与下载脚本
│   ├── download.sh
│   ├── README.md
│   └── README_cn.md
├── runtime/                           # 推理示例
│   ├── cpp/
│   └── python/
│       ├── main.py
│       ├── yolov5_det.py
│       ├── run.sh
│       ├── README.md
│       └── README_cn.md
├── test_data/                         # 示例图片与 benchmark 参考资源
│   ├── bus.jpg
│   └── *.png / *.jpg
├── README.md
└── README_cn.md
```

---

## 快速开始

Python 示例提供了默认 `run.sh`，可快速完成一次推理验证：

```bash
cd runtime/python
chmod +x run.sh
./run.sh
```

详细运行说明请参考 [runtime/python/README_cn.md](./runtime/python/README_cn.md)。

---

## 模型转换

Model Zoo 已提供可直接运行的 `.bin` 模型文件，用户可以在 `model/` 目录下通过下载脚本直接获取并运行。

如果需要从 YOLOv5 项目重新完成导出与转换，请参考 [conversion/README_cn.md](./conversion/README_cn.md)。文档包含以下内容：

- `v2.0` 与 `v7.0` 两条分支的环境准备
- ONNX 导出时的输出头修改方式
- `hb_mapper checker` 与 `hb_mapper makertbin`
- `hb_perf` 与 `hrt_model_exec` 的验证方法
- 当前 Python runtime 使用的输入输出 tensor 协议

---

## 运行时推理

YOLOv5 当前提供 Python 版本推理示例。

### Python 版本

- 以脚本形式提供，适合快速验证模型效果与算法流程
- 覆盖模型加载、推理执行、后处理与结果可视化完整流程
- 详细参数说明和接口定义请参考 [runtime/python/README_cn.md](./runtime/python/README_cn.md)

---

## 模型评测

`evaluator/` 目录用于整理 benchmark 数据、板端验证说明以及评测方法，详见 [evaluator/README_cn.md](./evaluator/README_cn.md)。

---

## 性能数据

下表给出 YOLOv5 在 RDK X5 平台上的参考性能数据。

| 模型 | 分辨率 | 参数量 | BPU 吞吐 | Python 后处理 |
| --- | --- | ---: | --- | --- |
| YOLOv5s_v2.0 | 640x640 | 7.5 M | 106.8 FPS | 12 ms |
| YOLOv5m_v2.0 | 640x640 | 21.8 M | 45.2 FPS | 12 ms |
| YOLOv5l_v2.0 | 640x640 | 47.8 M | 21.8 FPS | 12 ms |
| YOLOv5x_v2.0 | 640x640 | 89.0 M | 12.3 FPS | 12 ms |
| YOLOv5n_v7.0 | 640x640 | 1.9 M | 277.2 FPS | 12 ms |
| YOLOv5s_v7.0 | 640x640 | 7.2 M | 124.2 FPS | 12 ms |
| YOLOv5m_v7.0 | 640x640 | 21.2 M | 48.4 FPS | 12 ms |
| YOLOv5l_v7.0 | 640x640 | 46.5 M | 23.3 FPS | 12 ms |
| YOLOv5x_v7.0 | 640x640 | 86.7 M | 13.1 FPS | 12 ms |

---

## License

遵循仓库顶层 License。
