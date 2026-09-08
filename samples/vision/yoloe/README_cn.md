简体中文 | [English](./README.md)

# YOLOE 模型说明

本目录给出 YOLOE sample 在 Model Zoo 中的完整使用说明，包括算法概览、模型转换、运行时推理、模型文件管理和评测说明。

本示例提供 YOLOE-11s/m/l Seg Prompt-Free 的转换和 Python 推理流程，输入为 640x640 NV12。

---

## 算法简介

YOLOE（You Only Look Once Everything）是一种零样本、可提示的 YOLO 模型，用于开放词汇检测与分割。YOLOE-11 Seg Prompt-Free 支持 4585 个类别，无需任何文本提示即可直接进行实例分割。

### 算法功能

YOLOE 当前支持的任务：

- 实例分割

### 原始资源

- 论文：[YOLOE: Real-Time Seeing Anything](https://arxiv.org/pdf/2503.07465v1)
- 官方仓库：[um-assn/yoloe](https://github.com/um-assn/yoloe)

---

## 目录结构

```bash
.
├── conversion/                        # 模型转换说明与配置
│   ├── onnx_export/
│   │   └── export_yoloe11seg_bpu.py
│   ├── ptq_yamls/
│   │   └── yoloe_11s_seg_pf_bayese_640x640_nv12.yaml
│   ├── README.md
│   └── README_cn.md
├── evaluator/                         # 精度与性能评测说明
│   ├── README.md
│   └── README_cn.md
├── model/                             # 模型文件与下载脚本
│   ├── download_model.sh
│   ├── README.md
│   └── README_cn.md
├── runtime/                           # 推理示例
│   └── python/
│       ├── main.py
│       ├── yoloe_seg.py
│       ├── run.sh
│       ├── README.md
│       └── README_cn.md
├── test_data/                         # 示例图片与 benchmark 参考资源
│   └── *.jpg / *.png
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

Model Zoo 已提供 YOLOE-11s/m/l Seg-PF 的 `.bin` 下载，用户可以在 [model/README_cn.md](./model/README_cn.md) 中查看下载脚本与模型说明。

如果需要从 YOLOE 项目重新完成导出与转换，请参考 [conversion/README_cn.md](./conversion/README_cn.md)。文档包含以下内容：

- YOLOE 环境准备与适配 BPU 的 ONNX 导出方法
- PTQ 量化校准数据准备
- `hb_mapper makertbin` 转换流程
- 当前 Python runtime 使用的输入输出 tensor 协议

---

## 运行时推理

YOLOE 当前提供 Python 版本推理示例。

### Python 版本

- 以脚本形式提供，适合快速验证模型效果与算法流程
- 覆盖模型加载、推理执行、DFL 框解码、掩码生成、NMS 与结果可视化完整流程
- 详细参数说明和接口定义请参考 [runtime/python/README_cn.md](./runtime/python/README_cn.md)

---

## 模型评测

`evaluator/` 目录用于整理 benchmark 数据、板端验证说明以及评测方法，详见 [evaluator/README_cn.md](./evaluator/README_cn.md)。

---

## 性能数据

| 模型 | 分辨率 | 线程数 | 延迟 | FPS |
| --- | --- | ---: | --- | --- |
| YOLOE-11s-Seg-PF | 640x640 | 1 | 146.16 ms | 6.84 FPS |
| YOLOE-11m-Seg-PF | 640x640 | 1 | 177.14 ms | 5.65 FPS |
| YOLOE-11l-Seg-PF | 640x640 | 1 | 189.97 ms | 5.26 FPS |

---

## License

遵循仓库顶层 License。
