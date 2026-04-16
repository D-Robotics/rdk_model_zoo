[English](./README.md) | 简体中文

# YOLO26 模型说明

本目录给出了 YOLO26 在当前 Model Zoo 中的完整使用说明，包括算法概览、模型转换、运行时推理、模型文件管理和评测脚本使用方式。

---

## 算法概览

YOLO26 是 Ultralytics 提供的一套实时视觉模型系列。本示例在 RDK X5 平台上提供了以下任务的部署样例：

- 目标检测
- 实例分割
- 姿态估计
- 旋转框检测
- 图像分类

- **官方实现**: [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

### 平台说明

- 目标平台：`RDK X5`
- 推理后端：`hbm_runtime`
- 推理模型格式：`.bin`
- 输入格式：`NV12`

---

## 目录结构

```bash
.
├── conversion/                     # 模型转换流程
├── evaluator/                      # 精度评测和结果导出脚本
├── model/                          # 模型文件和下载脚本
│   ├── download_model.sh           # 下载 nano 模型
│   ├── fulldownload.sh             # 下载全部模型
│   └── README.md                   # 模型文件说明
├── runtime/                        # 运行时样例
│   └── python/                     # Python 推理样例
│       ├── main.py                 # Python 入口脚本
│       ├── yolo26_det.py           # 检测封装
│       ├── yolo26_seg.py           # 分割封装
│       ├── yolo26_pose.py          # 姿态封装
│       ├── yolo26_obb.py           # 旋转框封装
│       ├── yolo26_cls.py           # 分类封装
│       ├── run.sh                  # 一键运行脚本
│       └── README.md               # 运行时说明
├── test_data/                      # 推理结果目录
└── README.md                       # 当前总览文档
```

---

## 快速开始

如果只是想快速体验，可以直接运行 `runtime/python` 下的一键脚本。

### Python

```bash
cd runtime/python
chmod +x run.sh
./run.sh
```

脚本会在模型不存在时自动下载默认的 `yolo26n` 检测模型，并将结果图保存到 `test_data/`。

详细参数和任务示例请参考 [runtime/python/README_cn.md](./runtime/python/README_cn.md)。

---

## 模型转换

本示例已经提供了适配 RDK X5 的 `.bin` 模型文件。

- 如果只关注推理，可以直接使用 [model/README_cn.md](./model/README_cn.md) 中的下载脚本，跳过转换流程。
- 如果需要了解或自定义模型转换，可参考 [conversion/README_cn.md](./conversion/README_cn.md)。

---

## 运行时推理

当前示例提供 Python 版本运行时实现。

### Python 版本

- 使用 `hbm_runtime` 作为推理后端
- 所有任务统一采用 `Config + Model` 的封装方式
- `main.py` 支持零参数默认运行

详细使用方法请参考 [runtime/python/README_cn.md](./runtime/python/README_cn.md)。

---

## 评测

`evaluator/` 目录用于任务级精度评测和结果导出验证，具体使用方式请参考 [evaluator/README_cn.md](./evaluator/README_cn.md)。

---

## 验证状态

当前 Python 样例已经在 `RDK X5` 板端完成以下 `.bin` 模型验证：

- `detect`: `n / s / m / l / x`
- `seg`: `n / s / m / l / x`
- `pose`: `n / s / m / l / x`
- `obb`: `n / s / m / l / x`
- `cls`: `n / s / m / l / x`

---

## License

遵循 Model Zoo 顶层 License。
