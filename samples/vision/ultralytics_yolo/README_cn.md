[English](./README.md) | 简体中文

# Ultralytics YOLO 模型说明

本目录给出 Ultralytics YOLO sample 在 Model Zoo 中的完整使用说明，包括算法概览、
模型转换、运行时推理、模型文件管理和评测说明。

---

## 算法概览

Ultralytics YOLO 是一套覆盖目标检测、实例分割、人体姿态估计和图像分类的实时视觉
模型家族。本 sample 在 RDK X5 平台上提供以下模型家族的部署示例：

- 检测：
  `YOLOv5u / YOLOv8 / YOLOv9 / YOLOv10 / YOLO11 / YOLO12 / YOLO13`
- 实例分割：
  `YOLOv8 / YOLOv9 / YOLO11`
- 姿态估计：
  `YOLOv8 / YOLO11`
- 图像分类：
  `YOLOv8 / YOLO11`

- **官方实现**: [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

### 平台说明

- 目标平台：`RDK X5`
- 推理后端：`hbm_runtime`
- 推理模型格式：`.bin`
- 输入格式：packed `NV12`

---

## 目录结构

```bash
.
|-- conversion/                     # 模型转换流程
|-- evaluator/                      # 精度和性能评测文档
|-- model/                          # 模型文件和下载脚本
|   |-- download_model.sh           # 下载默认模型
|   |-- fulldownload.sh             # 下载全部支持模型
|   |-- README.md                   # 英文模型文件说明
|   `-- README_cn.md                # 中文模型文件说明
|-- runtime/                        # 运行时样例
|   |-- cpp/                        # C++ 参考运行时
|   `-- python/                     # Python 推理样例
|       |-- main.py                 # Python 入口脚本
|       |-- ultralytics_yolo_det.py # 检测封装
|       |-- ultralytics_yolo_seg.py # 分割封装
|       |-- ultralytics_yolo_pose.py# 姿态封装
|       |-- ultralytics_yolo_cls.py # 分类封装
|       |-- run.sh                  # 一键运行脚本
|       |-- README.md               # 英文运行时说明
|       `-- README_cn.md            # 中文运行时说明
|-- test_data/                      # 测试图片和结果图
|-- README.md                       # 英文总览文档
`-- README_cn.md                    # 中文总览文档
```

---

## 快速开始

如需快速体验，可直接运行 `runtime/python` 下的一键脚本。

### Python

```bash
cd runtime/python
chmod +x run.sh
./run.sh
```

默认命令会在模型不存在时自动下载 `yolo11n_detect_bayese_640x640_nv12.bin`，
并将结果图保存到 `test_data/`。

详细参数和任务示例请参考
[runtime/python/README_cn.md](./runtime/python/README_cn.md)。

---

## 模型转换

本 sample 提供了适用于 RDK X5 的预编译 `.bin` 模型。

- 如果只需要运行推理，可直接从 `model/` 目录下载模型并跳过转换流程。
- 如果需要导出 ONNX、准备校准数据或重新编译模型，请参考
  [conversion/README_cn.md](./conversion/README_cn.md)。

---

## 运行时推理

本 sample 提供 Python 和 C++ 两个运行时目录。

### Python 版本

- 使用 `hbm_runtime` 作为推理后端
- 所有任务共用统一入口 `main.py`
- 每个任务均采用 `Config + Wrapper + predict()` 组织方式

详细使用方式请参考
[runtime/python/README_cn.md](./runtime/python/README_cn.md)。

### C++ 版本

C++ 目录作为参考实现保留，详见
[runtime/cpp/README_cn.md](./runtime/cpp/README_cn.md)。

---

## 评测

`evaluator/` 目录用于保存支持模型的 benchmark 表、精度参考和运行验证记录。

详细内容请参考 [evaluator/README_cn.md](./evaluator/README_cn.md)。

---

## 验证状态

本目录中的 Python sample 已在 `RDK X5` 上完成以下模型的板端验证：

- Detect:
  `YOLOv5u / YOLOv8 / YOLOv9 / YOLOv10 / YOLO11 / YOLO12 / YOLO13`
- Seg:
  `YOLOv8 / YOLOv9 / YOLO11`
- Pose:
  `YOLOv8 / YOLO11`
- CLS:
  `YOLOv8 / YOLO11`

详细 benchmark 和验证汇总维护在 `evaluator/README_cn.md` 中。

---

## License

遵循 Model Zoo 顶层 License。
