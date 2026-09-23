[English](./README.md) | 简体中文

# YOLOv5

<a id="overview"></a>
## 算法与来源

YOLOv5 是单阶段 anchor 检测器，在三个特征尺度上预测目标框、置信度和类别。源工程为 [ultralytics/yolov5](https://github.com/ultralytics/yolov5)；本 sample 保留 X5 与 S 两套不同的输入协议。

<a id="support-matrix"></a>
## 支持与验证矩阵

| target | variant | Python | C++ | 状态 |
|---|---|---|---|---|
| X5 | n-v7.0、s/m/l/x-v2.0、s/m/l/x-v7.0 | supported-not-run | supported-not-run | 主机契约 fixture 通过；板测未运行 |
| S100 | x-672 | supported-not-run | supported-not-run | 主机契约 fixture 通过；板测未运行 |
| S100P | — | not-supported | not-supported | manifest 没有 YOLOv5 制品 |
| S600 | x-672 | supported-not-run | supported-not-run | 主机契约 fixture 通过；板测未运行 |

Python X5 使用单个 packed NV12 输入且默认直接拉伸；S Python 使用拆分的 Y/UV 输入且默认 letterbox。C++ 有独立源默认值，详见 `runtime/cpp/`；这里不把任一语言写成已板测。

<a id="prerequisites"></a>
## 环境前提

主机检查需要 Python 3、NumPy、OpenCV；manifest 工具还需要 PyYAML。板端推理需要匹配目标系统和 `hbm_runtime`；S 的跟踪/检测 CPU 后处理还需要 SciPy、`lap==0.5.12` 和 `cython-bbox==0.1.5`。转换需要对应 RDK OpenExplorer。active manifest 中发布 SHA-256 均未知。

<a id="quickstart"></a>
## 快速体验

在仓库根目录显式准备制品，再运行 Python 检测器。下面使用 X5 `n-v7.0`；本轮没有下载或运行：

```bash
python3 -m samples.vision.yolov5.model.download \
  --target x5 --variant n-v7.0 \
  --output-dir samples/vision/yolov5/model
python3 -m samples.vision.yolov5.runtime.python.main \
  --target x5 --variant n-v7.0
```

第一条命令写入 `samples/vision/yolov5/model/yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin` 并打印观测 digest；发布 SHA-256 未知。第二条读取 `test_data/bus.jpg`，写入 `test_data/result_unified.jpg`，打印检测数组，成功退出码为 `0`。S100 可将两条命令改为 `--target s100 --variant x-672`；模型路径是 `model/s100/yolov5x_672x672_nv12.hbm`，默认图片是 `test_data/kite.jpg`。

`--list-models` 只读 manifest；`--dry-run` 必须显式指定 target，只输出协议，不加载 SDK 或模型。runtime 和 `run.sh` 都不会下载或安装依赖。

<a id="expected-results"></a>
## 预期结果

成功的 Python 运行打印 `boxes`、`scores`、`class_ids` 三个 JSON 数组，并保存带框图片。boxes 是原图 XYXY，scores 在 `[0,1]`，class IDs 是 COCO 类别编号。X5 逆变换后框会截断为整数；S 保留浮点坐标。无检测时形状为 `(0,4)`、`(0,)`、`(0,)`。具体数量由模型和输入决定，本说明不编造。

<a id="directory"></a>
## 目录职责

```text
.
├── model/                 # 显式 manifest 制品准备
├── runtime/python/        # binding、NV12 I/O、task、runner、CLI、可视化
├── runtime/cpp/           # 目标相关 C++ 实现与 README
├── conversion/            # 源 YAML 与导出/编译限制
├── evaluator/             # 同板源/统一证据对照器
├── test_data/             # bus/kite、标签和历史结果制品
└── tests/                 # 主机数值、metadata、CLI、评估 fixture
```

<a id="entry-points"></a>
## 入口索引

- [`model/README_cn.md`](./model/README_cn.md)：X5/S 制品、URL、路径和校验值。
- [`runtime/python/README_cn.md`](./runtime/python/README_cn.md)：parser、三阶段和目标 tensor 契约。
- [`runtime/cpp/README_cn.md`](./runtime/cpp/README_cn.md)：现有 C++ 构建运行说明，由独立维护面负责。
- [`conversion/README_cn.md`](./conversion/README_cn.md)：逐字节保留的 X5 YAML 与源转换命令。
- [`evaluator/README_cn.md`](./evaluator/README_cn.md)：完整 raw 数组源/统一对照。

<a id="historical-performance"></a>
## 源历史性能

下表完整保留源 X5 参考数据；这些是历史测量，本轮没有复测：

| 模型 | 尺寸 | 参数量 | BPU 吞吐 | Python 后处理 |
|---|---|---:|---:|---:|
| YOLOv5s_v2.0 | 640x640 | 7.5 M | 106.8 FPS | 12 ms |
| YOLOv5m_v2.0 | 640x640 | 21.8 M | 45.2 FPS | 12 ms |
| YOLOv5l_v2.0 | 640x640 | 47.8 M | 21.8 FPS | 12 ms |
| YOLOv5x_v2.0 | 640x640 | 89.0 M | 12.3 FPS | 12 ms |
| YOLOv5n_v7.0 | 640x640 | 1.9 M | 277.2 FPS | 12 ms |
| YOLOv5s_v7.0 | 640x640 | 7.2 M | 124.2 FPS | 12 ms |
| YOLOv5m_v7.0 | 640x640 | 21.2 M | 48.4 FPS | 12 ms |
| YOLOv5l_v7.0 | 640x640 | 46.5 M | 23.3 FPS | 12 ms |
| YOLOv5x_v7.0 | 640x640 | 86.7 M | 13.1 FPS | 12 ms |

<a id="license"></a>
## 许可

仓库 wrapper 和源 sample 文档遵循 Apache-2.0。上游 YOLOv5 工程及下载权重遵循各自许可和来源；manifest `sha256: null (unknown)` 不代表认证。
