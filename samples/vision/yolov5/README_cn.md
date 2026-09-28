[English](./README.md) | 简体中文

# YOLOv5

<a id="overview"></a>
## 算法与来源

YOLOv5 是单阶段 anchor 检测器：CSPDarknet backbone 加 FPN+PAN 特征融合，三个特征尺度（stride 8/16/32）的检测头输出 COCO 风格的框、置信度和类别。`n/s/m/l/x` 规格在速度与精度间取舍。源工程为 [ultralytics/yolov5](https://github.com/ultralytics/yolov5)；本 sample 保留 X5 与 S 两套不同的输入协议。

<a id="support-matrix"></a>
## 支持与验证矩阵

| target | variant | Python | C++ | 状态 |
|---|---|---|---|---|
| X5 | n-v7.0、s/m/l/x-v2.0、s/m/l/x-v7.0 | supported-verified | supported-not-run | Python：九个变体全部在 8GB+4GB 源/统一对照。C++：无数值对照；smoke 记录仅覆盖 `s-v2.0` 默认——见表下说明 |
| S100 | x-672 | supported-verified | supported-not-run | Python：`x-672` `kite.jpg` 对照。C++：无数值对照；有 `x-672` 构建/smoke 记录——见表下说明 |
| S100P | — | not-supported | not-supported | manifest 没有 YOLOv5 制品；仅有拒绝路径记录 |
| S600 | x-672 | supported-verified | supported-not-run | Python：`x-672` `kite.jpg` 对照。C++：无数值对照；有 `x-672` 构建/smoke 记录——见表下说明 |

`supported-verified`（Python）表示 2026-09-24 记录的同板源/统一对照（板测固定提交 `ae0f185`/`4d45f9a`），归档于 [python 对照](../../../docs/releases/unified-migration/evidence/2026-09-24-b7-python-comparison/)、[X5 九变体矩阵](../../../docs/releases/unified-migration/evidence/2026-09-24-b7-yolov5-x5-variants/)与[扩展板测](../../../docs/releases/unified-migration/evidence/2026-09-24-b7-expanded-boards/)证据：X5 覆盖一个 8GB 和一个 4GB 板上的全部九个发布变体（`bus.jpg`），S100/S600 覆盖 `x-672` `kite.jpg` case。它们只验证这些制品/图片 case，不是 COCO 精度结论，而且属于历史记录——本工作树不对当前 HEAD 追加任何新的板端验证。C++ 列为 `supported-not-run`：尚未在任何板端记录源/统一数值对照。另有 2026-09-24 真实板端的编译+推理 smoke 记录，但只覆盖 C++ 默认变体——X5 8GB/4GB 上的 `s-v2.0` 制品、以及 S100（首轮编译失败修复后）和 S600 的 `x-672` 制品，dump 已归档（[首轮](../../../docs/releases/unified-migration/evidence/2026-09-24-b7-board-initial/)、[第二轮](../../../docs/releases/unified-migration/evidence/2026-09-24-b7-native-round2/)、[X5 4GB/S600](../../../docs/releases/unified-migration/evidence/2026-09-24-b7-expanded-boards/)）；smoke 不构成任何变体的数值验证，不延伸到其余 X5 C++ 变体，也不得出任何精度或性能结论。S100P 的拒绝路径记录（[负例证据](../../../docs/releases/unified-migration/evidence/2026-09-24-b7-s100p-negative/)）不是正向支持。

Python X5 使用单个 packed NV12 输入且默认直接拉伸；S Python 使用拆分的 Y/UV 输入且默认 letterbox。C++ 有独立源默认值，详见 `runtime/cpp/`。

<a id="prerequisites"></a>
## 环境前提

主机检查需要 Python 3、NumPy、OpenCV；manifest 工具还需要 PyYAML。板端推理需要匹配目标系统和 `hbm_runtime`；S 的跟踪/检测 CPU 后处理还需要 SciPy、`lap==0.5.12` 和 `cython-bbox==0.1.5`。转换需要对应 RDK OpenExplorer。active manifest 中发布 SHA-256 均未知。

<a id="quickstart"></a>
## 快速体验

在仓库根目录显式准备制品，再运行 Python 检测器。下面使用 X5 `n-v7.0`——2026-09-24 X5 8GB/4GB 记录中的默认制品/图片 case，当时由同一个下载器准备制品并完成完整源/统一对照（[扩展板测证据](../../../docs/releases/unified-migration/evidence/2026-09-24-b7-expanded-boards/)）；下面这组 argv 本身不是当时的板端命令，本工作树也不做新的下载或板端运行：

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
