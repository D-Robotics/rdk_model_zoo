[English](README.md) | 简体中文

# PointNet Python 运行时

<a id="overview"></a>
## Python 推理

使用 PointNet 将原始 XYZ 椅子点云分割为四类部件。

<a id="directory"></a>
## 目录结构

```text
python/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── cli.py  # 参数、模型选择与结果交付
├── main.py  # 命令行入口：构造模型并调用 predict
├── pointnet.py  # 模型初始化与推理阶段
├── run.sh  # 运行示例
└── visualization.py  # 结果渲染与图片保存
```

从 [main.py](main.py) 开始：入口构造 `PointNetSegmenter` 并调用 `predict`。[pointnet.py](pointnet.py) 实现模型初始化及推理阶段；[cli.py](cli.py) 负责参数、模型选择和结果交付。模型初始化会加载 Runtime，应用可复用同一个实例执行多次预测。

<a id="environment"></a>
## 环境

Python 3.10+、NumPy、PyYAML；默认绘图另需 matplotlib。真实推理需要 RDK S100 SDK 的
`hbm_runtime`。使用板端镜像提供的 Python 环境。

```bash
# cwd: repository root; board Python environment
python3 -m pip install numpy PyYAML matplotlib
python3 samples/vision/pointnet/runtime/python/main.py --help
```

<a id="usage"></a>
## 使用

直接脚本可从任意 cwd 调用，默认模型和输入按 sample 定位为绝对路径。
仓库根下 `bash samples/vision/pointnet/runtime/python/run.sh` 是等价启动器，会转发所有参数。
```bash
# cwd: repository root; prepare the HBM with model/download.sh first
python3 samples/vision/pointnet/runtime/python/main.py
python3 samples/vision/pointnet/runtime/python/main.py --target s100 --test-pts samples/vision/pointnet/test_data/chair.pts --output-dir outputs/chair --no-plot
python3 samples/vision/pointnet/runtime/python/main.py --dry-run --target s100
```

退出 0 并生成标签/JSON 表示 CLI 完成；错误输出说明并返回 2。Dry-run 不需要 HBM/SDK，
但不验证实际 metadata 或推理。运行时不会自动下载缺失模型。

<a id="parameters"></a>
## 参数

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `--target` | choice | `auto` | 解析为 s100；实际执行还校验本机板卡身份 |
| `--asset-id` | string | `None` | 精确 S100 reference；外部模型路径必须同时提供 |
| `--model-path` | string | `None` | 解析为 sample `model/s100/pointnet.hbm` |
| `--test-pts` | Path | `samples/vision/pointnet/test_data/chair.pts` | 空白分隔 XYZ 文本，每行一点 |
| `--output-dir` | Path | `outputs/pointnet` | 相对 cwd；启动脚本会切换至仓库根 |
| `--no-plot` | flag | `false` | 跳过 matplotlib 和 PNG，保留标签、JSON |
| `--priority` | int | `0` | SDK 优先级 0–255 |
| `--bpu-cores` | int list | `[0]` | 非负核编号；硬件可用性由 SDK 检查 |
| `--list-models` | flag | `false` | 输出清单，不加载 SDK 或下载文件 |
| `--dry-run` | flag | `false` | 仅解析准备信息；与 list-models 互斥 |

`-h` / `--help` 输出帮助后退出。上表默认输入会解析为 sample 下的绝对路径。

<a id="results"></a>
## 结果

`labels.npy`：int32 N 向量，0=back、1=seat、2=leg、3=arm，顺序对应输入行。
`result.json`：目标/制品、输入路径、point_count、counts、centroid、radius、实际张量 metadata。
`result_orig.png` 和 `result.png` 显示归一化点云，图轴保留源 X/Z/Y 排列。
`--no-plot` 不生成这两张图。同名输出会被替换；要保留不同实验，请指定不同输出目录。

<a id="integration-example"></a>
## 库集成

API 接收原始点坐标并在 preprocess 内归一化。使用 NumPy 点数组调用模型；通过 CLI 参数读取文件。

```python
# cwd: repository root; execute on S100 after model preparation
import numpy as np
from samples.vision.pointnet.runtime.python.cli import resolve_selection, SAMPLE_DIR
from samples.vision.pointnet.runtime.python.pointnet import PointNetSegmenter

points = np.loadtxt(SAMPLE_DIR / "test_data/chair.pts", dtype=np.float32)
selection = resolve_selection("s100")
task = PointNetSegmenter(selection)
prepared = task.preprocess(points)
raw = task.infer(prepared.tensors)
labels = task.postprocess(raw)
print(labels.shape, labels.dtype)
# Equivalent three-stage convenience call:
labels_again = task.predict(points)
# 同时保留本次调用的归一化点云与质心/半径上下文时：
details = task.predict(points, return_details=True)
print(details.prepared.tensors["point"].shape, details.prepared.context.radius)
```

`PointNetPredictionDetails`（通过 `return_details=True` 显式开启）将标签与单次调用的
prepared 记录（精确的 `(1,3,N)` 归一化张量加冻结的质心/半径上下文）打包返回，绘图与
归档无需二次执行；默认 `predict` 仍返回普通标签数组，task 不保存上一次点云。
真实 runner 懒加载，
先校验板卡/制品，再导入 SDK、核对 metadata。每个推理线程使用独立 runner，或串行调用共享 runner。

<a id="stage-io"></a>
## 阶段 I/O

| 阶段 | 输入 | 输出与语义 |
| --- | --- | --- |
| preprocess | 有限实数 ndarray `(N,3)` XYZ | 自有连续 float32 `(1,3,N)`；减去质心，再除最大欧氏半径 |
| infer | 使用绑定输入名的张量映射 | 从 runtime 取得自有 raw `(1,N,4)` logits，不做 argmax/反量化/IO |
| postprocess | 与绑定 shape/dtype 一致的 raw 张量 | int32 `(N,)` 标签；整数以 float64 做 SCALE 解码后再 argmax，float32 不变 |
| predict | 原始 `(N,3)` 坐标 | 串联同样阶段和标签结果；`return_details=True` 额外返回单次调用的 prepared 记录 |

发布的 HBM 同时输出 `pred` float32 `[1,N,4]` 椅子部件分数，以及
辅助 XYZ 变换 `trans` float32 `[1,3,3]`。Runtime 校验两个张量后，
按名称解码 `pred`，不依赖输出顺序。也支持只输出部件分数的制品；
其他输出职责或变换形状不满足契约。

既有的 `pre_process`、`forward`、`post_process` 名称保留为 `preprocess`、`infer`、`postprocess` 的可导入薄别名——同一实现，两个名字。

N 来自编译模型 metadata，必须精确匹配，不采样/补点。冻结的 `prepared.context` 保存每次
质心、半径和点数，不会被下一次调用覆盖。后处理无需消费 context，因为点序未变。
整数输出以 float64 做仿射解码，int8 到 int32 的不同 raw 值在 argmax 前保持大小关系；
float32 解码会把大整数舍入成人为平局（独立评审 POINTNET-R2）。只有解码后完全相等的
分数才算平局，取最小 ID。整数输出必须有有限正 SCALE 参数，缺失或无效时拒绝，不猜测。

<a id="troubleshooting"></a>
## 故障排查

- 模型缺失：先执行模型指南中的显式下载。
- Target mismatch/无已发布制品：S100P、S600 不能复用 S100 制品。
- 点数或列数不符：提供精确 N 行 XYZ，不接受法线/颜色列。
- 零半径或非有限输入：完全重合的点或 NaN/Inf 无法归一化。
- 未知输出、形状或 dtype 不符：核对实际制品，不绕过 binding 去运行另一种导出。
- 缺 matplotlib：安装它或使用 `--no-plot`，后者仍保存标签与 JSON。
