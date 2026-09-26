[English](README.md) | [简体中文](README_cn.md)

# 深度离线评估

本目录准备评估输入并比较**已保存数组**，不会加载 BIN/HBM 或执行板端模型。
运行时、数据集精度和延迟是不同声明。本轮只有主机测试数据验证，没有新增板测或 SUNRGBD 实测。

<a id="dataset"></a>
## 数据集与输入身份

请自行准备 SUN RGB-D 图像和以米为单位的真值 `.npy` 数组，仓库不附带数据集。
源 JSON 清单包含 `records`，例如：

```json
{
  "records": [
    {"index": 7, "sensor": "sensor_name", "image": "images/0007.png",
     "image_hw": [480, 640], "depth_m": "depth/0007.npy"}
  ]
}
```

图像和深度路径相对 `--source-root`。提供 `image_hw` 时会核对真实尺寸。
已有非负整数编号保持不变；没有编号时使用源列表位置，重复编号报错。
准备输入时允许暂缺真值，但之后不能对该条目计算真值精度。

保留三种源评估协议：

| 协议 | 准备的输入 | 还原与真值处理 |
|---|---|---|
| `deployment_letterbox` | RGB CHW uint8 `.bin`，768 letterbox，填充 114 | 已校准 log-depth → exp → 放大 768 → 去 padding → 原图 H×W；使用原始真值 |
| `deployment_scale_fill` | RGB NCHW float32 `/255` `.npy`，直接拉伸到 768 | 原始 logit → clip/scale/bias → exp → 原图 H×W；使用原始真值 |
| `ultralytics_validator` | RGB CHW uint8 `.bin`，源长边缩放后再次拉伸到方形 | 已校准 log-depth → exp → 768 方形；真值按相同两级尺寸用最近邻缩放 |

这些 RGB 二进制是准备好的模型输入，**不是运行时打包 NV12**。
归一化和模型输入适配必须与输出数组的生产过程一致，不能因为形状相同就比较不同协议生成的数组。

<a id="environment"></a>
## 环境

默认主机路径需要 Python、NumPy 和 OpenCV，不导入板端 SDK。
数据集指标保留源公式，以 float64 累积。
可选 `--resize-backend torch` 保留源评估器 `align_corners=False` 插值，仅使用时导入 Torch；
默认 OpenCV 插值与统一运行时一致。不声明新的数值累积与源 Torch float32 归约逐位相等。
本轮环境未运行可选 Torch 后端。

以下命令从本 `evaluator/` 目录执行。数据、报告和图像写入样例目录之外。
每次准备、单图比较使用新目录，数据集评估报告也不能覆盖已有文件。

<a id="command"></a>
## 命令与输出数组格式

```bash
python prepare_sunrgbd.py --source-root /work/sunrgbd \
  --source-manifest /work/sunrgbd/source.json --output /work/depth/prepared \
  --size 768 --screen-count 20 --screen-seed 20260726
```

默认生成三种协议，可重复指定 `--protocol` 只准备所需项以减少存储。
每个张量记录摘要、几何和类型。screen 子集按 sensor 确定性分层抽样；数量小于 sensor 数量
或为零也受支持。screen 是选择元数据，评估器不会据此静默丢弃其他输出记录。

请在匹配的导出或运行环境中分别生成浮点和编译模型的原始输出，本工具不代为推理。
保存两个 NPZ，包含一维、唯一整数 `indices` 和按该顺序排列的数组：

| 边界 | 参考数组键 | 候选数组键 | 每条记录形状 |
|---|---|---|---|
| `log` | `float_log` | `quant_log` | `[192,192]`、`[1,192,192,1]` 或 `[1,1,192,192]` |
| `raw` | `float_raw` | `quant_raw` | 同上 |

数值必须是深度还原之前的有限浮点输出。运行时 `log_depth.npy` 可以作为 **log** 记录，
lite 的 `raw_logit.npy` 可以作为 **raw** 记录。
`depth_native.npy` 已经解码，不能填入这两种原始输出位置。
以下仅打包已有的一对结果，不生成浮点参考推理：

```python
import numpy as np
reference = np.load('/work/depth/reference_log_depth.npy', allow_pickle=False)
candidate = np.load('/work/depth/candidate_log_depth.npy', allow_pickle=False)
np.savez('/work/depth/float.npz', indices=np.array([7], dtype=np.int64),
         float_log=reference[None])
np.savez('/work/depth/quant.npz', indices=np.array([7], dtype=np.int64),
         quant_log=candidate[None])
```

使用实际准备清单中的编号。参考与候选编号集合必须完全一致，且全部存在于准备清单。

```bash
python eval_sunrgbd.py --prepared-manifest /work/depth/prepared/manifest.json \
  --source-root /work/sunrgbd --float-outputs /work/depth/float.npz \
  --quant-outputs /work/depth/quant.npz --candidate-name x5-n-candidate \
  --protocol deployment_letterbox --boundary log --variant n \
  --report /work/depth/evaluation.json
```

S n/s/m 同样使用 letterbox/log。S l/x 则选择 `deployment_scale_fill`、
`--boundary raw`、对应 `--variant` 及 raw NPZ 键。
保留的实验性 S n/s/m lite 输出可按源校准系数使用 raw 协议，但不是已发布默认运行方案。
`--boundary log` 不会再应用这些校准系数。变体和几何声明必须与真实生产过程匹配；
评估器不能仅凭任意保存的数组独立推断其来源。

比较两个**已还原到原图尺寸**的深度图并生成统一色域展示：

```bash
python eval_numeric.py --image /work/depth/input.png \
  --official /work/depth/reference_depth.npy --candidate /work/depth/candidate_depth.npy \
  --reference-name fp32-reference --candidate-name s600-l-candidate \
  --output /work/depth/single-image
```

`--x5` 保留为 `--candidate` 别名，不再把所有候选写死成 X5 n。
`--official` 只是参考文件路径，并非真实性认证。
两个数组必须与图像形状一致，且所有值有限并大于 1e-6。

<a id="metrics"></a>
## 指标与对齐

数据集评估输出 delta1/2/3（阈值 `1.25`、`1.25²`、`1.25³`）、AbsRel、RMSE 和 SILog。
真值仅在有限且 `0.001 < depth < 100` 时有效；预测在可选逐图对齐后截断到 `[0.001,100]`。
指标按所有选中图像的有效像素汇总，不是每张图无权重平均。

保留不对齐和 median 对齐两组指标。数据集对齐使用偶数样本的**较小中位数**，
与源 `torch.median` 一致；单图比较保留源 NumPy 的**两个中间值平均数**。
二者可能给出不同尺度，报告会明确各自定义。

原始输出和还原深度的 fidelity 包含 MAE、RMSE、最大绝对误差、相对误差和 cosine。
空数组、非有限数值、形状或样本集合不匹配都会报错，不会静默删除问题值。
零范数 cosine 为 `null`；无有效真值时指标为 `null` 且 `valid_pixels=0`，不是完美成绩。
工具不会仅凭 cosine 自动判定候选验收通过。

<a id="outputs"></a>
## 输出文件

准备阶段生成各协议张量和 `manifest.json`，记录源清单、图像、张量摘要、编号、几何和 screen 选择。
数据集评估生成一个 JSON，包含两个模型对真值的指标、候选减参考的差值、原始/深度 fidelity、
逐图结果与输入数组文件摘要。board 字段明确本离线工具没有执行板测。

单图比较生成 `comparison-report.json`、`candidate_depth_median_aligned.npy`、
统一色域的深度/叠加 PNG、绝对相对误差 PNG，以及 1620×586 的 `comparison.jpg` 拼图。
两张深度图都采用参考的 2%/98% 分位色域，颜色不表示米制距离。
文件名使用 reference/candidate，不再把所有输入错误标注为 X5。

<a id="reference-results"></a>
## 源历史结果——未重测

X5 源记录为 OE 1.2.8 / Mapper 1.24.3、768 输入、max percentile 0.9999、
O3 latency 和尾部卷积 int16。HRT 数值只包含模型执行：

| 变体 | X5 单线程延迟 ms | 单线程 FPS | 双线程总 FPS |
|---|---:|---:|---:|
| n | 23.194 | 43.085 | 45.682 |
| s | 36.168 | 27.637 | 28.615 |
| m | 60.783 | 16.449 | 16.751 |
| l | 75.336 | 13.272 | 13.470 |
| x | 161.022 | 6.210 | 6.253 |

S 源 evaluator 的 HRT 延迟表如下，单位 ms：

| 变体 | S100 | S100P | S600 |
|---|---:|---:|---:|
| n | 3.165 | 2.254 | 1.760 |
| s | 4.490 | 3.244 | 2.363 |
| m | 8.246 | 5.986 | 4.062 |
| l | 9.790 | 7.090 | 4.881 |
| x | 19.059 | 12.853 | 9.097 |

该文档的示例选择 lite n，而发布方案 n/s/m 为 NV12；源记录未逐行绑定制品摘要，
因此不能将此表重新标为已确认的发布制品延迟。
其 log cosine 0.9985–0.9998 声明也不同于根 README 混合方案表；根表 s=0.9984
又低于同页“全部通过”的 0.999 门槛。这些冲突保留在
[源审计](../../../../docs/releases/unified-migration/2026-09-26-b8-yolo26-depth-source-review.md)。
表格不代表 SUNRGBD 数据集精度或本轮验收结论。

<a id="boundaries"></a>
## 验证边界

主机测试覆盖 raw/log 正确解码、统一运行时一致性、较小中位数与像素汇总、
非法数据/编号拒绝、三种准备协议、小数量 screen 抽样及实际报告/图像写入。
这些测试不证明 Torch 后端一致性、真实模型精度、OE 编译、板端行为或性能。
可选 Torch 插值、真实数据集运行和新增板测均为 not-run。
原始输出数组及摘要不能单独证明生成它们的模型、前处理或板卡，请与生产过程证据一起保存。
