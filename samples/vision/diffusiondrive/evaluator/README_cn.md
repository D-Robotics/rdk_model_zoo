[English](README.md) | [简体中文](README_cn.md)

# DiffusionDrive 输出对照

本工具离线比较浮点模型参考结果与运行时解码预测，不运行板卡，不对标注真值评分，不计算 NAVSIM PDM Score，也不据此批准模型用于驾驶。

<a id="dataset"></a>
## 参考数据

保留六组确定性输入/参考对：默认 `reference_inputs.npz` / `reference_outputs.npz`，以及五个 `case_*` 目录。形状、场景和历史图片见[测试数据说明](../test_data/README_cn.md)。每份参考含 float32 的 `trajectory`、`agent_states`、`agent_labels`（logits）和 `bev_semantic_map`（logits）。它们是模型输出，不是数据集标签。

噪声是显式输入。比较时应使用相同输入归档与固定噪声，重新生成噪声会改变规划问题。完整 NAVSIM 评估还需要本 sample 未提供的场景日志、传感器数据、地图与指标缓存。

<a id="environment"></a>
## 环境

评估器只需要 Python 和 NumPy，不导入 hbm_runtime，也不需要 HBM。先在相应目标上通过 [Python 运行入口](../runtime/python/README_cn.md)生成候选结果，再复制完整运行目录进行离线分析。`--board-npz` 沿用源接口名称；使用此参数本身不证明文件来自板端实测。

<a id="command"></a>
## 命令

从仓库根目录，将已保存的单案例结果与对应浮点参考比较：

```bash
python3 -m samples.vision.diffusiondrive.evaluator.compare_outputs --reference-npz samples/vision/diffusiondrive/test_data/reference_outputs.npz --board-npz outputs/diffusiondrive/outputs.npz --output outputs/diffusiondrive-comparison.json
```

比较批量结果时，两侧选择相同案例：

```bash
python3 -m samples.vision.diffusiondrive.evaluator.compare_outputs --reference-npz samples/vision/diffusiondrive/test_data/case_017/reference_outputs.npz --candidate-npz outputs/diffusiondrive_cases/case_017/outputs.npz --output outputs/diffusiondrive-case017-comparison.json
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--reference-npz` | 必填 | 含四个源输出名称的浮点参考归档 |
| `--board-npz` / `--candidate-npz` | 必填 | 解码后的六数组 `outputs.npz`，不是 `raw_outputs.npz` |
| `--output` | `null` | 可选的新 JSON 报告路径；否则仅 stdout |

拒绝覆盖已有报告。返回 0 表示输入有效且指标计算完成，**不表示**通过验收门槛。输入契约不符时返回 2。

<a id="metrics"></a>
## 指标与校验

比较前，两份归档必须精确匹配名称、类型和形状。浮点字段须为有限 float32，`agent_mask` 为 bool，`bev_labels` 为 uint8。候选概率限定在 [0,1]；BEV 标签限定在 0..6，并须等于候选 BEV logits 的 argmax。在任何 NumPy 运算前检查形状相等，修复源实现可能通过广播将不同维数标签图判为完全一致的问题。

| 参考字段 | 候选字段 | 比较方式 |
| --- | --- | --- |
| `trajectory` `[1,8,3]` | `trajectory` | 展平余弦、MAE、最大绝对误差 |
| `agent_states` `[1,30,5]` | `agent_states` | 同上 |
| `agent_labels` `[1,30]` | `agent_scores` | 先对参考 logits 使用源裁剪 sigmoid，再比较 |
| `bev_semantic_map` `[1,7,128,256]` | `bev_logits` | 张量指标，再比较类别 argmax 图 |

MAE 和最大误差使用 float64 累积。余弦使用展平后的 float64 向量；任一范数为零时 JSON 记为 `null` 并明确未定义，不通过给分母加小量将其当作普通分数。非零向量的有限结果裁剪到 [-1,1]，处理舍入误差。参考 sigmoid 将 logits 裁剪至 [-60,60]。

BEV 指标包括像素一致率、类别分布、各类 IoU，以及在任一预测中出现的类别（含背景）的宏平均 IoU。两侧都不存在的类别不参与平均。少量稀有静态物体像素可明显影响宏平均 IoU，却几乎不改变像素一致率，应同时保留两者。Agent mask 只做契约检查，不评分：单个归档无法确定阈值设置，因此必须保留运行报告。

<a id="outputs"></a>
## 报告与证据

JSON 包含参考/候选文件的精确路径和 SHA-256、张量与 BEV 指标、零范数策略，以及 `status: descriptive; no acceptance threshold`，并明确 `dataset_accuracy: false`。请同时保留运行时 `report.json`、物理输入、原始输出和图片。对照报告只能绑定两份文件；如果丢弃运行证据，它无法重建模型、运行库和输入来源。

<a id="reference-results"></a>
## 源分支历史结果

下列记录来自 S 源文档，**不是**本次主机迁移重新测得。精度对照使用 `case_000`；性能使用有效的量化 `case_017` 输入、固定一个 BPU 核与 200 帧。

| 指标 | S100P | S600 |
| --- | ---: | ---: |
| 轨迹余弦 | 0.999857 | 0.999833 |
| Agent 状态余弦 | 0.996879 | 0.997052 |
| BEV 余弦 | 0.998913 | 0.998918 |
| BEV 像素一致率 | 0.943726 | 0.944061 |
| BEV 平均 IoU | 0.865501 | 0.868425 |
| 单线程延迟 | 14.370 ms | 7.215 ms |
| 单线程吞吐 | 69.375 FPS | 138.247 FPS |
| 双线程平均任务延迟 | 28.024 ms | 13.856 ms |
| 双线程总吞吐 | 71.109 FPS | 143.767 FPS |
| CPU 推理时间 | 0.0 ms | 0.0 ms |

源 S100P 五案例均值：轨迹余弦 0.999785、Agent 状态余弦 0.997986、BEV 余弦 0.998799、像素一致率 0.955664、平均 IoU 0.819837。这些是历史对照结果，不是门槛，也不是标注数据集精度。源文档称模型完全运行于 BPU，本次没有新 profiling 验证该声明。

历史 HRT profiling 的 `--thread_num` 表示并发提交任务的主机线程数，不是 CPU 核数。不能用双线程总吞吐的倒数作为单请求延迟。源 S600 各案例结果及稀有类别解释保留于测试数据说明。

<a id="boundaries"></a>
## 验证边界

主机测试检查严格输入契约、零范数余弦未定义、参考自比较与文件摘要记录。自比较只验证评估器，不是量化模型精度证据。六案例任务后处理和绘图对照使用随附浮点数组，与源 Python 实现比较。实际板端推理、OE 转换、数据集评分、完整 NAVSIM PDM Score 和运行性能仍为 **not-run**。不会根据历史表格静默推导默认容差或发布门槛。
