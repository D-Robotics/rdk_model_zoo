[English](README.md) | 简体中文

# 模型评估 — DINOv2 ViT-S/14

本页汇总已记录的性能与精度参考，并给出复现其设置所需的板端命令。

<a id="dataset"></a>
## 数据集

runtime 冒烟路径使用仓库 fixture：`samples/vision/dinov2/test_data/dog.jpg` 和 `bus.jpg`。PTQ 报告使用 50 张多样真实校准图，并说明另一组 50 张校准图配合独立导出脚本复现了相同数值。没有提供评估数据集准备脚本或固定数据集压缩包。校准准备由 `conversion/mapper.py` 实现，见 [`../conversion/README_cn.md`](../conversion/README_cn.md)。

```text
# cwd：仓库根目录
samples/vision/dinov2/test_data/dog.jpg
samples/vision/dinov2/test_data/bus.jpg
# 板端 benchmark 输入：与 runtime 契约相同的预处理 float32 tensor
```

<a id="directory"></a>
## 目录结构

```text
evaluator/
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="environment"></a>
## 环境

- 板端记录：RDK S100/Nash-E、S100P/Nash-M、S600/Nash-P，使用 `hrt_model_exec` 或 `hbm_runtime`。
- PTQ 报告：OE 3.7.0、hmct 2.6.5 / hbdk 4.7.5，Nash-E。
- runtime 依赖：板端镜像 `hbm_runtime`；主机工具使用 Python 3.10+（NumPy、OpenCV）。
- 板端镜像、固件和 runtime 版本未固定。

<a id="command"></a>
## 评估命令

以下性能命令在匹配板卡和目标制品上运行时，可复现记录中的线程/核心设置。

```bash
# cwd：目标板上的 samples/vision/dinov2/evaluator；制品已准备在 ../model/
# S100 / Nash-E
hrt_model_exec perf --model_file ../model/nash-e/dinov2_vits14_224_int16_nashe.hbm --thread_num 1
hrt_model_exec perf --model_file ../model/nash-e/dinov2_vits14_224_int16_nashe.hbm --thread_num 2

# S100P / Nash-M
hrt_model_exec perf --model_file ../model/nash-m/dinov2_vits14_224_int16_nashm.hbm --thread_num 1
hrt_model_exec perf --model_file ../model/nash-m/dinov2_vits14_224_int16_nashm.hbm --thread_num 2

# S600 / Nash-P
hrt_model_exec perf --model_file ../model/nash-p/dinov2_vits14_224_int16_nashp.hbm --thread_num 1
hrt_model_exec perf --model_file ../model/nash-p/dinov2_vits14_224_int16_nashp.hbm --thread_num 12 --core_id 1,2,3,4
# 预期：锁定 performance governor 后，输出 200 帧的 BPU 延迟/吞吐
```

精度评估需在主机用 ONNXRuntime 对相同预处理输入运行 float ONNX，在板端用 `hbm_runtime.HB_HBMRuntime(...).run` 运行 HBM，再分别计算 `cls_feat`、`patch_feat` 的 cosine。源 CLI 双图路径见 [`../runtime/python/README_cn.md`](../runtime/python/README_cn.md)。

| 参数 | 类型 | 示例默认值 | 说明 |
| --- | --- | --- | --- |
| `thread_num` | int | 基线记录为 `1` | `hrt_model_exec perf` 工作线程数。 |
| `core_id` | CSV ints | 未设置；S600 高并发记录除外 | S600 12 线程记录使用 `1,2,3,4`。 |
| `frames` | int | 源记录为 `200` | 性能测量范围。 |
| `input` | tensor | `(1,3,224,224)` F32 | runtime 预处理生成的 RGB 归一化 tensor。 |

<a id="metrics"></a>
## 指标

| 指标 | 定义 | 条件 |
| --- | --- | --- |
| BPU 延迟 | 一次模型调用的纯 BPU 前向延迟。 | 200 帧，锁 performance governor；板卡及线程/核心设置见表。 |
| BPU 吞吐 | `hrt_model_exec perf` 报告的每秒帧数。 | 同一 200 帧运行；并发数按行给出。 |
| Calibrated cosine | 校准/工具链输出与 float 参照之间每个 output 的 cosine。 | PTQ 报告，Nash-E，featuremap float32 输入、全 int16、默认 KL 校准。 |
| Quantized cosine | 量化 output 与 float ONNX output 之间每个 output 的 cosine。 | PTQ 报告，仅 Nash-E；分别测量 `cls_feat`、`patch_feat`。 |
| 板端 cosine 范围 | 板端 output 与 float ONNX 参照对拍的最小/最大 cosine。 | 相同预处理；分别记录 S100、S100P、S600。 |

标准预处理为 OpenCV BGR→RGB，bicubic 将短边 resize 到 256，中心 crop 224，`/255`，ImageNet mean/std，contiguous float32 NCHW。比较发生在任何 softmax 或 L2 操作之前。

<a id="outputs"></a>
## 输出

runtime CLI 输出 JSON 统计并可按精确路径保存 NumPy tensor。raw 数组对照保存完整 input/raw/result 数组和 comparison.json。ONNX 精度需另行分别统计 cls_feat/patch_feat 的 cosine（无捆绑实现），用上文的板端复现命令执行。

<a id="reference-results"></a>
## 参考结果

以下表格列出全部已记录的行和列。来源：S 平台 evaluator README 及 S 发布 benchmark 记录对应条目。

### 性能记录

| Device | Model | Input Size | BPU Task Latency / BPU Throughput |
|---|---|---|---|
| RDK S100 | dinov2_vits14_224_int16 | 1x3x224x224 | 3.73 ms / 267.44 FPS (1 thread) <br> 288.26 FPS (2 threads) |
| RDK S100P | dinov2_vits14_224_int16 | 1x3x224x224 | 3.02 ms / 329.53 FPS (1 thread) <br> 357.63 FPS (2 threads) |
| RDK S600 | dinov2_vits14_224_int16 | 1x3x224x224 | 2.25 ms / 441.64 FPS (1 thread) <br> 1898.42 FPS (12 threads, `--core_id 1,2,3,4`) |

模型参数量：22.06 M。延迟为纯 BPU 前向，CPU 预处理另计。

### PTQ 逐输出 cosine——仅 Nash-E

此量化质量表仅属于 Nash-E 工具链报告，不应泛化到 S100P 或 S600。

| Output | Calibrated Cosine | Quantized Cosine |
|---|---|---|
| cls_feat | 0.9990 | 0.9989 |
| patch_feat | 0.9985 | 0.9983 |

记录说明，独立导出脚本和另一组 50 张校准图得到相同数值。

### 板端 cosine 对拍 float ONNX

| Device | cls_feat | patch_feat |
|---|---|---|
| RDK S100 | 0.9987 - 0.9989 | 0.9977 - 0.9986 |
| RDK S100P | 0.9987 - 0.9989 | 0.9977 - 0.9986 |
| RDK S600 | 0.9988 - 0.9989 | 0.9975 - 0.9986 |

<a id="boundaries"></a>
## 适用范围

- 本目录没有独立评估实现；复现使用 `hrt_model_exec`、ONNXRuntime、`hbm_runtime` 和 runtime CLI。
- 所有参考数值均为记录的 benchmark；当前板端测量请运行上文性能命令。
- PTQ 量化质量表明确仅适用于 Nash-E；板端 cosine 范围按 target 分别记录。
- DINOv2 在此作为视觉特征编码器，不覆盖文本编码、分类标签、检索数据集或 C++ 评估。

## 许可

评估文档遵循仓库 [LICENSE](../../../../LICENSE) 的 Apache-2.0。
