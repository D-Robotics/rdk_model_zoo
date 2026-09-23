# LPRNet 评估器

<a id="dataset"></a>
## 数据集

源没有精度数据集或标签文件。可复现 smoke 输入是 `../test_data/test_input.dat`（float32 `1x3x24x94`），`../test_data/example.jpg` 只是视觉参考。它是单个输入 fixture，不是车牌精度基准集。

<a id="environment"></a>
## 环境

评估器可在装有 Python 和 NumPy 的主机运行，不导入 `hbm_runtime`。生成 raw 证据需要在相同模型和输入条件下分别运行 X5 源 runtime 与统一 runtime。下表完整保留源历史行，且本轮没有复测。

| 模型 | 测试帧数 | FPS | 平均延迟 | BPU 使用率 | ION 内存 |
|---|---:|---:|---:|---:|---:|
| `lpr.bin` | 100 | 266 FPS | 3.75 ms | 9% | 1.11 MB |

<a id="command"></a>
## 评估命令

先分别保存源 runtime 和统一 runtime 的完整 float32 raw 输出到唯一文件。源路径为 `platforms/x5/samples/vision/lprnet/runtime/python`，统一路径为 `samples/vision/lprnet/runtime/python`。然后在仓库根目录运行：

```bash
python3 samples/vision/lprnet/evaluator/compare.py \
  --legacy-raw /tmp/lprnet-run-legacy/raw.bin \
  --unified-raw /tmp/lprnet-run-unified/raw.bin \
  --shape 1 68 18 \
  --output /tmp/lprnet-run-unified/compare.json
```

输入必须是完整 raw float32 数组。只有 shape、dtype、每个 raw 值和 CTC 解码车牌都一致时退出码才为 `0`，否则为 `2`。本轮没有运行板端对照。

<a id="metrics"></a>
## 指标

主要一致性指标是 raw float32 logits 的精确 `array_equal`，其次是源 CTC 风格解码车牌的精确相等。没有标签数据集，因此不报告精度分数。上面的性能数值是保留测试条件的源历史结果。

<a id="outputs"></a>
## 输出

`compare.py` 打印并可写出包含 shape、dtype、raw 相等性、两侧解码字符串和状态的 JSON。两个 raw 数组和报告应放在唯一 run 目录中，不覆盖旧证据。

<a id="reference-results"></a>
## 参考结果

源 evaluator 的历史参考见上表。当前主机及板端对照状态为 `not-run`。

<a id="boundaries"></a>
## 边界

本评估器只比较已保存的完整证据，不下载模型、不加载 SDK、不创建标签基准，也不宣称板端兼容。即使主机 fixture 对照成功，在记录同板源/统一运行前仍是 `supported-not-run`。
