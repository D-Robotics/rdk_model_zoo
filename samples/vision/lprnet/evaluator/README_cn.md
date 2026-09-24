# LPRNet 评估器

<a id="dataset"></a>
## 数据集

源没有精度数据集或标签文件。可复现输入是 `../test_data/test_input.dat`（float32
`1x3x24x94`），即固定源读取的同一个预打包张量；`../test_data/example.jpg` 只是
视觉参考。它是单个输入 fixture，不是车牌精度基准集。

<a id="environment"></a>
## 环境

`compare.py` 在板端自行运行两边：`platforms/x5/samples/vision/lprnet/runtime/python`
的固定源 wrapper 与 `samples/vision/lprnet/runtime/python` 的统一任务。它需要 X5
runtime（在身份 gate 之后惰性导入）、已准备的 `lpr.bin` 与 `.dat` 输入。主机上
不导入 SDK，也不下载任何内容。源历史性能保留如下，本轮未复测。

| 模型 | 测试帧数 | FPS | 平均延迟 | BPU 使用率 | ION 内存 |
|---|---:|---:|---:|---:|---:|
| `lpr.bin` | 100 | 266 FPS | 3.75 ms | 9% | 1.11 MB |

<a id="command"></a>
## 评估命令

在已准备资产与输入的板卡上，于仓库根目录运行：

```bash
python3 samples/vision/lprnet/evaluator/compare.py \
  --target x5 \
  --output-dir /tmp/lprnet-compare-$(date -u +%Y%m%dT%H%M%SZ)
```

用 `--asset-id x5:lprnet:lpr.bin --model-path <file>` 对照外部准备的资产，用
`--input-dat <file>` 指向其它打包输入。输出目录必须尚不存在。

<a id="metrics"></a>
## 指标

评估器记录两边实际产生的输入张量、raw float32 logits 与解码车牌，并报告每个张量
的 shape/dtype/finite 检查与 `max_abs_diff`。输入与解码车牌要求完全相等；raw
logits 使用 `atol=1e-5`。仅当全部检查通过时返回 `0`，两边不一致返回 `1`，运行
失败返回 `2`。没有标签数据集，因此不报告精度分数。

<a id="outputs"></a>
## 输出

`comparison.json` 用 SHA-256 绑定 `target`、`asset_id`、模型/输入/代码摘要，记录
两边的实际 runtime metadata、`argv`、`cwd`、`started_utc`/`finished_utc` 与
`return_code`，并逐个列出保存的数组文件及其自身摘要。每个输入、raw logits 与结果
数组都写入同目录下的独立 `.npy` 文件。执行失败时仍写出带 `error`、
`return_code: 2`、`passed: false` 的 manifest。

<a id="reference-results"></a>
## 参考结果

源历史参考为上表 `lpr.bin` 行（100 帧）。本轮未运行板端对照，因此当前板端状态为
`not-run`，sample 仍为 `closed=no`；评估器的主机 fixture 只证明证据结构，不代表
推理数值。

<a id="boundaries"></a>
## 边界

评估器自行运行两边，绝不用人工提供的文件替代真实推理。它不下载模型、不准备精度
数据集、不测性能，也不宣称板端兼容；在真正记录同板源/统一运行之前，结果保持
`not-run`。
