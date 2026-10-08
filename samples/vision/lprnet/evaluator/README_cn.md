# LPRNet 评估器

<a id="dataset"></a>

## 数据集

源没有精度数据集或标签文件。可复现输入是 `../test_data/test_input.dat`（float32
`1x3x24x94`），即源 runtime 读取的同一个预打包张量；`../test_data/example.jpg` 只是
视觉参考。它是单个输入 fixture，不是车牌精度基准集。

<a id="directory"></a>
## 目录结构

```text
evaluator/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── compare.py  # Python 脚本
└── source_reference.py  # Python 脚本
```

<a id="environment"></a>
## 环境

`compare.py` 在板端自行运行两边：锁定的原始 脚本（从 Git 历史加载）与本 sample
`samples/vision/lprnet/runtime/python` 的任务。它需要 X5
runtime（在身份 gate 之后惰性导入）、已准备的 `lpr.bin` 与 `.dat` 输入。主机上
不导入 SDK，也不下载任何内容。源记录性能保留如下。

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

源参考为上表 `lpr.bin` 行（100 帧）。板端对照流程：在同一块 X5 上用内置
`test_input.dat` 运行两套实现；通过要求全部检查为 true，且输入张量、raw
`(1,68,18,1)` logits 与解码车牌的 `max_abs_diff` 均为 0.0。板端日志加载
`lpr.bin` 时会打印 HBRT 库与模型构建小版本不一致的警告；该警告不影响对照。
结果是单个输入上的数值一致性，不是精度基准。

<a id="boundaries"></a>
## 适用范围

评估器自行运行两边，绝不用人工提供的文件替代真实推理。它不下载模型、不准备精度
数据集、也不测性能。已记录运行覆盖一块 X5 8GB 与一块 X5 4GB 的内置输入
（见上方参考结果）；其它板卡或输入仍需各自运行，且不从这些对照得出
车牌识别精度结论。
