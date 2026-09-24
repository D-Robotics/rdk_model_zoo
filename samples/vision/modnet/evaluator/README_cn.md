# MODNet 评估器

<a id="dataset"></a>
## 数据集

源没有抠图 benchmark 数据集或 ground-truth alpha mask。`../test_data/person.jpg` 和
`../test_data/bg.jpg` 是一组推理/合成 fixture，不是精度数据集。

<a id="environment"></a>
## 环境

`compare.py` 在板端自行运行两边：`platforms/x5/samples/vision/modnet/runtime/python`
的固定源 wrapper 与 `samples/vision/modnet/runtime/python` 的统一任务。它需要 X5
runtime（在身份 gate 之后惰性导入）、手工准备的 `modnet_512x512_rgb.bin` 与一张
BGR 图像。主机上不导入 SDK，也不下载任何内容。下表和条件完整保留源历史数据，
本轮未复测。

| 模型 | 尺寸 | 输入格式 | 延迟 (ms) | FPS |
|---|---|---|---:|---:|
| MODNet | 512x512 | Float32 NCHW RGB | 89.88 | 11.12 |
| MODNet（2 threads） | 512x512 | Float32 NCHW RGB | 130.49 | 15.27 |

条件：RDK X5、CPU 8xA55@1.8G、BPU 1xBayes-e@1G（10TOPS INT8）；单线程延迟使用
单帧、单线程、单 BPU core，多线程 FPS 使用 2 个并发线程。

<a id="command"></a>
## 评估命令

在已准备手工资产的板卡上，于仓库根目录运行：

```bash
python3 samples/vision/modnet/evaluator/compare.py \
  --target x5 \
  --output-dir /tmp/modnet-compare-$(date -u +%Y%m%dT%H%M%SZ)
```

用 `--asset-id x5:modnet:modnet_512x512_rgb.bin --model-path <file>` 指向手工资产，
用 `--test-img <file>` 换其它图像。输出目录必须尚不存在。ref-size 固定为 512，
与部署 metadata 一致。

<a id="metrics"></a>
## 指标

评估器记录两边的预处理 NCHW 输入张量、raw float32 matte 与原图几何下的最终 uint8
matte，并报告每个张量的 shape/dtype/finite 检查与 `max_abs_diff`。输入与最终 matte
要求完全相等；raw matte 使用 `atol=1e-5`。仅当全部检查通过时返回 `0`，出现差异返回
`1`，运行失败返回 `2`。源没有 ground truth，因此不报告数据集 SAD/MAD/IoU。

<a id="outputs"></a>
## 输出

`comparison.json` 用 SHA-256 绑定 `target`、`asset_id`、模型/图像/代码摘要，记录两边
的实际 runtime metadata、`argv`、`cwd`、`started_utc`/`finished_utc` 与 `return_code`，
并逐个列出保存的数组文件及其摘要。每个输入、raw matte 与结果 matte 都写入独立
`.npy`。执行失败时仍写出带 `error`、`return_code: 2`、`passed: false` 的 manifest。
合成图是独立的视觉制品，不参与 matte tensor 对照。

<a id="reference-results"></a>
## 参考结果

源历史参考见上表。本轮未运行板端对照，因此源/统一板端和精度评估为 `not-run`，
sample 仍为 `closed=no`；主机 fixture 只证明证据结构。

<a id="boundaries"></a>
## 边界

评估器自行运行两边，绝不用人工提供的 matte 文件替代真实推理。它不是人像数据集
评估器，不虚构质量标签，不下载手工资产，也不会把主机 fixture 结果升级为板端验证。
