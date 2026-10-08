[English](README.md) | 简体中文

# HGNetV2 评测
本指南介绍 X5 单图功能检查和数据集级 Top-K 评估。数据集命令从 CSV 读取图像路径及逐图真值类别索引。

<a id="dataset"></a>

## 数据集

进行 ImageNet-1k 数据集评测时，准备验证集 JPEG 和 UTF-8 CSV，表头为 `image:file,category`。每行用相对 `--image-path` 的路径关联该图像的模型真值类别索引（0–999）。评测器递归扫描 JPEG/PNG，并在匹配路径时保留子目录。示例布局：

```text
/data/imagenet-val/n01440764/example.JPEG
/data/imagenet-labels.csv:
image:file,category
n01440764/example.JPEG,0
```

示例必须换成真实数据标签。CSV 缺字段、非法或冲突类别会报错，路径反斜杠转换为斜杠。CSV 中未出现的图片计为未匹配，不从目录名猜标签。

<a id="directory"></a>
## 目录结构

```text
evaluator/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
└── eval.py  # Python 脚本
```

<a id="environment"></a>
## 环境

eval.py 在 X5 使用 HGNetV2Classifier 和共享 RuntimeModelRunner，依赖相同板端 SDK、NumPy、OpenCV、PyYAML，无额外推理框架。--help 及纯数据/指标测试不需 SDK。SciPy 仅用于源对照主机测试。

<a id="command"></a>
## 命令

```bash
# cwd: repository root; expected: unittest OK, exit 0
python3 -m unittest discover -s samples/vision/hgnetv2/tests -v
```

板端功能检查：

```bash
# cwd: repository root
bash samples/vision/hgnetv2/model/download.sh x5 b0
python3 samples/vision/hgnetv2/runtime/python/main.py \
  --target x5 --variant b0 \
  --test-img samples/vision/hgnetv2/test_data/sandbar.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

在匹配板卡上对每个已发布变体分别执行。同板多次运行对照时，保持
模型字节、图像、resize 类型、Top-K 和调度参数一致，在标签格式化之前
比较类别 ID 与原始分数；预期类别 ID 相同、分数差在 1e-5 内。当 Top-K
边界出现完全平局时，核对逐 ID 分数而不是放宽容差。

数据集评测（cwd：X5 仓库根）。准备验证图片与 CSV 后运行下方命令。运行时间随图片数变化。命令写入 JSON 报告；结合覆盖率计数查看精度字段。

```bash
bash samples/vision/hgnetv2/model/download.sh x5 b0
python3 samples/vision/hgnetv2/evaluator/eval.py \
  --target x5 --variant b0 \
  --image-path /data/imagenet-val --val-csv /data/imagenet-labels.csv \
  --resize-type 0 --top-k 5 --limit 0 \
  --json-save-path outputs/hgnetv2-b0-evaluation.json
```

| Option | Default | 含义 |
| --- | --- | --- |
| `--target` | auto | 执行目标，仅匹配 X5 才能运行 |
| `--variant` | null | 默认 b0，其余显式指定 |
| `--asset-id` | null | 外部模型路径对应的准确清单引用 |
| `--model-path` | null | 已准备文件，需配合 --asset-id，不自动下载 |
| `--image-path` | required | 验证图像根目录，递归扫描 |
| `--val-csv` | required | 相对路径/类别 CSV |
| `--label-file` | empty string | 可选显示标签，不是 ground truth |
| `--json-save-path` | hgnetv2_cls_results.json | JSON 输出，会覆盖指定文件 |
| `--limit` | 0 | 匹配标签前取排序前 N 张，0=全部 |
| `--top-k / --topk` | 5 | Top-K 精度，1–1000 |
| `--resize-type` | 0 | 直接缩放，1 为 letterbox，与运行示例默认 1 不同 |
| `--priority` | 0 | 调度优先级 |
| `--bpu-cores` | [0] | BPU 核编号 |

<a id="metrics"></a>
## 指标

Top-1 为首位正确数/成功推理数；Top-K 为真值落在 K 个预测内的数量/
成功推理数。分母只统计成功推理，因此必须同时查看缺失/失败计数。
仅 K=5 时写入 top5_acc；topk_acc 对任意 K 都是命名准确的指标。FPS
统计循环中的读图及前处理/推理/后处理，不含模型加载与 CSV/目录扫描，
无预热——不能当作发布表的多线程吞吐。固定图对照使用 ID 一致、分数
绝对差 <1e-5，完全平局须提供逐 ID 证据。

<a id="outputs"></a>
## 输出

写入 --json-save-path 并打印报告。字段包含 status（complete/partial/no-results）、扫描/匹配/未匹配/失败/成功数、逐图错误、精度分母、top1_acc/topk_acc（无成功推理时为 null）、可选 top5_acc、elapsed_seconds/fps、asset_id/target/model、数据路径和配置。complete 只表示本次扫描图片全部评测，不证明已覆盖完整 50,000 张
数据；未匹配/失败计数需一并核对。

<a id="reference-results"></a>
## 参考结果

X5 发布（x5-v1.1.3）的已发布数值。

条件：X5 CPU 8×A55@1.8GHz 性能模式、BPU Bayes-e@1GHz。Float Top-1 为
量化前 ONNX，Quant Top-1 为部署结果；单线程延迟为单帧单 BPU 核，
多线程延迟和 FPS 使用并发提交。发布记录未说明数据子集、预热或重复
次数。

| Model | Input Size | Params (M) | Float Top-1 | Quantized Top-1 | Single‑thread Latency (ms) | Multi‑thread Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HGNetv2_b0 | 224x224 | 6.0 | 77.342 | 72.17 | 1.96 | 3.29 | 902.09 |
| HGNetv2_b1 | 224x224 | 6.34 | 78.872 | 73.47 | 2.41 | 3.89 | 760.13 |
| HGNetv2_b2 | 224x224 | 11.2 | 81.578 | 75.55 | 3.52 | 7.41 | 401.16 |
| HGNetv2_b3 | 224x224 | 16.3 | 82.916 | 76.51 | 4.53 | 10.37 | 287.27 |
| HGNetv2_b4 | 224x224 | 19.8 | 83.694 | 81.93 | 5.29 | 12.32 | 241.94 |

<a id="boundaries"></a>
## 数据集级评估

完整验证集评测时，使用数据集命令的 `--limit 0` 并提供全部 50,000 张验证图片。查看 Top-K 精度时同时核对 `scanned`、`matched`、`unmatched`、`failed` 和 `successful` 计数；完整数据集运行应有 50,000 张成功且有标签的图片。外部 `--model-path` 须配对准确的 `--asset-id`，CSV 类别须有效；根据 K 选择结果字段（K=5 使用 `top5_acc`，其他 K 使用 `topk_acc`）。
