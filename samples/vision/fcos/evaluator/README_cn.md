# FCOS 评估器

<a id="dataset"></a>

## 数据集

- 数据集：以 COCO validation 为源参考；未随附版本、标注或准备脚本。
- 冒烟输入：`samples/vision/fcos/test_data/bus.jpg` 是一张随 sample 提供的 BGR 图片，不是 COCO 评测集。

<a id="directory"></a>
## 目录结构

```text
evaluator/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
└── compare.py  # Python 脚本
```

<a id="environment"></a>
## 环境

- 执行目标：已识别的 RDK X5 板卡、`hbm_runtime` 和一个精确 manifest 制品。
- 主机依赖：`requirements-host.txt` 中的 Python 3.10+、NumPy、OpenCV、PyYAML、SciPy；源 postprocess helper 会导入 SciPy。
- 评估器执行锁定的原始 X5 脚本（从 Git 历史加载），并在相同图片、制品、阈值和 direct resize 几何下执行本 sample 的 FCOS task。

<a id="command"></a>
## 评估命令

下面命令创建一个新的证据目录并执行两套实现的对照，不下载任何文件。模型必须提前存在，target gate 也必须确认本机为 X5 板卡。

```bash
# cwd：仓库根目录；执行前准备精确制品
EVIDENCE="/tmp/fcos-evidence-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$EVIDENCE"
python3 samples/vision/fcos/evaluator/compare.py \
  --target x5 \
  --asset-id x5:fcos:fcos_efficientnetb0_detect_512x512_bayese_nv12.bin \
  --test-img samples/vision/fcos/test_data/bus.jpg \
  --output-dir "$EVIDENCE"
```

评估器使用源默认的 direct resize（`--resize-type 0`）。runtime 已实现并有回归测试的 letterbox 逆几何不作为对照模式，因为源 decoder 使用直接比例恢复坐标。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--target` | str | 必填 | 必须为 `x5`；加载 SDK 前检查本机板卡身份 |
| `--asset-id` | str | `None` | manifest 精确身份；未提供时可用 `--variant` 选择 B0/B2/B3 |
| `--variant` | str | `None` | 省略时选择 B0 |
| `--model-path` | path | `None` | 外部制品，必须同时提供精确 asset ID |
| `--test-img` | path | 内置 `bus.jpg` | BGR 输入图片 |
| `--output-dir` | path | 必填 | 目录必须不存在，所有证据写入其中 |
| `--resize-type` | int | `0` | 源 direct-resize 对照模式 |
| `--conf-thres` | float | `0.5` | 源 FCOS confidence 阈值 |
| `--iou-thres` | float | `0.6` | OpenCV NMS IoU 阈值 |
| `--priority` | int | `0` | runtime 调度优先级 |
| `--bpu-cores` | int 列表 | `[0]` | runtime BPU 核 |

返回码：`0` 表示两套实现完全一致，`1` 表示双方都执行完成但任一检查不同，`2` 表示 target、输入、模型、SDK 或证据保存失败。差异不会被静默判为成功。

<a id="metrics"></a>
## 指标

| 指标 | 定义 | 条件 |
| --- | --- | --- |
| 输入一致性 | 比较 packed input 的名称、shape、dtype 和逐元素值 | 同一 BGR 图片及 direct resize |
| raw tensor 一致性 | 对照反量化前全部 15 个 tensor 的名称、shape、dtype 和精确值 | 同一制品、metadata、图片和 target |
| 检测结果一致性 | 源 dequant、FCOS decode、NMS 后比较框、分数和类别 ID | `conf=0.5`、`IoU=0.6`、结果数组精确一致 |
| COCO mAP | 外部另行提供 COCO evaluator 后执行 | 记录 COCO 版本、split、标注和容差 |

<a id="outputs"></a>
## 输出

新的证据目录包含：

```text
comparison.json       # argv、cwd、UTC 时间、target、rc、检查、错误和哈希
errors.json           # 执行失败时的异常类型/消息/traceback
input.npy             # 双方实际使用的 BGR 输入
source/metadata.json  source/result.json  source/inputs/*.npy  source/raw/*.npy
unified/metadata.json unified/result.json unified/inputs/*.npy unified/raw/*.npy
```

`comparison.json` 记录模型、输入、两套实现的代码、metadata JSON 以及每个保存数组的 SHA-256，同时记录板卡身份、精确命令参数、cwd、UTC 起止时间和评估器返回码。raw 目录中每一侧都保存 5 个 classification、5 个 box 和 5 个 center-ness 数组。

<a id="reference-results"></a>
## 参考结果

| 指标 | 参考值 | 条件 | 来源 |
| --- | --- | --- | --- |
| B0/B2/B3 throughput 与后处理时间 | B0 323.0 FPS/9 ms、B2 70.9 FPS/16 ms、B3 38.7 FPS/20 ms | 源 benchmark 条件 | 源 README 和 evaluator README |
| 板端实现一致性，B0/B2/B3 | 在准备好的 X5 上运行对照器，两套实现须在声明容差内一致 | `bus.jpg`、direct resize、`conf=0.5`、`IoU=0.6` | 本评估器 |
| COCO mAP | 未随附数据集 harness | 源没有数据集 harness 或标注 | — |

板端日志加载这些制品时会打印 HBRT 库与模型构建小版本不一致的警告；该警告不影响对照结果。

<a id="boundaries"></a>
## 适用范围

- 评估器不下载模型、不准备 COCO、不测性能。除已记录运行外，其他板卡、图片或阈值组合须实际执行本命令生成各自证据。
- 三个 manifest 行的 publisher SHA-256 均未知，因此本地观测哈希只能识别所捕获文件，不能证明发布者来源。
- 截图和 FPS 数值为源记录，不是新测量。
