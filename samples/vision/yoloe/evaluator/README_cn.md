# YOLOE PF 评估

[English](README.md) | 简体中文

使用与[运行时](../runtime/python/README_cn.md)相同的后处理评估本地浮点输出模型。ONNX 后端在 CPU 执行；板端后端需要匹配的 SDK 和另行准备的浮点 BIN/HBM。本目录不下载模型/数据集，不编译模型，也不自动复现历史 BPU 性能。

<a id="dataset"></a>
## 数据集与类别映射

使用独立验证集，格式为包含 `images`、`categories`、`annotations` 的 COCO 实例标注。图片需要整数 `id`、相对路径 `file_name`、`width`、`height`；类别需要整数 `id` 和 `name`。框/掩码计分需要有效的实例框、分割标注、面积和 crowd 标记。只导出预测时，可以提供 annotations 为空的图片/类别清单。数据准备参考统一的 [COCO](../../../../datasets/coco/README_cn.md) 指南，X5/S 平台快照（[X5 COCO](../../../../platforms/x5/datasets/coco/README.md)、[S COCO](../../../../platforms/s/datasets/coco/README.md)）保留作溯源；数据集不随本仓库提供。

PF 类别 ID **不是 COCO category ID**。必须提供经过检查的映射，以固定 [4585 类词表](../test_data/classes.names)为依据，同时核对源/目标名称。[mapping.example.json](mapping.example.json) 演示 person（PF 2163 → COCO 1）和 chair（PF 821 → COCO 62）；它只有两个类别，**不是完整 COCO-80 映射**。请按自己的标注类别扩展或替换：

```json
{
  "vocabulary_sha256": "1a6c943dd251993770e7cf6fed23a38b7ac068f4c8fbc7a0db85cbe0fe5221b3",
  "mapping": [
    {"pf_id": 2163, "pf_name": "person", "category_id": 1, "category_name": "person"},
    {"pf_id": 821, "pf_name": "chair", "category_id": 62, "category_name": "chair"}
  ]
}
```

标注中的每个类别都必须覆盖，不隐式启用部分类别计分。PF ID 不能重复，但允许显式声明多个 PF 类映射到同一数据集类别；这些预测仍独立保留，映射后不追加 NMS。未映射的 PF 预测不参与数据集结果，数量会记录在报告中。名称校验能防止编号漂移，不能替用户证明语义映射正确。

按数字图片 ID 排序执行。`--limit 0` 表示全部，正整数表示前 N 张并将指标标记为子集。重复 JSON key、重复 ID/图片路径、越出 `--image-dir` 的路径、不可读图片和标注尺寸不符都会失败；不跳过困难或无效图片。每次运行使用新的输出目录。

<a id="environment"></a>
## 环境

CPU ONNX 评估在目标 Python 3.10+ 环境安装主机依赖：

```bash
# cwd: repository root
python3 -m pip install -r samples/vision/yoloe/evaluator/requirements-host.txt
python3 samples/vision/yoloe/evaluator/evaluate.py --help
```

依赖包含 ONNX/ONNX Runtime、NumPy、OpenCV、SciPy、PyYAML 和 pycocotools；只有[权重导出](../conversion/README_cn.md)需要 PyTorch。板端沿用运行时前提和系统镜像提供的 `hbm_runtime`，另安装 `pycocotools`。仅导出预测也需要它完成 RLE 编码。帮助命令无需这些第三方包或板端 SDK。

主机 ONNX 后端使用 `CPUExecutionProvider`，显式关闭图优化，与导出检查一致。输入为 float RGB NCHW，**不模拟 NV12 往返、量化或 BPU 执行**。主机上的 `--target` 选择源前后处理协议，不表示当前电脑就是该板卡。板端后端会检查真实硬件身份、模型 SHA-256 和十个 float32 输出角色；原始发布的 S 量化模型仍不能用于浮点入口。

<a id="command"></a>
## 命令

以下命令均在仓库根目录执行。将路径和 `REPLACE_WITH_64_HEX_SHA256` 替换成实际本地模型身份。ONNX 摘要来自 `export.json`；BIN/HBM 使用显式准备文件的摘要，不能拿原发布哈希标识不同的本地转换产物。

CPU ONNX 框/掩码评估：

```bash
python3 samples/vision/yoloe/evaluator/evaluate.py \
  --backend onnx --target s100 --variant 26n \
  --model-path /work/export26n/yoloe_26n_seg_pf.onnx \
  --model-sha256 REPLACE_WITH_64_HEX_SHA256 \
  --image-dir /data/coco/val2017 \
  --annotation /data/coco/annotations/instances_val2017.json \
  --category-map /data/coco/pf-category-map.json \
  --output-dir /work/evaluation-26n
```

只导出预测、不声明精度时，增加 `--predictions-only` 并使用新输出目录；仍需要 COCO 格式图片/类别清单，但可以没有真值。流程检查可加 `--limit 1`，正式验证目标 split 时移除限制。子集或只导出预测成功，不能当作完整数据集精度。

板端评估仅在准备好兼容浮点制品、进入匹配板端环境后运行：

```bash
python3 samples/vision/yoloe/evaluator/evaluate.py \
  --backend board --target s100 --variant 26n \
  --model-path /models/yoloe_26n_float.hbm \
  --model-sha256 REPLACE_WITH_64_HEX_SHA256 \
  --image-dir /data/coco/val2017 \
  --annotation /data/coco/annotations/instances_val2017.json \
  --category-map /data/coco/pf-category-map.json \
  --output-dir /work/evaluation-board-26n
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--backend` | 必填 | `onnx` CPU 或 `board`，不自动推断 |
| `--target`、`--variant` | 必填 | X5 11s/m/l；S100 11s 或 26n/s/m/l/x；S100P 26n/s/m/l/x |
| `--model-path`、`--model-sha256` | 必填 | 本地模型与精确 64 位 SHA-256 |
| `--image-dir`、`--annotation`、`--category-map` | 必填 | 图片、COCO JSON、经过检查的 PF 映射 |
| `--output-dir` | 必填 | 新目录，不覆盖原内容 |
| `--limit` | 0 | 全部图片，或按数字图片 ID 排序后的前 N 张 |
| `--predictions-only` | false | 保存预测，不计算指标 |
| `--score-thres` | 0.25 | Sigmoid 置信度阈值，严格位于 0 和 1 之间 |
| `--nms-thres` | null | E11 未指定时用 0.7；E26 拒绝显式 NMS 阈值 |
| `--resize-type` | 1 | Letterbox；E11 也允许 0（拉伸），E26 必须为 1 |
| `--no-morph` | false | 关闭 S E11 在 CLI 中默认启用的掩码形态学；其他路线不启用 |
| `--max-det` | 300 | E26 Top-K 上限，范围 1..8400；E11 只接受原默认值 |
| `--multi-label` | false | E26 每个 anchor 可保留多个类别；E11 拒绝此选项 |
| `--threads` | 2 | ONNX CPU 线程；板端路线拒绝非默认值 |

阈值影响精确率和召回率。默认值与演示 CLI 对齐，不代表已经验收的低阈值 COCO 精度配方，修改后必须保留记录。E11 使用类别内 NMS；E26 使用 Top-K、无 NMS。X5 E11 输出整图掩码，S E11/E26 输出 ROI 掩码；所有路线共用运行时解码，不维护另一套评估算法。

<a id="metrics"></a>
## 指标与可比性

`pycocotools.COCOeval` 在选中图片 ID 和全部标注类别上分别计算 bbox、segm AP/AR。AP 为比例，不是百分数；无法定义的尺寸/类别项保留 `-1`，不改成零。实际 IoU 阈值、召回采样点数量、面积标签、图片/类别 ID 和最大检测数设置随指标保存。COCO 默认 maxDets `[1,10,100]` 与 E26 模型默认 300 候选上限是不同设置。

预测为空时仍按空检测计分：存在真值则 AP 为零，不跳过。选中图片完全没有非 crowd 真值时，指标模式失败并提示使用 predictions-only；不会用无标注图片制造精度值。

掩码按原图几何编码。ROI 必须严格匹配裁剪后、整数截断的框边界；整图掩码必须匹配图片宽高，不通过评估器自动缩放来掩盖运行时几何错误。框与掩码结果分文件保存，让 COCO 分割面积来自 RLE 像素，而非包围框面积。

比较两次运行时，模型/权重身份、图片、类别映射、目标协议、预处理、阈值和指标设置必须对应。主机浮点、板端量化和浮点舍入引起的 Top-K 排序差异是不同的对照。逐图墙钟时间覆盖前处理、forward、后处理和 COCO/RLE 编码，不含图片/JSON 文件 I/O，不是 BPU-only 延迟，也不是规范的性能基准。

<a id="outputs"></a>
## 输出与失败处理

成功退出 0，已处理的失败退出 2；初始化错误可能发生在输出目录建立前。开始数据集执行后，`evaluation.json` 会记录最终状态和错误，每张完成图片的证据及时落盘。

| 文件 | 内容 |
| --- | --- |
| `evaluation.json` | 模型/标注/映射身份、配置、环境、计数、选中 ID、指标、状态 |
| `annotations.json`、`category-map.json` | 校验过的输入快照，同时记录原文件和快照摘要 |
| `images.jsonl` | 逐图 ID/路径/SHA、尺寸、预测/映射数量、墙钟时间 |
| `bbox-predictions.json` | COCO 图片/类别 ID、分数、`[x,y,width,height]` |
| `segm-predictions.json` | COCO 图片/类别 ID、分数、RLE 掩码，不含 bbox 字段 |
| `metrics.log` | 请求计分时的完整 COCO 输出 |
| `*-predictions.partial.json` | 图片中途失败时显式保存的部分预测，不当作完整运行计分 |

状态为 `predictions-only`、`evaluated` 或 `failed`；`metric_scope` 区分全部标注图片、选中子集和未请求指标。引用数字前检查已处理/选中数量及未映射预测。完成指标计算不会自动建立应用场景的精度验收阈值。

<a id="reference-results"></a>
## 历史参考与本轮验证

下表是**源分支原始 Runtime-only 测量**，不是统一浮点路线的新结果：

| 源模型 | 板卡 | Runtime 延迟 / FPS | P50 / P95 |
| --- | --- | --- | --- |
| E11s | X5 | 146.16 ms / 6.84 | 144.72 / 152.73 ms |
| E11m | X5 | 177.14 ms / 5.65 | 176.17 / 182.54 ms |
| E11l | X5 | 189.97 ms / 5.26 | 187.99 / 196.30 ms |
| E26n | S100 | 4.943 ms / 200.74 | 未发布 |
| E26s | S100 | 9.944 ms / 100.08 | 未发布 |
| E26m | S100 | 11.765 ms / 84.55 | 未发布 |
| E26l | S100 | 13.417 ms / 74.18 | 未发布 |
| E26x | S100 | 22.013 ms / 45.31 | 未发布 |

X5 条件：RDK X5 V1.0、OS 3.4.1-rp1.0.2、libdnn 1.24.5/HBRT 3.15.55、1000 MHz、单线程/core_id=1（BPU core 0）、固定 NV12，三轮、每轮 10 帧 warmup 加 200 帧计时。11s 双线程因 ION 分配错误失败，详见 [X5 完整源记录](../../../../platforms/x5/samples/vision/yoloe/evaluator/README_cn.md)。

S100 条件：V1P0、OS 4.0.5-Beta、UCP 3.13.6/HBRT 4.7.5、OE 3.7.0 INT8 KL、2026-09-08、200 帧/warmup、thread_num=1/core_id=0。S100P 未测性能。[S26 源记录](../../../../platforms/s/samples/vision/yoloe26_seg/evaluator/README_cn.md)另有原 n 模型在 S100P 的单图 Python/C++ 掩码逐像素对照，不能替代本轮代码、其他尺寸或数据集 mAP。[S11 源评估目录](../../../../platforms/s/samples/vision/yoloe11_seg/evaluator/README.md)原本为空占位，没有可迁移的已验收指标。

本轮真实 ONNX 预测使用随附图片和显式生成的 PF 类别清单，**没有真值**，不是 COCO AP 测量。近期导出所用 E26 PT 文件与归档发布 sidecar 的文件哈希不同；这本身不证明参数张量变化，但不能声称复现同一原始权重基线。尤其不能把浮点检测数写成复现 INT8 数量，也不能把差异仅归因于量化。完整证据见[评估实现记录](../../../../docs/releases/unified-migration/2026-09-28-yoloe-evaluation-review.md)。

<a id="boundaries"></a>
## 边界与检查

本轮未执行真实板端/SDK/OE 评估、独立验证集 mAP 或新性能测试。板端后端已经实现，但硬件未验证。CPU RGB 评估不包含 NV12 转换影响；固定词表、映射与模型字节必须配套，编辑词表不会获得新类别。

```bash
# cwd: repository root; evaluator host dependencies installed
python3 -m unittest discover -s samples/vision/yoloe/evaluator/tests
python3 -m unittest discover -s samples/vision/yoloe/tests
```

测试覆盖已知合成掩码的真实 COCO 计分、空预测、显式映射、严格图片/掩码几何、缺文件时部分失败记录、predictions-only 边界。合成样例满分只验证计分器，不是 YOLOE 模型精度。源运行契约测试保留在归档中，统一运行时测试覆盖共用阶段。原生 C++ 运行时已提供实现，其主机运行时范围已获独立评审（[评审记录](../../../../docs/releases/unified-migration/2026-09-28-yoloe-independent-review.md)）；真实 SDK／板端执行仍为 not-run，全分支独立验收仍未关闭。
