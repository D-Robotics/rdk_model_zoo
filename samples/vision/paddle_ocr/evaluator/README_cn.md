[English](./README.md) | 简体中文

# PaddleOCR 记录评估

`evaluate.py` 比较保存下来的检测器/识别器记录。它不加载 `hbm_runtime`、
不重跑模型、不下载数据集、不定义新的精度协议——它是板端运行之后、
图像集与标注都已就绪时使用的工具。

<a id="dataset"></a>
## 数据集

随仓形态下不适用：sample 附带演示图像而非标注语料，因此不声明任何
数据集级精度。评估器消费两个用户提供的记录文件，描述同一批图像的
预测与真值。每个输入可以是单个 JSON 对象、JSON 数组或每行一个对象的
JSONL；`image` 为首选标识（接受 `image_id` 与 `id` 别名；都没有时用
行号/序号）。每条记录必须包含对齐的 `boxes` 与 `texts` 列表；框为
多边形 `[[x, y], ...]`，也接受多一层 OpenCV 轮廓嵌套
（`[[[x, y], ...]]`）。

真值示例：

```json
{"image":"street-001.jpg","boxes":[[[20,30],[180,30],[180,70],[20,70]]],"texts":["RDK"]}
```

canonical Python 的 JSON 结果加上 `image` 字段即为预测对象，或按每行
一图组成 JSONL：

```json
{"image":"street-001.jpg","target":"s100","boxes":[[[21,31],[179,31],[179,69],[21,69]]],"texts":["RDK"]}
```

<a id="environment"></a>
## 环境

仅需带标准库的主机 Python 3——不需要板端 SDK、OpenCV 或模型文件。
按下方路径可在任意目录执行。

<a id="command"></a>
## 评估命令

```bash
python3 samples/vision/paddle_ocr/evaluator/evaluate.py \
  --ground-truth /data/labels.jsonl \
  --predictions /data/predictions.jsonl \
  --iou-threshold 0.5 \
  --output /data/paddleocr-evaluation.json
```

`--iou-threshold` 默认 `0.5`；`--output` 可选——报告总是打印到
stdout。逐图匹配按输入顺序遍历真值框，在达到阈值的未用预测中取 IoU
最高者；IoU 相同时保持预测文件顺序。

<a id="metrics"></a>
## 指标

| 指标 | 定义 | 条件 |
| --- | --- | --- |
| 检测 precision / recall / F1 | 匹配数与总数之比 | 按配置的 IoU 阈值；多边形 IoU 在轴对齐外接框上计算，与本 sample 的框级输出一致 |
| 未匹配计数 | 无匹配的真值与预测 | 每次运行 |
| 识别 `exact_rate` | IoU 匹配区域上字符串完全相等占比 | 仅匹配区域 |
| 识别 `normalized_similarity` | 1 − Levenshtein / max(len)，分母最小为 1 | 仅匹配区域 |
| 识别状态 | `not_run` 且数值为零 | 无匹配区域时 |

空真值记录合法，不产生隐式分数。

<a id="outputs"></a>
## 输出

JSON 报告（stdout，指定 `--output` 时另写文件），含真值/预测/匹配
总数、未匹配计数、检测 precision/recall/F1 及上表所述 `recognition`
对象。两份输入记录文件与报告一并留作评估证据。

<a id="reference-results"></a>
## 参考结果

| 项目 | 数值 | 来源 |
| --- | --- | --- |
| 主机测试 | 43 OK（2026-09-21，主机测试套件） | 迁移证据 |
| 板端对照 | canonical 与旧管线在两块 X5 板与 S100 上一致（默认与保持长宽比路径，阶段/输入/输出逐字节校验，含旧包装入口） | 2026-09-17 集成评审 |
| S100 C++ | 渲染输出像素与源基线一致 | 2026-09-17 集成评审 |
| 数据集精度 / 延迟 | 本 sample not-run | — |

同板前后对照使用相同图像、制品字节、词典与阈值，分别在旧入口与
canonical 入口运行，先比较多边形框与解码字符串再谈渲染；各维度的
数值容差即阶段 I/O 契约的容差（相同输入下的框坐标应完全相等）。

<a id="boundaries"></a>
## 边界

评估器度量用户提供的记录；绝不把随仓演示图像变成精度声明。它不做
推理、不校验制品——运行时/板端保真证据单独保存（见上表）。真实语料
上的识别质量需要用户提供的标注语料与按目标的实测；在此之前这些数值
保持 `not-run`。
