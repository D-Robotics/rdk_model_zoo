[English](./README.md) | 简体中文

# PaddleOCR 记录评估

`evaluate.py` 比较保存下来的检测器/识别器记录。它不加载 `hbm_runtime`、
不重跑模型、不下载数据集、不定义新的精度协议——它是板端运行之后、
图像集与标注都已就绪时使用的工具。

<a id="dataset"></a>
## 数据集

随仓形态下不适用：sample 附带演示图像而非标注语料，数据集级精度需按下文
提供标注记录后评测。评估器消费两个用户提供的记录文件，描述同一批图像的
预测与真值。每个输入可以是单个 JSON 对象、JSON 数组或每行一个对象的
JSONL；`image` 为首选标识（接受 `image_id` 与 `id` 别名；都没有时用
行号/序号）。每条记录必须包含对齐的 `boxes` 与 `texts` 列表；框为
多边形 `[[x, y],...]`，也接受多一层 OpenCV 轮廓嵌套
（`[[[x, y],...]]`）。

真值示例：

```json
{"image":"street-001.jpg","boxes":[[[20,30],[180,30],[180,70],[20,70]]],"texts":["RDK"]}
```

Python 的 JSON 结果加上 `image` 字段即为预测对象，或按每行
一图组成 JSONL：

```json
{"image":"street-001.jpg","target":"s100","boxes":[[[21,31],[179,31],[179,69],[21,69]]],"texts":["RDK"]}
```

<a id="directory"></a>
## 目录结构

```text
evaluator/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
└── evaluate.py  # Python 脚本
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

Python 默认与保持长宽比两条管线（含兼容包装入口）在 X5 与 S100 上运行，
对照基于各阶段张量及其解码结果——多边形框与识别文本；S100 C++ 构建将识别
结果渲染到输出图像。各平台的
检测/识别延迟与 FPS 用[转换说明](../conversion/README_cn.md)中针对
det/rec 制品的 `hrt_model_exec perf` 命令测量。

同板前后对照使用相同图像、制品字节、词典与阈值，分别在快捷入口与
入口运行，先比较多边形框与解码字符串再谈渲染；各维度的
数值容差即阶段 I/O 契约的容差（相同输入下的框坐标应完全相等）。

<a id="boundaries"></a>
## 适用范围

评估器度量用户提供的记录；绝不把随仓演示图像变成精度声明。它不做
推理、不校验制品——运行时/板端保真证据单独保存（见上表）。真实语料
上的识别质量需要用户提供的标注语料与按目标的实测。
