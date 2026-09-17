[English](./README.md) | 简体中文

# PaddleOCR 记录评估器

`evaluate.py` 比较已经保存的检测/识别记录。它不会加载 `hbm_runtime`、重新
执行模型、下载数据集，也不会另行定义模型准确率协议。板端运行完成后，只
要有对应图片集和标注，就可以使用这个工具。

## 记录格式

输入可以是一个 JSON 对象、JSON 数组，或每行一个对象的 JSONL。推荐使用
`image` 作为稳定标识，也接受 `image_id` 和 `id`。如果没有标识，会使用
记录的行号/序号。每条记录必须有长度相同的 `boxes` 与 `texts` 列表。框是
`[[x, y], ...]` 形式的多边形，也接受多一层 OpenCV contour 嵌套
（`[[[x, y], ...]]`）。

标注示例：

```json
{"image":"street-001.jpg","boxes":[[[20,30],[180,30],[180,70],[20,70]]],"texts":["RDK"]}
```

canonical Python 结果可以直接作为预测对象：补充一个 `image` 字段即可；
也可以在每张图片推理一次后逐行写入 JSONL：

```json
{"image":"street-001.jpg","target":"s100","boxes":[[[21,31],[179,31],[179,69],[21,69]]],"texts":["RDK"]}
```

## 执行

```bash
python samples/vision/paddle_ocr/evaluator/evaluate.py \
  --ground-truth /data/labels.jsonl \
  --predictions /data/predictions.jsonl \
  --iou-threshold 0.5 \
  --output /data/paddleocr-evaluation.json
```

报告同时打印到 stdout。对每张图片，评估器按照输入顺序遍历 GT 框，在未使
用的预测框中选择 IoU 最高且达到阈值的框；IoU 相同则保持预测文件中的先后
顺序。多边形 IoU 使用各自的轴对齐外接框计算，与本 sample 输出的框级结果
一致，同时不引入额外依赖。

报告包含 GT/预测/匹配总数、未匹配数、检测 precision/recall/F1，以及
`recognition` 对象。识别指标只在 IoU 匹配区域上计算：`exact_rate` 统计字
符串完全相同的比例，`normalized_similarity` 为 1 减去编辑距离除以较长
字符串长度（分母至少为 1）。没有匹配区域时，识别状态为
`status: "not_run"`，两个值均为 0。空 GT 记录合法，不会隐式生成分数。

评估器只报告用户提供记录的测量值。仓库示例图片没有完整标注数据集，因此
工具不会据此声称数据集准确率；运行时/板端一致性证据单独保存在迁移验收
记录中。
