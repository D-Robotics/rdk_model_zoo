# Python 运行

`python main.py --help` 查看参数。使用 `--platform x5|s100|s100p|s600`、`--family`、`--task detect|seg|pose|cls`、`--model-size` 选择组合。内置模型和测试数据路径相对本 Sample，用户传入路径相对调用目录。

共用类为 YoloDetect、YoloSeg、YoloPose、YoloCls；配置中传入 `platform=resolve_platform(...)`。S YOLOv10 使用 YoloV10Detect。共用 Pose 返回 boxes、scores、class_ids、xy、confidence 五项；旧 X5 包装类保留三项返回值。`--classes-num`、逗号分隔的 `--strides`、分割 `--mc` 保留显式配置；这些参数必须与实际模型输出一致。

## YOLO26

五任务统一用 `--family yolo26`。分类 resize 默认 0，运行 CLI 的 NMS 默认 X5=0.70、S=0.45。OBB 通过 `--angle-sign`、`--angle-offset`（度）和 `--regularize` 调整。详细平台协议及修正见 [Sample 说明](../../README_cn.md)。
