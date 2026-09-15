# Python 运行

`python main.py --help` 查看参数。使用 `--platform x5|s100|s100p|s600`、`--family`、`--task detect|seg|pose|cls`、`--model-size` 选择组合。内置模型和测试数据路径相对本 Sample，用户传入路径相对调用目录。

共用类为 YoloDetect、YoloSeg、YoloPose、YoloCls；配置中传入 `platform=resolve_platform(...)`。S YOLOv10 使用 YoloV10Detect。共用 Pose 返回 boxes、scores、class_ids、xy、confidence 五项；旧 X5 包装类保留三项返回值。`--classes-num`、逗号分隔的 `--strides`、分割 `--mc` 保留显式配置；这些参数必须与实际模型输出一致。
