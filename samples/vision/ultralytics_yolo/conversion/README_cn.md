# 模型转换

导出在 Ultralytics 训练环境执行；编译在对应 OpenExplorer 工具链容器执行。依赖需显式准备，不能用板卡运行环境代替。

```bash
python export_monkey_patch.py --pt yolo11n.pt --opset 11
python mapper.py --platform x5 --onnx yolo11n.onnx --cal-images ./cal_images
python export_monkey_patch.py --pt yolo11n.pt --opset 19
python mapper.py --platform s600 --onnx yolo11n.onnx --cal-images ./cal_images
```

历史拼写 `--optse` 仍可用，参数真实传递给导出。X5/S 的标定数据、量化配置、输入配置和编译命令保持独立。`--march` 与平台冲突时报错。在配置好依赖的容器中使用 `--toolchain-help` 查看工具链参数。本次尚未实际执行编译。
