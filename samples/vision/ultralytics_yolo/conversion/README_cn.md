# 模型转换

导出在 Ultralytics 训练环境执行；编译在对应 OpenExplorer 工具链容器执行。依赖需显式准备，不能用板卡运行环境代替。

```bash
python export_monkey_patch.py --pt yolo11n.pt --opset 11
python mapper.py --platform x5 --onnx yolo11n.onnx --cal-images ./cal_images
python export_monkey_patch.py --pt yolo11n.pt --opset 19
python mapper.py --platform s600 --onnx yolo11n.onnx --cal-images ./cal_images
```

历史拼写 `--optse` 仍可用，参数真实传递给导出。X5/S 的标定数据、量化配置、输入配置和编译命令保持独立。`--march` 与平台冲突时报错。在配置好依赖的容器中使用 `--toolchain-help` 查看工具链参数。本次尚未实际执行编译。

## YOLO26

```bash
python export_monkey_patch.py --family yolo26 --task detect --platform x5 --pt yolo26n.pt --output yolo26n_detect.onnx
python mapper.py --family yolo26 --platform x5 --onnx yolo26n_detect.onnx --cal-images ./cal_images
python export_monkey_patch.py --family yolo26 --task cls --platform s600 --pt yolo26n-cls.pt --output yolo26n_cls.onnx
python mapper.py --family yolo26 --platform s600 --toolchain-help
```

五任务使用同一导出入口。X5 默认 opset=11、simplify=1；S 默认 opset=19、simplify=0；可显式覆盖。分类默认 imgsz=224，其余为 640。导出失败返回非零状态，不自动安装依赖。两套原有标定/编译流程仍保留在 `yolo26/mapper_x5.py`、`mapper_s.py`。本轮未执行实际 ONNX 导出和工具链编译。
