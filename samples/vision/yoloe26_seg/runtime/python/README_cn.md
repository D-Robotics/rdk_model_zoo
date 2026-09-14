[English](./README.md) | 简体中文

# Python 推理

板端依赖 `hbm_runtime`、NumPy 和 OpenCV，不需要 Torch 或 ONNX Runtime，
也不会自动执行 pip 安装。

在本 `runtime/python/` 目录执行：

```bash
bash run.sh --size n
bash run.sh --size x --test-img /path/image.jpg --output result.jpg
python3 main.py --size s --march nash-m --output result.jpg
```

`run.sh` 会下载所选模型并校验哈希。直接运行 `main.py` 时，需要先将模型
及配套文件下载到 `../../model/<march>/`。内置资源路径根据脚本位置解析，
不依赖启动时的工作目录。

可用参数包括：`--size`、`--march`、`--model-path`、`--metadata`、
`--test-img`、`--output`、`--json-output`、`--score-thres`、
`--max-det` 和 `--multi-label`。

显式指定的 march 和模型元数据必须与通过 `soc_name`、`board_type`
识别的板型一致。S600 及其他板型会被拒绝。

模型封装返回原图坐标系的 xyxy 检测框、分数、类别 ID 和原图尺寸的二值 mask。
默认采用单标签 top-k；`--multi-label` 允许一个候选位置保留多个类别。
两种模式均不执行 NMS。前处理使用居中 letterbox，填充值为 114，与转换流程一致。
