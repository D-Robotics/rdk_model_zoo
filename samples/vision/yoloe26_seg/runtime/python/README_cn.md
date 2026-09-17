[English](./README.md) | 简体中文

# Python 推理

## 环境与文件

请在 S100 或 S100P 开发板上运行推理，并使用板端提供的 `hbm_runtime`、
NumPy、OpenCV 和 SciPy。板端不需要 Torch 或 ONNX Runtime，运行脚本也不会自动安装依赖。

本示例涉及的目录结构如下：

```text
yoloe26_seg/
├── model/
│   ├── nash-e/              # S100 的 HBM、JSON 元数据和词表
│   └── nash-m/              # S100P 的 HBM、JSON 元数据和词表
├── runtime/python/
│   ├── main.py              # 图片推理命令行入口
│   ├── run.sh               # 下载并运行的封装脚本
│   └── yoloe26seg.py        # 分阶段 Python API
└── test_data/
    └── office_desk.jpg      # 默认输入图片
```

内置输入和默认模型路径均从示例目录解析，不受当前工作目录影响。相对输出路径
则从执行命令时所在的目录解析。

## 运行

从 `samples/vision/yoloe26_seg/` 示例根目录执行：

```bash
bash runtime/python/run.sh --size n
bash runtime/python/run.sh --size x --test-img /path/image.jpg --output result.jpg
python3 runtime/python/main.py --size s --march nash-m --output result.jpg
```

未指定 `--model-path` 时，`run.sh` 会识别板型、下载所选模型并校验发布哈希。
直接执行 `main.py` 时，需要提前将模型文件放在 `model/<march>/` 下，或显式
传入文件路径。指定的 march 和模型元数据必须与识别出的开发板一致；S600 及
其他板型会被拒绝。

## 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--size {n,s,m,l,x}` | `n` | 已发布的模型规格，必须与元数据一致。 |
| `--march {nash-e,nash-m}` | 自动识别开发板 | 目标架构：S100 使用 `nash-e`，S100P 使用 `nash-m`。 |
| `--model-path PATH` | `model/<march>/<规格对应的 HBM>` | HBM 文件；指定后 `run.sh` 不再自动下载模型。 |
| `--metadata PATH` | `model/<march>/yoloe_26<SIZE>_seg_pf.json` | 与 HBM 匹配的 JSON 元数据。 |
| `--test-img PATH` | `test_data/office_desk.jpg` | 由 OpenCV 以 BGR 格式解码的输入图片。 |
| `--output PATH` | `result.jpg` | 包含 mask、检测框、类别名称和分数的可视化结果。 |
| `--json-output PATH` | 不生成 | 可选 JSON 数组，记录每个结果的 `box`、`score`、`class_id` 和 `name`，不写入 mask。 |
| `--score-thres FLOAT` | `0.25` | 最低 sigmoid 置信度，必须严格位于 0 和 1 之间。 |
| `--max-det INTEGER` | `300` | 最多保留的检测数量，取值范围为 1 到 8400。 |
| `--multi-label` | 关闭 | 为同一个候选位置保留多个类别。 |

默认使用单标签 top-k 筛选，两种筛选模式都不执行 NMS。前处理使用居中的
640×640 letterbox，填充值为 114。

## Python API

`YoloE26Seg` 同时支持分阶段接口和单次调用接口。请从 `runtime/python/`
目录运行导入脚本，或将该目录加入 `PYTHONPATH`：

```python
import cv2

from yoloe26seg import (
    SAMPLE,
    YoloE26Seg,
    YoloE26SegConfig,
    detect_march,
    hbm_name,
    model_stem,
)

size = "n"
march = detect_march()
model_dir = SAMPLE / "model" / march
model = YoloE26Seg(YoloE26SegConfig(
    model_path=str(model_dir / hbm_name(size, march)),
    metadata_path=str(model_dir / f"{model_stem(size)}.json"),
))
image = cv2.imread(str(SAMPLE / "test_data" / "office_desk.jpg"))

input_tensor = model.pre_process(image, image_format="BGR")
raw_outputs = model.forward(input_tensor)
boxes, scores, labels, masks = model.post_process(raw_outputs, image.shape)

# 以下两种写法会在内部执行相同的三个阶段。
boxes, scores, labels, masks = model.predict(image)
boxes, scores, labels, masks = model(image)
```

各阶段的数据结构如下：

```text
pre_process(...) -> {model_name: {y_input_name: Y_plane, uv_input_name: UV_plane}}
                     Y_plane:  (1, 640, 640, 1)
                     UV_plane: (1, 320, 320, 2)

forward(...)     -> {model_name: {output_name: raw_runtime_tensor, ...}}
```

`forward()` 原样返回 `HB_HBMRuntime.run()` 的嵌套字典。`post_process()` 完成
输出反量化，并返回一一对应的原图 xyxy 检测框、浮点分数、整数类别 ID 和框内
局部 mask 列表。每个 mask 都是 `uint8` 0/1 数组，对应
`image[y1:y2, x1:x2]`；边界由裁剪后的框坐标经整数截断得到，与公共
`draw_masks` 工具一致。退化的裁剪框仍会保留空 mask，以保证结果索引对齐。

`set_scheduling_params(priority=..., bpu_cores=...)` 会把已提供的调度参数传给
Runtime；未提供的参数保持不变。
