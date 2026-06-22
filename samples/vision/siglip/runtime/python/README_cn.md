[English](./README.md) | 简体中文

# SigLIP Python 运行时

本目录提供 RDK S100/S100P 上 SigLIP 视觉编码器的 Python 推理入口。

## 目录结构

```text
.
├── main.py
├── siglip.py
├── run.sh
├── README.md
└── README_cn.md
```

## 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-path` | SigLIP `.hbm` 模型路径 | `../../model/s100/bpu-siglip-base-patch16-224.hbm` |
| `--test-img` | 输入图片路径 | `../../test_data/dog.jpg` |
| `--image-size` | 模型输入尺寸 | `224` |
| `--submodel` | 子模型：`pooler_output` 或 `last_hidden_state` | `pooler_output` |
| `--priority` | hbm_runtime 优先级 | `0` |
| `--bpu-cores` | BPU 核心索引 | `0` |

## 快速运行

```bash
cd samples/vision/siglip/runtime/python
bash run.sh
bash run.sh last_hidden_state
```

## 手动运行

```bash
python3 main.py \
  --model-path ../../model/bpu-siglip-base-patch16-224.hbm \
  --test-img ../../test_data/dog.jpg \
  --image-size 224 \
  --submodel pooler_output
```

`main.py` 会打印输出 tensor 的形状、均值、标准差、最小值、最大值和 L2 范数。结果正确性检查应确认输出不包含 NaN/Inf，L2 范数非零，输出 shape 与所选模型和子模型一致。

## HBM 调用方式

在 NVIDIA 设备上使用 PyTorch 和 HuggingFace Transformers 时，SigLIP 视觉编码通常通过 `SiglipVisionModel` 获取 `pooler_output` 或 `last_hidden_state`：

```python
from transformers import SiglipVisionModel
import torch

model = SiglipVisionModel.from_pretrained("siglip-so400m-patch14-384").to(torch.device("cuda:0"))

# input_tensor: torch.tensor, float32, NCHW RGB, (1, 3, size, size), -1.0 ~ +1.0
pooler_output = model.forward(input_tensor).pooler_output
last_hidden_state = model.forward(input_tensor).last_hidden_state
```

在 RDK S100/S100P 上，可以使用本目录的 wrapper，也可以直接通过 `hbm_runtime` 调用 HBM 内的固定子模型：

```python
from hbm_runtime import HB_HBMRuntime

model = HB_HBMRuntime("bpu-siglip-so400m-patch14-384.hbm")

# input_tensor: numpy.ndarray, float32, NCHW RGB, (1, 3, size, size), -1.0 ~ +1.0
pooler_output = model.run({"pooler_output": {"_input_0": input_tensor}})["pooler_output"]["_output_0"]
last_hidden_state = model.run({"last_hidden_state": {"_input_0": input_tensor}})["last_hidden_state"]["_output_0"]
```

## 参考预处理

SigLIP HBM 输入为 RGB NCHW float32，取值范围 `[-1, 1]`。图像预处理流程如下：

```python
import cv2
import numpy as np


def preprocess(image, target_size=384):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    image_padded = cv2.copyMakeBorder(
        image_resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=[127, 127, 127],
    )
    image_chw = np.transpose(image_padded, (2, 0, 1))
    image_nchw = np.expand_dims(image_chw, axis=0)
    return image_nchw.astype(np.float32) / 127.5 - 1.0
```

## 接口说明

`SigLIP` wrapper 提供标准接口：

```python
def set_scheduling_params(...)
def pre_process(...)
def forward(...)
def post_process(...)
def predict(...)
def __call__(...)
```

## License

本目录遵循 [Apache 2.0 License](../../../../LICENSE)。
