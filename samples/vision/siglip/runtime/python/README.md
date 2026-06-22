English | [简体中文](./README_cn.md)

# SigLIP Python Runtime

This directory provides the Python runtime entry for the SigLIP vision encoder on RDK S100/S100P.

## Directory Structure

```text
.
├── main.py
├── siglip.py
├── run.sh
├── README.md
└── README_cn.md
```

## Arguments

| Argument | Description | Default |
|---|---|---|
| `--model-path` | SigLIP `.hbm` model path | `../../model/s100/bpu-siglip-base-patch16-224.hbm` |
| `--test-img` | Input image path | `../../test_data/dog.jpg` |
| `--image-size` | Model input size | `224` |
| `--submodel` | Submodel: `pooler_output` or `last_hidden_state` | `pooler_output` |
| `--priority` | hbm_runtime priority | `0` |
| `--bpu-cores` | BPU core indexes | `0` |

## Quick Start

```bash
cd samples/vision/siglip/runtime/python
bash run.sh
bash run.sh last_hidden_state
```

## Manual Run

```bash
python3 main.py \
  --model-path ../../model/bpu-siglip-base-patch16-224.hbm \
  --test-img ../../test_data/dog.jpg \
  --image-size 224 \
  --submodel pooler_output
```

`main.py` prints output tensor shape, mean, standard deviation, minimum, maximum, and L2 norm. Result validation should confirm that the output has no NaN/Inf values, has a non-zero L2 norm, and has the expected shape for the selected model and submodel.

## HBM Invocation

On NVIDIA devices with PyTorch and HuggingFace Transformers, SigLIP vision encoding commonly uses `SiglipVisionModel` to obtain `pooler_output` or `last_hidden_state`:

```python
from transformers import SiglipVisionModel
import torch

model = SiglipVisionModel.from_pretrained("siglip-so400m-patch14-384").to(torch.device("cuda:0"))

# input_tensor: torch.tensor, float32, NCHW RGB, (1, 3, size, size), -1.0 ~ +1.0
pooler_output = model.forward(input_tensor).pooler_output
last_hidden_state = model.forward(input_tensor).last_hidden_state
```

On RDK S100/S100P, use the wrapper in this directory or call the fixed submodels in the HBM file directly through `hbm_runtime`:

```python
from hbm_runtime import HB_HBMRuntime

model = HB_HBMRuntime("bpu-siglip-so400m-patch14-384.hbm")

# input_tensor: numpy.ndarray, float32, NCHW RGB, (1, 3, size, size), -1.0 ~ +1.0
pooler_output = model.run({"pooler_output": {"_input_0": input_tensor}})["pooler_output"]["_output_0"]
last_hidden_state = model.run({"last_hidden_state": {"_input_0": input_tensor}})["last_hidden_state"]["_output_0"]
```

## Reference Preprocessing

The SigLIP HBM input is RGB NCHW float32 in the `[-1, 1]` range. The image preprocessing flow is:

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

## API

The `SigLIP` wrapper exposes the standard interfaces:

```python
def set_scheduling_params(...)
def pre_process(...)
def forward(...)
def post_process(...)
def predict(...)
def __call__(...)
```

## License

This directory is licensed under the [Apache 2.0 License](../../../../LICENSE).
