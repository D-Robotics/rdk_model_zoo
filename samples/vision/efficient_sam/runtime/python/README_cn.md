[English](README.md) | 简体中文

# EfficientSAM Python Runtime

<a id="overview"></a>
## Python 推理

运行 EfficientSAM 编码器与解码器，生成图片分割掩码。

<a id="directory"></a>
## 目录结构

```text
python/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── cli.py  # 参数、模型选择与结果交付
├── main.py  # 命令行入口：构造模型并调用 predict
├── pipeline.py  # 模型初始化与推理阶段
└── run.sh  # 运行示例
```

从 [main.py](main.py) 开始：入口构造 `EfficientSAMPipeline.from_models` 并调用 `predict`。[pipeline.py](pipeline.py) 实现模型初始化及推理阶段；[cli.py](cli.py) 负责参数、模型选择和结果交付。模型初始化会加载 Runtime，应用可复用同一个实例执行多次预测。

<a id="environment"></a>
## 环境

使用 Python 3.10+、NumPy、OpenCV 和 PyYAML，以及目标板镜像提供的 `hbm_runtime`。CLI 的 `--help`、`--list-models` 和显式 target 的 `--dry-run` 不构造 SDK。

板端 SDK 应由匹配的系统镜像提供，不从无关主机环境安装 `hbm_runtime`。在仓库根目录检查必要依赖：

```bash
# cwd: 所选板卡上的仓库根目录
python3 -c "import numpy, cv2, yaml, hbm_runtime; print('runtime dependencies available')"
```

在板端 SDK 使用的 Python 环境安装依赖（`python3 -m pip install numpy opencv-python PyYAML`）。为 encoder 和 decoder 同时驻留预留内存。

<a id="usage"></a>
## 使用

在仓库根目录为检测到的板卡准备默认模型对，然后运行：

```bash
# cwd：仓库根目录；前置：默认模型对和匹配的板端 runtime
python3 samples/vision/efficient_sam/runtime/python/main.py
# 预期：退出码 0、stdout 有 JSON，并在指定路径生成 overlay 与二值 mask
```

完整指定模型对和输出位置的 S100 命令如下：

```bash
python3 samples/vision/efficient_sam/runtime/python/main.py \
  --target s100 \
  --encoder-asset-id s:efficient_sam:nash-e/efficient_sam_vitt_encoder_512x512_nashe.hbm \
  --decoder-asset-id s:efficient_sam:nash-e/efficient_sam_vitt_decoder_512_nashe.hbm \
  --encoder-model-path samples/vision/efficient_sam/model/nash-e/efficient_sam_vitt_encoder_512x512_nashe.hbm \
  --decoder-model-path samples/vision/efficient_sam/model/nash-e/efficient_sam_vitt_decoder_512_nashe.hbm \
  --test-img samples/vision/efficient_sam/test_data/dogs.jpg \
  --img-save-path /absolute/efficient-overlay.jpg \
  --mask-save-path /absolute/efficient-mask.png \
  --priority 0 --bpu-cores 0
```

`run.sh` 只是同一 CLI 的委托，不会下载。自定义模型路径必须配对对应的精确 asset ID。

<a id="parameters"></a>
## 参数

| 参数 | 类型 | 默认值 | 含义 |
|---|---|---|---|
| `--target` | choice | `auto` | `x5`、`s100`、`s100p` 或 `s600` |
| `--encoder-asset-id` | string | `null` | 精确 encoder manifest reference |
| `--decoder-asset-id` | string | `null` | 精确 decoder manifest reference |
| `--encoder-model-path` | path | `null` | 外部 encoder 路径，必须有 encoder ID |
| `--decoder-model-path` | path | `null` | 外部 decoder 路径，必须有 decoder ID |
| `--test-img` | path | samples/vision/efficient_sam/test_data/dogs.jpg | BGR 输入图像 |
| `--img-save-path` | path | samples/vision/efficient_sam/test_data/efficient_sam_full_mask_result.jpg | overlay 输出 |
| `--mask-save-path` | path | samples/vision/efficient_sam/test_data/efficient_sam_binary_mask_result.png | 二值 mask 输出 |
| `--priority` | integer | `0` | runtime 优先级 |
| `--bpu-cores` | integers | `null` | S 系列 core；省略时为 `[0]`，X5 不选择 core |
| `--list-models` | flag | false | 不加载 SDK，仅列制品 |
| `--dry-run` | flag | false | 不加载 SDK，仅解析模型对和 tensor 契约 |

<a id="results"></a>
## 结果

CLI 输出包含 target、encoder/decoder asset ID、输入路径、mask 输出路径、选中 mask index 和 IoU 的 JSON，并写入 overlay 和 `512x512` 二值 mask。结果 mask 由 pipeline 独立持有，不是 runtime 输出的视图。

<a id="integration-example"></a>
## 集成示例

下面示例假设已准备 S100 模型对，并在安装 `hbm_runtime` 的板端执行：

```python
from pathlib import Path
import cv2

root = Path.cwd()
from samples.vision.efficient_sam.runtime.python.cli import resolve_selection
from samples.vision.efficient_sam.runtime.python.pipeline import EfficientSAMPipeline

selection = resolve_selection("s100")
pipeline = EfficientSAMPipeline.from_models(selection)
pipeline.set_scheduling_params(priority=0, bpu_cores=[0])
image = cv2.imread(str(root / "samples/vision/efficient_sam/test_data/dogs.jpg"), cv2.IMREAD_COLOR)
if image is None:
    raise FileNotFoundError("test_data/dogs.jpg")
result = pipeline.predict(image)
assert result["mask"].shape == (512, 512)
```

<a id="stage-io"></a>
## 编码与解码阶段 I/O

每个 stage 都有独立的三个方法，以 [pipeline.py](pipeline.py) 中本地 `EfficientSAMEncoder`/`EfficientSAMDecoder` 视图的规范拼写 `preprocess`/`infer`/`postprocess` 暴露。encoder `preprocess` 校验 BGR HWC 输入并返回 contiguous RGB NCHW float32 tensor 和不可变的本次几何 context；encoder `infer` 发送该 tensor 并保留经 metadata 校验的 native embedding；encoder `postprocess` 负责独立持有的 float32 embedding。decoder `preprocess` 负责独立持有 embedding 和固定 prompt context；decoder `infer` 发送 decoder tensor 并保留 native `low_res_masks`/`iou_predictions`；decoder `postprocess` 将已接受的 native 数值数组 cast 为独立 float32，选择 IoU 最大者，将 raw logits resize 到 `512x512` 后以 `>=0` 阈值化。`predict` 按 `encoder.preprocess → encoder.infer → encoder.postprocess → decoder.preprocess → decoder.infer → decoder.postprocess` 串联（等价于先 `encode_image` 再 `decode_masks`），失败经由 `StageError` 归属到出错 stage，且不会执行后续 stage。不会隐式 dequantize。encoder/decoder 名称和 S decoder 尺寸来自 runtime metadata；S mask 为 `[1,3,H,W]` 且观察到的 `H/W` 为正，IoU 为 `[1,3]` 或 `[1,3,1,1]`。

| 绑定输出 | X5 | S100 / S100P / S600 |
| --- | --- | --- |
| Encoder embedding | `[1,256,32,32]` | `[1,256,32,32]` |
| Decoder masks | `[1,3,128,128]` | `[1,3,H,W]`；正数 H/W 从实际 metadata 读取 |
| Decoder IoU | `[1,3,1,1]` | 按实际 metadata 为 `[1,3]` 或 `[1,3,1,1]` |

允许的 native 输出 dtype 为 `float16`、`float32`、`int8`、`uint8`、`int16`、`int32`，每个数组必须与观察到的 metadata 完全匹配。每个数组的实际 dtype 以观察到的 metadata 为准；本 runtime 仅做源 float32 cast，不做反量化。IoU 是模型预测的 mask 质量分数，带标注的数据集实测需另行执行。结果字段为 `mask`（独立持有的 bool `[512,512]`）、`iou`（float）、`mask_index`（整数 0–2）和 `low_res_masks`（独立 float32 `[1,3,H,W]`）。PNG 将 false/true 编码为 0/255。不声明 SDK 实例可并发使用。

<a id="troubleshooting"></a>
## 故障排查

- 模型缺失：显式执行模型准备命令并确认两个路径。
- 自定义路径被拒绝：提供对应 stage 的精确 asset ID。
- X5 core 错误：省略 `--bpu-cores`，X5 不支持此选择。
- shape 不匹配：检查 `--dry-run` 和 runtime metadata，不从文件名猜尺寸。
- 板卡身份或 SDK 错误：确认 target 与匹配的 `hbm_runtime` 镜像。

显式阶段调用 `pipeline.decoder.preprocess(embedding)` 也只接受 embedding；传入非空 `box` 会报错，固定提示不会被静默替换。CLI 参数声明、model-free 的 `--list-models`/`--dry-run` 模式、图像读取与 overlay/mask 写入位于 [cli.py](cli.py)；`main.py` 负责解析、构造 pipeline 并调用 `predict`。
