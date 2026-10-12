[English](README.md) | 简体中文

# MobileSAM Python Runtime

<a id="overview"></a>
## Python 推理

运行 MobileSAM 编码器与方框提示解码器，生成图片分割掩码。

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

从 [main.py](main.py) 开始：入口构造 `MobileSAMPipeline.from_models` 并调用 `predict`。[pipeline.py](pipeline.py) 实现模型初始化及推理阶段；[cli.py](cli.py) 负责参数、模型选择和结果交付。模型初始化会加载 Runtime，应用可复用同一个实例执行多次预测。

<a id="environment"></a>
## 环境

使用 Python 3.10+、NumPy、OpenCV 和 PyYAML，以及目标板镜像提供的 `hbm_runtime`。`--list-models` 和显式目标板卡的 `--dry-run` 只解析清单与张量契约，不构造 SDK。

板端 SDK 应由匹配的系统镜像提供，不从无关主机环境安装 `hbm_runtime`。在仓库根目录检查必要依赖：

```bash
# cwd: 所选板卡上的仓库根目录
python3 -c "import numpy, cv2, yaml, hbm_runtime; print('runtime dependencies available')"
```

在板端 SDK 使用的 Python 环境安装依赖（`python3 -m pip install numpy opencv-python PyYAML`）。为编码器和解码器同时驻留预留内存。

<a id="usage"></a>
## 使用

在仓库根目录为检测到的板卡准备默认模型对，然后运行：

```bash
# cwd：仓库根目录；前置：默认模型对和匹配的板端 runtime
python3 samples/vision/mobile_sam/runtime/python/main.py
# 预期：退出码 0、stdout 有 JSON，并在指定路径生成 overlay 与二值 mask
```

完整指定模型对、box 和输出位置的 S100 命令如下：

```bash
python3 samples/vision/mobile_sam/runtime/python/main.py \
  --target s100 \
  --encoder-asset-id s:mobile_sam:nash-e/mobile_sam_image_encoder_norm_512x512_nashe.hbm \
  --decoder-asset-id s:mobile_sam:nash-e/mobile_sam_decoder_512_nashe.hbm \
  --encoder-model-path samples/vision/mobile_sam/model/nash-e/mobile_sam_image_encoder_norm_512x512_nashe.hbm \
  --decoder-model-path samples/vision/mobile_sam/model/nash-e/mobile_sam_decoder_512_nashe.hbm \
  --test-img samples/vision/mobile_sam/test_data/dogs.jpg \
  --img-save-path /absolute/mobile-overlay.jpg \
  --mask-save-path /absolute/mobile-mask.png \
  --box 185,120,380,445 --priority 0 --bpu-cores 0
```

`run.sh` 仅委托此 CLI，不会下载。box 使用缩放后 `512x512` 图像坐标。自定义模型路径必须配对对应的精确清单 asset ID。

<a id="parameters"></a>
## 参数

| 参数 | 类型 | 默认值 | 含义 |
|---|---|---|---|
| `--target` | choice | `auto` | `x5`、`s100`、`s100p` 或 `s600` |
| `--encoder-asset-id` | string | `null` | 精确编码器清单 reference |
| `--decoder-asset-id` | string | `null` | 精确解码器清单 reference |
| `--encoder-model-path` | path | `null` | 外部编码器路径，必须有编码器 ID |
| `--decoder-model-path` | path | `null` | 外部解码器路径，必须有解码器 ID |
| `--test-img` | path | samples/vision/mobile_sam/test_data/dogs.jpg | BGR 输入图像 |
| `--img-save-path` | path | samples/vision/mobile_sam/test_data/mobile_sam_full_mask_result.jpg | 叠加图输出 |
| `--mask-save-path` | path | samples/vision/mobile_sam/test_data/mobile_sam_binary_mask_result.png | 二值掩码输出 |
| `--box` | `x1,y1,x2,y2` | `[185.0, 120.0, 380.0, 445.0]` | 512 图像坐标中的有序 box |
| `--priority` | integer | `0` | 运行时优先级 |
| `--bpu-cores` | integers | `null` | S 系列 core；省略时为 `[0]`；X5 拒绝此参数 |
| `--list-models` | flag | false | 不加载 SDK，仅列清单制品 |
| `--dry-run` | flag | false | 不加载 SDK，仅解析模型对和张量契约 |

<a id="results"></a>
## 结果

CLI 以 JSON 输出目标板卡、encoder/decoder asset ID、输入路径、掩码输出路径、选中掩码 index 和 IoU，并写入叠加图与 `512x512` 二值掩码。结果掩码和低分辨率掩码是 pipeline 结果的独立副本。

<a id="integration-example"></a>
## 集成示例

下面完整示例假设已准备 S100 模型对，并在安装 `hbm_runtime` 的板端执行：

```python
from pathlib import Path
import cv2

root = Path.cwd()
from samples.vision.mobile_sam.runtime.python.cli import resolve_selection
from samples.vision.mobile_sam.runtime.python.pipeline import MobileSAMPipeline

selection = resolve_selection("s100")
pipeline = MobileSAMPipeline.from_models(selection)
pipeline.set_scheduling_params(priority=0, bpu_cores=[0])
image = cv2.imread(str(root / "samples/vision/mobile_sam/test_data/dogs.jpg"), cv2.IMREAD_COLOR)
if image is None:
    raise FileNotFoundError("test_data/dogs.jpg")
result = pipeline.predict(image, box=(185, 120, 380, 445))
assert result["mask"].shape == (512, 512)
```

<a id="stage-io"></a>
## 阶段 I/O

每个阶段都有独立的三个方法，以 [pipeline.py](pipeline.py) 中本地 `MobileSAMEncoder`/`MobileSAMDecoder` 视图的规范拼写 `preprocess`/`infer`/`postprocess` 暴露。编码器 `preprocess` 校验 BGR HWC 输入，按参考 ImageNet mean/std 归一化并返回独立持有的 contiguous RGB NCHW float32 张量，以及不可变的本次几何上下文；编码器 `infer` 发送该张量并保留经元数据校验的原生嵌入特征；编码器 `postprocess` 负责独立持有的 float32 副本。解码器 `preprocess` 负责独立持有嵌入特征和 box 上下文；解码器 `infer` 发送解码器张量并保留原生 `low_res_masks`/`iou_predictions`；解码器 `postprocess` 将已接受的原生数值数组转换为独立 float32，选择 IoU 最大者，将 raw logits 缩放到 `512x512` 后以 `>0` 阈值化。`predict(image, box=...)` 按 `encoder.preprocess → encoder.infer → encoder.postprocess → decoder.preprocess → decoder.infer → decoder.postprocess` 串联（等价于先 `encode_image` 再 `decode_masks`），透传每次调用的 box prompt，失败经由 `StageError` 归属到出错阶段，且不会执行后续阶段。不会隐式反量化。解码器 box 遵循实际元数据：S 使用 `[1,4]`，X5 接受 `[1,4]` 或 `[1,4,1,1]`；解码器掩码为 `[1,3,H,W]` 且观察到的 `H/W` 为正，IoU 为 `[1,3]` 或 `[1,3,1,1]`。

| 绑定输出 | X5 | S100 / S100P / S600 |
| --- | --- | --- |
| Encoder 嵌入特征 | `[1,256,32,32]` | `[1,256,32,32]` |
| Decoder masks | `[1,3,128,128]` | `[1,3,H,W]`；正数 H/W 从实际元数据读取 |
| Decoder IoU | `[1,3,1,1]` | 按实际元数据为 `[1,3]` 或 `[1,3,1,1]` |

允许的原生输出数据类型为 `float16`、`float32`、`int8`、`uint8`、`int16`、`int32`，每个数组必须与观察到的元数据完全匹配。每个数组的实际数据类型以观察到的元数据为准；本运行时仅做源 float32 转换，不做反量化。IoU 是模型预测的掩码质量分数，带标注的数据集实测需另行执行。结果字段为 `mask`（独立持有的 bool `[512,512]`）、`iou`（float）、`mask_index`（整数 0–2）和 `low_res_masks`（独立 float32 `[1,3,H,W]`）。PNG 将 false/true 编码为 0/255。每个推理线程分别创建模型实例。

<a id="troubleshooting"></a>
## 故障排查

- 模型缺失：执行显式模型准备命令并确认两个路径。
- 自定义路径被拒绝：提供对应阶段的精确 asset ID。
- X5 core 错误：省略 `--bpu-cores`，X5 不支持此选择。
- box 无效：传入四个有限、递增且位于 `[0,512]` 的坐标。
- 形状不匹配：检查 `--dry-run` 和运行时元数据，不从文件名猜尺寸。
- 板卡身份或 SDK 错误：确认目标板卡与匹配的 `hbm_runtime` 镜像。

CLI 参数声明（含 box 解析）、model-free 的 `--list-models`/`--dry-run` 模式、图像读取与 overlay/mask 写入位于 [cli.py](cli.py)；`main.py` 负责解析、构造 pipeline 并调用 `predict`。
