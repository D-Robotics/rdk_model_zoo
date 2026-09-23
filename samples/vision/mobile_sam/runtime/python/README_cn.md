[English](README.md) | 简体中文

# MobileSAM Python Runtime

<a id="environment"></a>
## 环境

主机检查使用仓库 `.venv`、Python、NumPy、OpenCV 和读取 manifest 所需的 PyYAML；记录的主机 fixture 版本为 Python 3.14.7、NumPy 2.5.3、OpenCV 4.14.0、PyYAML 6.0.3。本 sample 未固定 NumPy/OpenCV 版本。runtime 语法要求 Python 3.10 或更新版本；板端 SDK/系统版本未知且未运行。`--list-models` 和显式 target 的 `--dry-run` 只解析 manifest 与 tensor 契约，不构造 SDK。

板端 SDK 应由匹配的系统镜像提供，不从无关主机环境安装 `hbm_runtime`。在仓库根目录检查必要依赖：

```bash
# cwd: 所选板卡上的仓库根目录
python3 -c "import numpy, cv2, yaml, hbm_runtime; print('runtime dependencies available')"
```

若仅缺普通 Python 依赖，在板端 SDK 实际使用的 Python 环境安装（`python3 -m pip install numpy opencv-python PyYAML`）。源配方未固定板端依赖版本，应保留镜像与 SDK 的兼容约束。上述命令仅检查导入可用性；磁盘/RAM需求尚未测量，两个模型都须能由目标 runtime 同时加载。

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

`run.sh` 仅委托此 CLI，不会下载。box 使用 resize 后 `512x512` 图像坐标。自定义模型路径必须配对对应的精确 manifest asset ID。

<a id="parameters"></a>
## 参数

| 参数 | 类型 | 默认值 | 含义 |
|---|---|---|---|
| `--target` | choice | `auto` | `x5`、`s100`、`s100p` 或 `s600` |
| `--encoder-asset-id` | string | `null` | 精确 encoder manifest reference |
| `--decoder-asset-id` | string | `null` | 精确 decoder manifest reference |
| `--encoder-model-path` | path | `null` | 外部 encoder 路径，必须有 encoder ID |
| `--decoder-model-path` | path | `null` | 外部 decoder 路径，必须有 decoder ID |
| `--test-img` | path | samples/vision/mobile_sam/test_data/dogs.jpg | BGR 输入图像 |
| `--img-save-path` | path | samples/vision/mobile_sam/test_data/mobile_sam_full_mask_result.jpg | overlay 输出 |
| `--mask-save-path` | path | samples/vision/mobile_sam/test_data/mobile_sam_binary_mask_result.png | 二值 mask 输出 |
| `--box` | `x1,y1,x2,y2` | `[185.0, 120.0, 380.0, 445.0]` | 512 图像坐标中的有序 box |
| `--priority` | integer | `0` | runtime 优先级 |
| `--bpu-cores` | integers | `null` | S 系列 core；省略时为 `[0]`；X5 拒绝此参数 |
| `--list-models` | flag | false | 不加载 SDK，仅列 manifest 制品 |
| `--dry-run` | flag | false | 不加载 SDK，仅解析模型对和 tensor 契约 |

<a id="results"></a>
## 结果

CLI 以 JSON 输出 target、encoder/decoder asset ID、输入路径、mask 输出路径、选中 mask index 和 IoU，并写入 overlay 与 `512x512` 二值 mask。结果 mask 和低分辨率 mask 是 pipeline 结果的独立副本。

<a id="integration-example"></a>
## 集成示例

下面完整示例假设已准备 S100 模型对，并在安装 `hbm_runtime` 的板端执行：

```python
import importlib
from pathlib import Path
import cv2

root = Path.cwd()
binding = importlib.import_module("samples.vision.mobile_sam.runtime.python.model_binding")
runner_type = importlib.import_module("samples.vision.mobile_sam.runtime.python.model_runner").RuntimeModelRunner
pipeline_type = importlib.import_module("samples.vision.mobile_sam.runtime.python.pipeline").MobileSAMPipeline
selection = binding.resolve_selection("s100")
runner = runner_type(selection)
bound = runner.load()
runner.set_scheduling_params(priority=0, bpu_cores=[0])
pipeline = pipeline_type(runner, bound)
image = cv2.imread(str(root / "samples/vision/mobile_sam/test_data/dogs.jpg"), cv2.IMREAD_COLOR)
if image is None:
    raise FileNotFoundError("test_data/dogs.jpg")
result = pipeline.predict(image, box=(185, 120, 380, 445))
assert result["mask"].shape == (512, 512)
```

<a id="stage-io"></a>
## 阶段 I/O

每个 stage 都有独立的三个方法。encoder `pre_process` 校验 BGR HWC 输入，按 source ImageNet mean/std 归一化并返回独立持有的 contiguous RGB NCHW float32 tensor，以及不可变的本次几何 context；encoder `forward` 发送该 tensor 并保留经 metadata 校验的 native embedding；encoder `post_process` 负责独立持有的 float32 副本。decoder `pre_process` 负责独立持有 embedding 和 box context；decoder `forward` 发送 decoder tensor 并保留 native `low_res_masks`/`iou_predictions`；decoder `post_process` 将已接受的 native 数值数组 cast 为独立 float32，选择 IoU 最大者，将 raw logits resize 到 `512x512` 后以 `>0` 阈值化。`predict` 按 `encoder.pre_process → encoder.forward → encoder.post_process → decoder.pre_process → decoder.forward → decoder.post_process` 串联。不会隐式 dequantize。decoder box 遵循实际 metadata：S 使用 `[1,4]`，X5 接受 `[1,4]` 或 `[1,4,1,1]`；decoder mask 为 `[1,3,H,W]` 且观察到的 `H/W` 为正，IoU 为 `[1,3]` 或 `[1,3,1,1]`。

| 绑定输出 | X5 | S100 / S100P / S600 |
| --- | --- | --- |
| Encoder embedding | `[1,256,32,32]` | `[1,256,32,32]` |
| Decoder masks | `[1,3,128,128]` | `[1,3,H,W]`；正数 H/W 从实际 metadata 读取 |
| Decoder IoU | `[1,3,1,1]` | 按实际 metadata 为 `[1,3]` 或 `[1,3,1,1]` |

允许的 native 输出 dtype 为 `float16`、`float32`、`int8`、`uint8`、`int16`、`int32`，每个数组必须与观察到的 metadata 完全匹配。兼容规则不代表已读到发布模型的实际 dtype；仅保留源 float32 cast，不做反量化。IoU 是模型预测的 mask 质量分数，不是带真实标注的数据集实测。结果字段为 `mask`（独立持有的 bool `[512,512]`）、`iou`（float）、`mask_index`（整数 0–2）和 `low_res_masks`（独立 float32 `[1,3,H,W]`）。PNG 将 false/true 编码为 0/255。不声明 SDK 实例可并发使用。

<a id="troubleshooting"></a>
## 故障排查

- 模型缺失：执行显式模型准备命令并确认两个路径。
- 自定义路径被拒绝：提供对应 stage 的精确 asset ID。
- X5 core 错误：省略 `--bpu-cores`，X5 不支持此选择。
- box 无效：传入四个有限、递增且位于 `[0,512]` 的坐标。
- shape 不匹配：检查 `--dry-run` 和 runtime metadata，不从文件名猜尺寸。
- 板卡身份或 SDK 错误：确认 target 与匹配的 `hbm_runtime` 镜像。
