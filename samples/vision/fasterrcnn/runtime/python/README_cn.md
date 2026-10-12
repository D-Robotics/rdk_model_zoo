[English](README.md) | 简体中文

# Faster R-CNN Python runtime

<a id="overview"></a>
## 功能

对图片执行 COCO 目标检测，对视频执行 ByteTrack 行人跟踪。`main.py` 创建模型并调用 `predict`。

<a id="directory"></a>
## 目录结构

```text
main.py       # 运行入口
cli.py        # 参数、模型选择与图片/视频读写
detect.py     # 模型初始化与推理阶段
tracking.py   # 行人跟踪
run.sh        # Shell 入口
```

<a id="environment"></a>
## 环境

使用板卡系统中的 `hbm_runtime`、NumPy、OpenCV 和 SciPy。X5/S100/S100P 使用 Python 3.10，S600 使用 Python 3.12。行人跟踪还需安装两个跟踪依赖，在仓库根目录创建环境并执行下面的命令。源码构建需要 C++ 编译器和当前 NumPy 的头文件。
```bash
python3 -m venv --system-site-packages .venv-fasterrcnn
. .venv-fasterrcnn/bin/activate
python3 -m pip install Cython setuptools wheel
python3 -m pip install --no-build-isolation --no-deps lap==0.5.12 cython-bbox==0.1.5
```

<a id="usage"></a>
## 运行

先按[模型准备说明](../../model/README_cn.md#preparation)准备两个文件。默认命令使用板卡目录中的模型组合；自行编译的模型通过两个路径参数一起指定。在仓库根目录执行：
```bash
cd samples/vision/fasterrcnn/runtime/python
python3 main.py
python3 main.py --target x5 \
  --parta-path ../../model/x5/frcnn_mbv3_320_parta.bin \
  --partb-path ../../model/x5/frcnn_mbv3_320_partb.bin \
  --input ../../test_data/bus.jpg --save-path result.jpg
python3 main.py --target x5 \
  --parta-path ../../model/x5/frcnn_mbv3_320_parta.bin \
  --partb-path ../../model/x5/frcnn_mbv3_320_partb.bin \
  --video ../../../bytetrack/test_data/track_test.mp4 \
  --save-path people.mp4 --records people.tsv
```
`bash run.sh` 接收相同的运行参数。S 系列选择 `s100`、`s100p` 或 `s600`，并使用对应板卡的 `.hbm` 文件。

<a id="parameters"></a>
## 参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--target` | `str` | `auto` | `auto`, `x5`, `s100`, `s100p`, `s600` |
| `--parta-path` | `Path` | `None` | 第一段模型 |
| `--partb-path` | `Path` | `None` | 第二段模型，与 PartA 一起指定 |
| `--input` | `Path` | `samples/vision/fasterrcnn/test_data/bus.jpg` | 图片，配合 `--track` 时为视频 |
| `-i` | `Path` | `samples/vision/fasterrcnn/test_data/bus.jpg` | `--input` 的简写 |
| `--video` | `Path` | `None` | 视频输入，执行行人跟踪 |
| `--track` | `bool` | false | 跟踪输入视频 |
| `--reset-every` | `int` | `None` | 重新开始跟踪的正整数帧间隔 |
| `--records` | `Path` | `None` | 轨迹 TSV 路径 |
| `--no-encode` | `bool` | false | 不生成标注视频 |
| `--save-path` | `Path` | `samples/vision/fasterrcnn/test_data/result.jpg` | 输出路径，视频使用 `.mp4` 后缀 |
| `--threads` | `int` | `None` | 图像转换线程数，可选 1、2、4 |
| `--label-file` | `Path` | `samples/vision/fasterrcnn/model/coco91_labels.txt` | COCO 类别文件 |
| `--list-models` | `bool` | false | 查看板卡与本地模型路径 |
| `--dry-run` | `bool` | false | 查看模型选择结果 |

`None` 表示未指定参数：模型路径从所选板卡目录获取，线程数沿用 OpenCV 默认值。表中默认文件路径相对仓库根目录。通过 `--help` 或 `-h` 查看命令帮助。

<a id="results"></a>
## 输出

图片 JSON 包含 `boxes`（N×4，原图像素 xyxy）、`scores`（N）和 `labels`（N，原始 COCO ID），以及模型哈希和 `result_path`。类别文件保留 91 个编号位置，包含背景 0 和 COCO 未使用的编号。视频 JSON 输出帧数、平均推理/跟踪耗时和文件路径。TSV 列为 `frame, track_id, x1, y1, x2, y2, score`，`frame` 从 0 开始。使用 `--no-encode` 时，`result_path` 为 null。

<a id="integration-example"></a>
## 代码调用

准备好 X5 模型组合后，在板端 Python 环境的仓库根目录运行：
```python
from pathlib import Path
import cv2
from samples.vision.fasterrcnn.runtime.python.cli import resolve_selection
from samples.vision.fasterrcnn.runtime.python.detect import FasterRCNNTask

root = Path.cwd()  # repository root
model_dir = root / "samples/vision/fasterrcnn/model/x5"
selection = resolve_selection(
    "x5", parta_path=model_dir / "frcnn_mbv3_320_parta.bin",
    partb_path=model_dir / "frcnn_mbv3_320_partb.bin")
model = FasterRCNNTask(selection)
image = cv2.imread(str(root / "samples/vision/fasterrcnn/test_data/bus.jpg"))
if image is None:
    raise FileNotFoundError("bus.jpg")
result = model.predict(image)
print(result.boxes, result.scores, result.labels)

prepared = model.preprocess(image)
outputs = model.infer(prepared)
result = model.postprocess(outputs, prepared.context)
```
视频逐帧处理时，从 `tracking.py` 导入 `PersonTrackerTask`，以 `PersonTrackerTask(model)` 创建跟踪器，每帧调用一次 `predict(image)`。开始新的视频时调用 `reset()`。

<a id="stage-io"></a>
## 推理阶段

| 方法 | 输入 → 输出 |
| --- | --- |
| `preprocess` | HWC BGR uint8 图片 → `PreparedInput`（320×320 NV12 张量和缩放信息） |
| `infer` | 前处理结果 → PartA float32 目标分数（3375）、回归量（3375×4）、feat0（1×256×10×10） |
| `postprocess` | PartA 输出与缩放信息 → CPU 候选区域/RoIAlign、PartB 推理、最终 `DetectionResult` |
| `predict` | BGR 图片 → 独立保存的原图 boxes/scores/labels |

PartB 接收 float32：X5 为 `(150,256,7,7)`，S 系列为 `(1,256,1050,7)`。同一模型实例可连续处理输入。

<a id="troubleshooting"></a>
## 模型与输入检查

通过 `--list-models` 查看本地路径。缺少模型时，准备所选板卡的两段文件。出现模型哈希错误时检查文件来源，自行编译的组合通过两个路径参数一起指定。`auto` 无法识别板卡时，显式指定 `--target`。跟踪依赖无法导入时，激活安装了 `lap` 和 `cython_bbox` 的环境。

