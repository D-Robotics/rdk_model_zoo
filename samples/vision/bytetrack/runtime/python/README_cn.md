[English](README.md) | 简体中文

# ByteTrack Python 运行时

<a id="overview"></a>
## Python 推理

使用 YOLOv5 检测行人，通过 ByteTrack 在有序视频帧之间关联目标。

<a id="directory"></a>
## 目录结构

```text
python/
├── tracker_backend/  # 轨迹关联、卡尔曼滤波与匹配
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── cli.py  # 参数、检测模型选择与结果预览
├── main.py  # 命令行入口：构造模型并调用 predict
├── run.sh  # 运行示例
└── tracking.py  # 模型初始化与推理阶段
```

从 [main.py](main.py) 开始：入口构造 `ByteTrackTask.from_model` 并调用 `predict`。[tracking.py](tracking.py) 实现模型初始化及推理阶段；[cli.py](cli.py) 负责参数、模型选择和结果交付。模型初始化会加载 Runtime，应用可复用同一个实例执行多次预测。

<a id="environment"></a>
## 环境

安装 Python 3、NumPy、OpenCV、SciPy、`lap==0.5.12` 和 `cython-bbox==0.1.5`。四种板卡均使用系统提供的 `hbm_runtime`。X5 模型输入为 640×640 packed-NV12，S 系列为 672×672 split-NV12。

在没有板卡的电脑上，也可以用 `--help` 查看参数、用 `--list-models` 查看模型列表，或指定 `--target` 后用 `--dry-run` 预览运行配置。

<a id="usage"></a>
## 使用

按 [快速体验](../../README_cn.md#quickstart) 准备模型和视频，然后从仓库根目录运行：

```bash
python3 -m samples.vision.bytetrack.runtime.python.main \
  --target x5 --variant n-v7.0 \
  --input samples/vision/bytetrack/test_data/track_test.mp4 \
  --output samples/vision/bytetrack/test_data/result_x5.mp4
```

S100：

```bash
python3 -m samples.vision.bytetrack.runtime.python.main \
  --target s100 --asset-id s:bytetrack:s100/yolov5x_672x672_nv12.hbm \
  --model-path samples/vision/bytetrack/model/s100/yolov5x_672x672_nv12.hbm \
  --input samples/vision/bytetrack/test_data/track_test.mp4 \
  --output samples/vision/bytetrack/test_data/result_unified.mp4
```

运行完成后，程序保存带行人框和 ID 的视频，并打印 `Saved <N> tracked frames...`。添加 `--records tracks.jsonl` 可保存每帧轨迹；用 `--max-frames` 限制处理帧数，`0` 表示处理整个视频。

<a id="parameters"></a>
## 参数

| option | type | default | 说明 |
|---|---|---|---|
| `--target` | choice | `auto` | `auto`、`x5`、`s100`、`s100p`、`s600` |
| `--variant` | string | `null` | 检测模型变体；X5 默认 `n-v7.0`，S 系列默认 `x-672` |
| `--asset-id` | string | `null` | 模型清单中的完整标识 |
| `--model-path` | path | `null` | 已存在 BIN/HBM；必须带 asset ID |
| `--input` | path | `samples/vision/bytetrack/test_data/track_test.mp4` | 用户准备的视频；仓库不随附该文件 |
| `--output` | path | `samples/vision/bytetrack/test_data/result_unified.mp4` | 输出视频 |
| `--records` | path | `null` | 可选 JSONL 帧记录 |
| `--score-thres` | float | `0.25` | 检测置信度 |
| `--nms-thres` | float | `0.45` | 检测框 NMS IoU 阈值 |
| `--track-thresh` | float | `0.3` | BYTETracker 高分阈值 |
| `--track-buffer` | int | `60` | 丢失帧缓存 |
| `--match-thresh` | float | `0.8` | 首次关联阈值 |
| `--frame-rate` | int | `30` | 用于计算轨迹保留时间的帧率 |
| `--mot20` | flag | `false` | MOT20 匹配模式（纯 1 − IoU 代价，不与分数融合） |
| `--priority` | int | `0` | 调度优先级 |
| `--bpu-cores` | int list | `[0]` | BPU 核编号 |
| `--max-frames` | int | `0` | 0 表示全部帧 |
| `--list-models` | flag | `false` | 显示可用模型列表 |
| `--dry-run` | flag | `false` | 指定板卡后预览配置 |

`--list-models` 与 `--dry-run` 互斥。用户错误返回 `2`。

<a id="results"></a>
## 结果

程序输出带行人框和 ID 的 MP4。添加 `--records` 后，还会保存 JSONL 记录，格式为 `{frame, tracks:[{track_id, tlbr, score, frame_id}]}`。`tlbr` 为原图上的左上角、右下角坐标。

`ByteTrackTask` 只跟踪 COCO 行人类别（类别编号 `0`），返回的 `Track` 记录不会被后续帧修改。没有检测到行人的帧也会更新轨迹和帧号。同一 Python 进程中的新目标 ID 持续递增。

<a id="integration-example"></a>
## 集成示例

准备模型和依赖后，调用 `predict` 获取一帧中的行人轨迹：

```python
from pathlib import Path
import cv2
from samples.vision.bytetrack.runtime.python.cli import resolve_selection
from samples.vision.bytetrack.runtime.python.tracking import ByteTrackTask, TrackingConfig

selection = resolve_selection("x5", variant="n-v7.0")
task = ByteTrackTask.from_model(
    selection, config=TrackingConfig(), score_thres=0.25, nms_thres=0.45
)
frame_path = Path("samples/vision/bytetrack/test_data/bus.jpg")
frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
if frame is None:
    raise FileNotFoundError(frame_path)
tracks = task.predict(frame)
for track in tracks:
    print(track.track_id, track.tlbr, track.score)
```

实际处理视频时，每帧选择一种调用方式：调用一次 `predict`，或依次调用三个阶段。

<a id="stage-io"></a>
## 三阶段 I/O

- `preprocess(frame)` 准备检测模型输入，返回张量和本帧的缩放信息。
- `infer(tensors)` 执行检测模型推理。
- `postprocess(outputs, context)` 提取有效行人框、更新轨迹，返回 `tuple[Track, ...]`。
- `predict(frame)` 按顺序完成上述三个步骤。
- `reset()` 清空已有轨迹并将帧号归零，新目标 ID 继续递增。

<a id="troubleshooting"></a>
## 故障排查

- 找不到视频：检查 `--input` 路径，或按 [快速体验](../../README_cn.md#quickstart) 下载示例视频。
- 使用自己的模型路径：同时传入对应的 `--asset-id`。
- 预览配置：`--dry-run` 需要具体的 `--target`，例如 `x5` 或 `s100`。
- 处理多路视频：每路创建独立实例，并按顺序逐帧调用。
