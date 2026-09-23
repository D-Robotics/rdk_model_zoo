# ByteTrack Python 运行时

<a id="environment"></a>
## 环境

需要 Python 3、NumPy、OpenCV、SciPy、`lap==0.5.12` 和 `cython-bbox==0.1.5`，后两者用于 CPU matching。板端还需要 S `hbm_runtime`。`--help`、`--list-models` 和显式 target 的 `--dry-run` 不加载 SDK、模型、视频，也不联网。detector 是 S 的 672x672 split-NV12；tracker 接收 XYXY、score 和 class ID。

<a id="usage"></a>
## 使用

输入视频不随仓库提供。显式准备模型和视频后，从仓库根目录运行：

```bash
python3 -m samples.vision.bytetrack.runtime.python.main \\
  --target s100 --asset-id s:bytetrack:s100/yolov5x_672x672_nv12.hbm \\
  --model-path samples/vision/bytetrack/model/s100/yolov5x_672x672_nv12.hbm \\
  --input samples/vision/bytetrack/test_data/track_test.mp4 \\
  --output samples/vision/bytetrack/test_data/result_unified.mp4
```

成功会写出非空视频，打印 `Saved <N> tracked frames ...` 并退出 `0`。加 `--records ...jsonl` 保存每帧 track；`--max-frames` 限制 smoke，`0` 表示全部视频。

<a id="parameters"></a>
## 参数

| option | type | default | 说明 |
|---|---|---|---|
| `--target` | choice | `auto` | `auto`、`x5`、`s100`、`s100p`、`s600`；只有三个 S target 可解析 |
| `--asset-id` | string | `null` | 精确 detector identity |
| `--model-path` | path | `null` | 已存在 HBM；必须带 asset ID |
| `--input` | path | `samples/vision/bytetrack/test_data/track_test.mp4` | 用户准备的视频；源树没有该文件 |
| `--output` | path | `samples/vision/bytetrack/test_data/result_unified.mp4` | 输出视频 |
| `--records` | path | `null` | 可选 JSONL 帧记录 |
| `--score-thres` | float | `0.25` | detector 置信度 |
| `--nms-thres` | float | `0.45` | detector IoU NMS |
| `--track-thresh` | float | `0.3` | BYTETracker 高分阈值 |
| `--track-buffer` | int | `60` | 丢失帧缓存 |
| `--match-thresh` | float | `0.8` | 首次关联阈值 |
| `--frame-rate` | int | `30` | tracker 帧率缩放 |
| `--mot20` | flag | `false` | 源 matching 模式 |
| `--priority` | int | `0` | 调度优先级 |
| `--bpu-cores` | int list | `[0]` | BPU 核编号 |
| `--max-frames` | int | `0` | 0 表示全部帧 |
| `--list-models` | flag | `false` | 只读 manifest |
| `--dry-run` | flag | `false` | 需显式 target，不加载 SDK/模型/视频 |

`--list-models` 与 `--dry-run` 互斥。用户错误返回 `2`。

<a id="results"></a>
## 结果

CLI 写出带 person track 的 MP4，并可写 JSONL：`{frame, tracks:[{track_id, tlbr, score, frame_id}]}`。`ByteTrackTask.post_process` 过滤 class `0`，在更新 tracker 前丢弃宽/高不正的框，并返回自有内存的不可变 `Track` snapshot。空检测仍调用 `tracker.update` 并推进状态。ID 是进程级单调计数；新进程会从自己的计数开始。

<a id="integration-example"></a>
## 集成示例

准备精确 HBM 和 CPU 依赖后，下面示例定义所有变量，按顺序处理两次调用并展示显式阶段：

```python
from pathlib import Path
import cv2
import numpy as np
from samples.vision.bytetrack.runtime.python.model_binding import resolve_selection
from samples.vision.bytetrack.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.yolov5.runtime.python.detection import YOLOv5Task
from samples.vision.bytetrack.runtime.python.tracking import ByteTrackTask, TrackingConfig

target = "s100"
asset_id = "s:bytetrack:s100/yolov5x_672x672_nv12.hbm"
model_path = Path("samples/vision/bytetrack/model/s100/yolov5x_672x672_nv12.hbm")
selection = resolve_selection(target, asset_id=asset_id, model_path=model_path)
runner = RuntimeModelRunner(selection)
binding = runner.load()
runner.set_scheduling_params(priority=0, bpu_cores=[0])
detector = YOLOv5Task(runner, binding, score_thres=0.25, nms_thres=0.45)
task = ByteTrackTask(detector, config=TrackingConfig())
frame_path = Path("samples/vision/bytetrack/test_data/bus.jpg")
frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
if frame is None:
    raise FileNotFoundError(frame_path)
prepared = task.pre_process(frame)
native_outputs = task.forward(prepared.tensors)
explicit_result = task.post_process(native_outputs, prepared.context)
composed_result = task.predict(frame)
assert isinstance(explicit_result, tuple) and isinstance(composed_result, tuple)
print(explicit_result, composed_result)
```

`predict` 有状态，每个有序帧只能调用一次；不能把第二次调用和第一次当作纯函数相等性比较。

<a id="stage-io"></a>
## 三阶段 I/O

- `pre_process(frame)` 委托 detector，返回 tensor 和不可变几何 context。
- `forward(tensors)` 委托 native detector 推理，不推进 tracker 状态。
- `post_process(outputs, context)` 解码 detector、过滤 person、丢弃无效 clip 框、更新 tracker 一次，返回 `tuple[Track,...]`。
- `predict(frame)` 组合一个有序帧；`reset()` 建立 `frame_index == 0` 的新流，但保留进程级 ID 单调性。

<a id="troubleshooting"></a>
## 故障排查

- 缺视频会显式报错，先准备 `track_test.mp4`；不会自动下载。
- 自定义 HBM 缺少精确限定 asset ID 会被拒绝。
- `auto` 使用 `--dry-run` 会被拒绝，执行 target 必须明确。
- Letterbox padding 可能把源框 clip 成零面积；统一代码会丢弃，而 evaluator 会把源 NaN 记录为失败证据。
- 不要在独立视频或多线程间共享同一 task；tracker 状态刻意不是线程安全的。
