# YOLOv5 Python 运行时

<a id="environment"></a>
## 环境

主机需要 Python 3、NumPy 和 OpenCV；板端还需要匹配的 `hbm_runtime`。S 后处理使用 metadata 中的量化描述，CPU 检查还需要 sample 列出的依赖。`--help`、`--list-models` 和显式 target 的 `--dry-run` 不加载 SDK、不下载模型、不读取板卡身份。X5 绑定一个 packed uint8 NV12 tensor `(640*640*3/2,)`；S 绑定 Y `(1,672,672,1)` 与 UV `(1,336,336,2)`。

<a id="usage"></a>
## 使用

在仓库根目录按 `model/README_cn.md` 显式准备模型后运行：

```bash
python3 -m samples.vision.yolov5.runtime.python.main --target x5 --variant n-v7.0
```

命令写出 `samples/vision/yolov5/test_data/result_unified.jpg`，打印检测 JSON 数组，退出码 `0` 表示成功。自定义 S 命令如下：

```bash
python3 -m samples.vision.yolov5.runtime.python.main \
  --target s100 --variant x-672 \
  --asset-id s:yolov5:s100/yolov5x_672x672_nv12.hbm \
  --model-path samples/vision/yolov5/model/s100/yolov5x_672x672_nv12.hbm \
  --test-img samples/vision/yolov5/test_data/kite.jpg
```

`bash samples/vision/yolov5/runtime/python/run.sh ...` 等价。S 命令需要准备模型并在可识别 S100 板端执行，本轮均未使用。

<a id="parameters"></a>
## 参数

| option | type | default | 说明 |
|---|---|---|---|
| `--target` | choice | `auto` | `auto`、`x5`、`s100`、`s100p`、`s600`；执行仍需有支持制品 |
| `--variant` | string | `null` | X5 有效默认 `n-v7.0`，S 有效默认 `x-672` |
| `--asset-id` | string | `null` | 自定义路径的精确 manifest 引用 |
| `--model-path` | path | `null` | 已存在模型；必须匹配 asset ID |
| `--test-img` | path | `null` | X5 有效路径 `test_data/bus.jpg`，S 有效路径 `test_data/kite.jpg` |
| `--label-file` | path | `samples/vision/yolov5/test_data/coco_classes.names` | 可视化标签 |
| `--img-save-path` | path | `samples/vision/yolov5/test_data/result_unified.jpg` | 带框输出图 |
| `--score-thres` | float | `0.25` | `[0,1]` 置信度阈值 |
| `--nms-thres` | float | `0.45` | `[0,1]` IoU 阈值 |
| `--resize-type` | int choice | `null` | X5 有效 `0` 拉伸，S 有效 `1` letterbox |
| `--classes-num` | int choice | `80` | 发布契约固定 80 类 |
| `--anchors` | comma float list | `[10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]` | 9 组 anchor |
| `--strides` | comma int list | `[8, 16, 32]` | 三个输出 stride |
| `--priority` | int | `0` | 0..255 调度优先级 |
| `--bpu-cores` | int list | `[0]` | 非负 BPU 核编号 |
| `--list-models` | flag | `false` | 不加载 SDK 打印 manifest |
| `--dry-run` | flag | `false` | 需显式 target，输出选择/协议，不加载 SDK |

`--list-models` 与 `--dry-run` 互斥。用户或 runtime 错误返回 `2`。

<a id="results"></a>
## 结果

CLI 打印 `boxes`、`scores`、`class_ids`，并保存到 `--img-save-path`。`YOLOv5Task.forward` 原样返回 native 输出，不做反量化或 reshape；`post_process` 在需要时按 S metadata 反量化，然后完成 sigmoid/anchor 解码、阈值和 NMS、几何逆变换。X5 刻意保留源把 XYXY 传给 OpenCV `NMSBoxes`（其 Rect 解释为 XYWH）的历史 quirk；S 使用按类 XYXY NMS，这是目标协议差异。

<a id="integration-example"></a>
## 集成示例

准备匹配模型且板卡身份已识别后，下面示例定义所有变量，并将显式阶段与 `predict` 对照：

```python
from pathlib import Path
import cv2
import numpy as np
from samples.vision.yolov5.runtime.python.model_binding import resolve_selection
from samples.vision.yolov5.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.yolov5.runtime.python.detection import YOLOv5Task

target = "x5"
variant = "n-v7.0"
asset_id = "x5:yolov5:yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin"
model_path = Path("samples/vision/yolov5/model/yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin")
image_path = Path("samples/vision/yolov5/test_data/bus.jpg")
image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
if image is None:
    raise FileNotFoundError(image_path)
selection = resolve_selection(target, variant=variant, asset_id=asset_id, model_path=model_path)
runner = RuntimeModelRunner(selection)
binding = runner.load()
runner.set_scheduling_params(priority=0, bpu_cores=[0])
task = YOLOv5Task(runner, binding, score_thres=0.25, nms_thres=0.45)
prepared = task.pre_process(image)
native_outputs = task.forward(prepared.tensors)
explicit_result = task.post_process(native_outputs, prepared.context)
composed_result = task.predict(image)
for key in ("boxes", "scores", "class_ids"):
    assert np.array_equal(getattr(explicit_result, key), getattr(composed_result, key))
print(explicit_result.boxes, explicit_result.scores, explicit_result.class_ids)
```

<a id="stage-io"></a>
## 三阶段 I/O

- `pre_process(image, resize_type=None)` 接受 HWC uint8 BGR，返回 `PreparedInput(tensors, context)`；转换为目标 NV12，并把原图高宽和 resize 模式放入不可变 context。
- `forward(tensors)` 校验名称、shape、dtype 和有限值，返回 native 输出，不解码、反量化、reshape 或修改。
- `post_process(outputs, context, score_thres=None, nms_thres=None)` 消费匹配 context，返回自有内存的 `DetectionResult(boxes, scores, class_ids)`。
- `predict` 串联同样的阶段；context 按调用隔离，显式 0 阈值不会被替换。

<a id="troubleshooting"></a>
## 故障排查

- 自定义路径缺少精确匹配的 `--asset-id` 会被拒绝。
- X5 `--variant` 必须是 9 个发布 tag 之一；S100/S600 只接受 `x-672`。
- 错误的 native 输出 shape/dtype/量化 metadata 会在推理前拒绝，不按名字猜布局。
- X5 与 S 的输入容器和 resize 默认值不同，不能交叉使用 packed X5 tensor 与 split S tensor。
