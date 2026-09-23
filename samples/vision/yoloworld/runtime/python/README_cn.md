# YOLOWorld Python runtime

<a id="environment"></a>
## 环境

推理使用与系统 `hbm_runtime` 匹配的 X5 Python。主机 fixture 使用 `.venv`
Python 3.14.7、NumPy 2.5.3 和 OpenCV 4.14.0；模块导入、help、list 和 dry-run
都不会导入板端 SDK。sample 不负责安装 runtime 依赖。

<a id="usage"></a>
## 用法

保持源默认入口：

```bash
python3 samples/vision/yoloworld/runtime/python/main.py --target x5 --prompts dog
```

无 SDK 协议检查：

```bash
.venv/bin/python samples/vision/yoloworld/runtime/python/main.py --dry-run --target x5 --prompts dog
```

自定义路径必须给出精确身份：

```bash
python3 samples/vision/yoloworld/runtime/python/main.py --target x5 --model-path /absolute/yolo_world.bin --asset-id x5:yoloworld:yolo_world.bin --vocab-file /absolute/offline_vocabulary_embeddings.json --test-img /absolute/dog.jpeg --prompts dog,person --img-save-path /absolute/result.png --priority 0 --bpu-cores 0
```

`--prompts` 拒绝空项和词表外词语；`--score-thres` 默认 `0.05`，
`--nms-thres` 默认 `0.45`，`--priority` 默认 0，`--bpu-cores` 默认 `[0]`。

<a id="parameters"></a>
## 参数

dry-run 必须显式给 `--target`，真实加载前会把它与板卡身份比较。
`--model-path` 与 `--asset-id` 用于精确模型身份。词表和图片路径默认指向
sample 的 `test_data`。prompt 属于每次调用自己的 context，最多接受 32 个非空
词，最后一个词填充剩余文本槽；没有标签文件回退。

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--target` | str | `auto` | Requested target; dry-run requires explicit `x5`. |
| `--model-path` | path | `None` | External model path; requires exact asset ID. |
| `--asset-id` | str | `None` | Exact `x5:yoloworld:yolo_world.bin` identity. |
| `--vocab-file` | path | samples/vision/yoloworld/test_data/offline_vocabulary_embeddings.json | Offline F32 embeddings. |
| `--test-img` | path | samples/vision/yoloworld/test_data/dog.jpeg | BGR input image. |
| `--prompts` | str | `dog` | Nonempty comma-separated vocabulary words. |
| `--img-save-path` | path | samples/vision/yoloworld/test_data/inference.png | Annotated output image. |
| `--priority` | int | `0` | Runtime priority 0..255. |
| `--bpu-cores` | int list | `[0]` | Runtime BPU core indexes. |
| `--score-thres` | float | `0.05` | Score filter. |
| `--nms-thres` | float | `0.45` | Class-wise IoU threshold. |
| `--list-models` | flag | `false` | Print manifest identity without SDK. |
| `--dry-run` | flag | `false` | Print protocol without board/SDK. |

<a id="results"></a>
<a id="integration-example"></a>
## 结果与集成示例

```python
import cv2, json
from samples.vision.yoloworld.runtime.python.model_binding import resolve_selection
from samples.vision.yoloworld.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.yoloworld.runtime.python.yoloworld import YOLOWorldTask
selection = resolve_selection('x5')
runner = RuntimeModelRunner(selection)
binding = runner.load()
with open('samples/vision/yoloworld/test_data/offline_vocabulary_embeddings.json') as f:
    vocabulary = json.load(f)
task = YOLOWorldTask(runner, binding, vocabulary)
image = cv2.imread('samples/vision/yoloworld/test_data/dog.jpeg')
prepared = task.pre_process(image, ['dog'])
raw = task.forward(prepared)
explicit_result = task.post_process(raw, prepared.context)
composed_result = task.predict(image, ['dog'])
```

`explicit_result` 与 `composed_result` 是 `DetectionResult`，含 `boxes[N,4]`、
`scores[N]`、`class_ids[N]` 和 prompt 元组。示例需要 X5 和模型；主机测试
通过注入 runtime 运行。

<a id="stage-io"></a>
## 阶段输入输出

| 阶段 | 输入 | 输出和职责 |
| --- | --- | --- |
| `pre_process` | BGR HxWx3 uint8/F32 图片、prompt 序列 | 不可变图片 F32[1,3,640,640]、文本 F32[1,32,512,1]、缩放/原尺寸/prompt ID context |
| `forward` | prepared tensors | 原生 raw F32[1,8400,32] 和 F32[1,8400,4]，不 reshape/dequant/NMS |
| `post_process` | raw tensors 与本次 context | 槽位 argmax、分数阈值、按类 NMS、坐标还原和结果数组 |
| `predict` | 图片和 prompts | 对一次调用严格编排上述三阶段 |

runner 先校验实际 metadata，再执行且不改变原生输出；task 负责源 F32
预处理、槽位填充、NMS 与坐标换算。

<a id="troubleshooting"></a>
## 故障排查

空 prompt、未知 prompt、非有限输入、错误 metadata 形状/类型、缺模型、未知
板卡或 target 不匹配都会报错。缺模型时先运行显式模型命令。评估器要求新的
输出目录并生成真实板端证据；主机测试不能替代它。
