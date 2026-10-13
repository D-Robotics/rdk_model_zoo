[English](README.md) | 简体中文

<a id="overview"></a>
## Python 运行时

使用 `HobotMotTracker` 接收检测框并生成轨迹 ID，也可用命令行回放 JSONL 文件。
Python 通过 `native.py` 调用 C++ 构建生成的 `libhobot_mot_capi.so`。

<a id="directory"></a>
## 目录结构

```text
runtime/python/
├── main.py    # 参数 → 创建 tracker → predict → 输出
├── cli.py     # CLI：JSONL 加载校验、渲染、结果 JSON
├── track.py   # HobotMotTracker 模型类 + 配置/输入校验
├── native.py  # libhobot_mot_capi.so 的 ctypes 绑定
└── run.sh     # 转发参数运行 main.py
```

<a id="environment"></a>
## 环境

- RDK X5、S100、S100P 或 S600（ARM64 Linux）
- Python 3.8 及以上，仅使用标准库
- C++17 编译器和 CMake 3.16 及以上，用于构建原生库

原生库查找顺序：`--native-lib` 参数、`HOBOT_MOT_NATIVE_LIB` 环境变量、`runtime/cpp/build/libhobot_mot_capi.so`。

<a id="usage"></a>
## 使用

在样例根目录执行默认命令（自带示例序列与配置）：

```bash
python3 model/download.py
cmake -B runtime/cpp/build -S runtime/cpp
cmake --build runtime/cpp/build
./runtime/python/run.sh
```

写出结果文件的自定义命令：

```bash
./runtime/python/run.sh --input test_data/detections.jsonl \
    --config model/configs/iou2_kalman.json --output results.json
```

两者成功退出码为 0；输入非法、配置非法或原生库缺失时退出码为 2 并打印
`Error: ...`。

<a id="parameters"></a>
## 参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--input` | 路径 | `samples/vision/hobot_mot/test_data/detections.jsonl` | 检测 JSONL（记录格式见"结果"） |
| `--config` | 路径 | `samples/vision/hobot_mot/model/configs/iou2.json` | 跟踪配置 JSON（`tracker_type` 必须为 `"IOU_2.0"`） |
| `--output` | 路径 | null | 同时写出完整结果 JSON |
| `--frame-width` | 整数 | 960 | 记录未提供 `image_width` 时使用的帧宽 |
| `--frame-height` | 整数 | 544 | 记录未提供 `image_height` 时使用的帧高 |
| `--native-lib` | 路径 | null | `libhobot_mot_capi.so` 的显式路径 |

<a id="results"></a>
## 结果

每帧打印一行：

```text
frame <frame_index> ts <timestamp_ms>: [cls <class_id> id <track_id> (<x1>,<y1>,<x2>,<y2>) score <score> <state>] ... [removed cls <class_id> id <track_id>] ...
```

输入 JSONL 每行代表一帧。例如：

```json
{"frame_index": 0, "timestamp_ms": 1000, "image_width": 640, "image_height": 480, "detections": [{"bbox": [100, 100, 160, 260], "score": 0.95, "class_id": 0}]}
```

`frame_index` 和毫秒时间戳 `timestamp_ms` 必须递增。坐标单位为像素，尺寸字段可省略，省略时使用命令行默认尺寸。没有检测框的帧填写 `"detections": []`。

`--output` 保存 JSON 数组，每帧一个元素：

```json
[{"frame_index": 0, "timestamp_ms": 1000, "tracks": [{"class_id": 0, "track_id": 1, "bbox": [100, 100, 160, 260], "score": 0.95, "state": "valid"}], "disappeared": []}]
```

`tracks` 顺序与输入检测框一致。`valid` 表示已分配轨迹 ID；`invalid` 表示本帧未匹配，ID 为 -1。ID 按类别分别编号，使用 `(class_id, track_id)` 区分目标。`disappeared` 列出本帧到期移除的轨迹。

<a id="integration-example"></a>
## 集成示例

```python
import sys
sys.path.insert(0, "samples/vision/hobot_mot/runtime/python")
from track import Detection, HobotMotTracker

tracker = HobotMotTracker(
    config_path="samples/vision/hobot_mot/model/configs/iou2.json",
    frame_width=640, frame_height=480)
with tracker:  # 退出时释放原生实例
    ts_ms = 1000
    for boxes in [
        [Detection((100, 100, 160, 260), 0.95, 0)],       # 出现行人
        [Detection((104, 100, 164, 260), 0.95, 0)],       # 行人移动
        [],                                              # 空帧使轨迹老化
        [Detection((112, 100, 172, 260), 0.95, 0)],       # 同一身份
    ]:
        result = tracker.predict(boxes, ts_ms)
        print([(t.track_id, t.state) for t in result.tracks])
        ts_ms += 40
# -> [(1, 'valid')], [(1, 'valid')], [], [(1, 'valid')]
```

<a id="stage-io"></a>
## 分阶段调用

- `preprocess(detections, timestamp_ms, frame_width=None, frame_height=None) -> PreparedFrame`：准备当前帧的检测框、时间戳和图像尺寸。
- `infer(prepared) -> RawMotFrame`：运行跟踪器，返回轨迹 ID、坐标、状态和本帧消失的 ID。
- `postprocess(raw, prepared) -> FrameTracks`：按输入检测框的顺序组织结果。

`predict` 依次调用这三个阶段。每路视频使用一个实例，按时间顺序逐帧调用。
开始新的视频或从头回放前调用 `reset()`，轨迹 ID 从 1 重新编号。

<a id="troubleshooting"></a>
## 常见问题

- 找不到 `libhobot_mot_capi.so`：先执行上面的构建命令，或用 `--native-lib` 指定已有库文件。
- `use_kalman_filter` 类型错误：在 JSON 中填写 `true` 或 `false`。
- 时间戳没有递增：检查视频帧顺序；从头回放前调用 `reset()`。
- 配置文件已修改：重新创建 `HobotMotTracker` 后运行。

