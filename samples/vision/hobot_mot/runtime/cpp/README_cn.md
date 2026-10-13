[English](README.md) | 简体中文

<a id="overview"></a>
## C++ 运行时

使用 `HobotMotTracker` 跟踪检测框序列，或通过 `hobot_mot_replay` 读取 JSONL 文件。
构建同时生成供 Python 调用的 `libhobot_mot_capi.so`。

<a id="directory"></a>
## 目录结构

```text
runtime/cpp/
├── inc/
│   ├── track.hpp       # HobotMotTracker：preprocess/infer/postprocess/reset
│   ├── mot_c_api.h     # libhobot_mot_capi.so 的 C ABI（兼容 ctypes）
│   └── cli.hpp         # 回放 CLI 辅助声明
├── src/
│   ├── track.cpp       # 配置校验 + 按类别路由到上游
│   ├── mot_c_api.cpp   # C 边界：错误码、句柄所有权
│   ├── cli.cpp         # JSONL 加载、渲染、结果 JSON
│   └── main.cpp        # 可见入口
├── CMakeLists.txt      # 构建 hobot_mot_capi + hobot_mot_replay
├── run.sh              # 运行 build/hobot_mot_replay
├── THIRD_PARTY_LICENSES.md
└── build/              # CMake 构建产物
```

<a id="supported-boards"></a>
## 适用板卡

支持 RDK X5、S100、S100P、S600。所有板卡使用以下相同命令，在板端构建并运行。

<a id="dependencies"></a>
## 依赖

- 板端 CMake >= 3.16 与 C++17 编译器
- 已准备的上游源码树
  `model/vendor/mot-0120205146d88c3474bfc2936522e349e169fa8d/`（可用
  `-DHOBOT_MOT_SOURCE_ROOT=<路径>` 覆盖）
- 从该树链接：预编译 ARM64 `iou_based_mot` 核心以及 `hobotlog`、
  `hobotsdk`、`hobotthread`、`jsoncpp` 归档；系统库 `rt`、`dl`、
  `pthread`。不需要 OpenCV，也不需要 BPU 运行时
- 编译定义：`-DHR_POSIX -DENABLE_IOU2`（仅启用 IOU2 的上游构建选项）

<a id="build"></a>
## 构建

在样例根目录执行：

```bash
python3 model/download.py
cmake -B runtime/cpp/build -S runtime/cpp
cmake --build runtime/cpp/build
```

生成 `runtime/cpp/build/hobot_mot_replay` 和 `runtime/cpp/build/libhobot_mot_capi.so`。

<a id="run"></a>
## 运行

```bash
./runtime/cpp/run.sh                                  # 默认参数
./runtime/cpp/run.sh --config model/configs/iou2_kalman.json --output results.json
```

默认值：输入 `test_data/detections.jsonl`、配置
`model/configs/iou2.json`、帧尺寸 960x544。成功退出码 0；输入或配置非法
时退出码 2 并在 stderr 打印 `Error: ...`。

<a id="parameters"></a>
## 参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--input` | 路径 | `test_data/detections.jsonl` | 检测 JSONL |
| `--config` | 路径 | `model/configs/iou2.json` | 跟踪配置 JSON |
| `--output` | 路径 | 无 | 同时写出完整结果 JSON |
| `--frame-width` | 整数 | 960 | 记录无 `image_width` 时的帧宽 |
| `--frame-height` | 整数 | 544 | 记录无 `image_height` 时的帧高 |
| `--help` | 开关 | — | 用法说明（退出码 0） |

<a id="interface-lifecycle"></a>
## 接口与使用顺序

使用 `TrackOptions{config_path, frame_width, frame_height}` 创建 `HobotMotTracker`。
每路视频创建一个实例，按帧顺序调用 `predict(detections, timestamp_ms, width, height)`，返回 `FrameTracks`。
没有检测框的帧也要传入空列表和当前时间戳。

也可依次调用 `preprocess`、`infer`、`postprocess` 获取各阶段结果。
开始新的视频或从头回放时调用 `reset()`；轨迹 ID 从 1 重新编号。
修改配置后创建新实例。配置错误抛出 `MotConfigError`，输入错误抛出 `MotInputError`。

Python 使用的 C 接口定义在 [inc/mot_c_api.h](inc/mot_c_api.h)。

<a id="results-interpretation"></a>
## 结果解释

stdout 每帧一行：

```text
frame <frame_index> ts <timestamp_ms>: [cls <class_id> id <track_id> (<x1>,<y1>,<x2>,<y2>) score <score> <state>] ... [removed cls <class_id> id <track_id>] ...
```

行保持输入顺序（每个输入检测对应一行）。`state valid` 表示已匹配并获得
身份 `track_id`（按类别独立编号，各自从 1 开始）；`invalid` 行的
`track_id` 为 -1。`[removed ...]` 列出消失窗口到期后在本帧被移除的身份。
`--output` 以 JSON 写出相同数据（格式见
[../python/README_cn.md](../python/README_cn.md)）。退出码 2 且 stderr 出现
`Error: ...` 表示输入或配置被拒绝，此时不写结果文件。
