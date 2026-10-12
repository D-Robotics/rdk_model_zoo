[English](README.md) | 简体中文

# ByteTrack C++ 运行

<a id="overview"></a>
## C++ 推理

本目录提供 C++ 推程所需的程序与操作说明。YOLOv5 在板端 BPU 上检测行人，C++ BYTETracker 在 CPU 上将检测关联为轨迹。`bytetrack` 可执行程序提供两种模式：`video`（板端构建）读取准备好的视频和检测模型，输出轨迹、可选的标注视频与可选的记录 TSV；`replay`（主机与板端构建）仅从文件读取检测序列并运行跟踪器。参数名与 Python CLI 的同义项一致（`--input`、`--model-path`、`--output`、`--records`、各阈值与调度参数）。

视频模式采用 YOLOv5 C++ 检测器的 letterbox 预处理与浮点坐标；Python X5 检测器采用直接缩放、裁剪并取整坐标，两种视频模式可能产生不同轨迹，见[结果解释](#results-interpretation)。已有检测器的应用可直接调用 `ByteTracker`，自行决定检测输入。

<a id="directory"></a>
## 目录结构

```text
cpp/
├── CMakeLists.txt  # 显式 target 构建：host、x5、s100、s100p、s600
├── run.sh  # 按目标构建（一次）并运行
├── inc/
│   ├── tracking.hpp  # ByteTracker 与 ByteTrackModel 公开接口（不含 SDK 头）
│   ├── cli.hpp  # CLI 选项、回放读取、记录写出、视频读写
│   ├── parallel.hpp  # 帧内并行使用的固定槽执行器
│   └── lapjv.h  # 内置线性分配求解器接口
├── src/
│   ├── tracking.cpp  # BYTETracker 移植 + 板端模型组合（基于 YOLOv5）
│   ├── lapjv.cpp  # 内置 lap 0.5.12 求解器（原样编译）
│   ├── parallel.cpp  # ByteTracker 并行区间的线程池
│   ├── cli.cpp  # 参数解析、回放 IO、记录 TSV、视频 IO 与渲染
│   └── main.cpp  # 薄入口：解析 -> 构造模型 -> 逐帧 predict
├── THIRD_PARTY_LICENSES.md
└── licenses/  # lap-LICENSE
```

`inc/tracking.hpp` 与 `inc/cli.hpp` 不包含任何板端 SDK 头文件：SDK 类型只出现在 `.cpp` 编译单元内，应用可以直接包含公开头文件并链接库。

<a id="supported-boards"></a>
## 适用板卡

| 板卡 | C++ 入口 | 前提 |
| --- | --- | --- |
| X5 | `bytetrack video` / `replay` | OpenCV、系统 HB-DNN SDK、X5 YOLOv5 640×640 packed-NV12 `.bin` 模型（默认 `n-v7.0`，见[模型说明](../../model/README_cn.md)） |
| S100 | `bytetrack video` / `replay` | OpenCV、系统 DNN/UCP SDK、`s100/yolov5x_672x672_nv12.hbm` |
| S100P | `bytetrack video` / `replay` | OpenCV、系统 DNN/UCP SDK、`s100p/yolov5x_672x672_nv12.hbm` |
| S600 | `bytetrack video` / `replay` | OpenCV、系统 DNN/UCP SDK、`s600/yolov5x_672x672_nv12.hbm` |
| 主机（无板卡） | `bytetrack replay` | C++17 编译器；无需 SDK 与 OpenCV |

S 系列制品按目标区分：使用[模型说明](../../model/README_cn.md)中对应目标自己的 `.hbm` 行。

<a id="dependencies"></a>
## 依赖

- 工具链：C++17 编译器——板端系统自带 GCC（实测 X5 为 GCC 11.2）或面向目标的 OE 交叉工具链。
- CMake ≥ 3.16，需要 `Threads`。
- 板端构建：OpenCV 开发库（视频解码/编码与渲染）。
- X5：系统 HB-DNN SDK（`libdnn`；头文件位于 `/usr/hobot/include` 与 `/usr/include/hobot/dnn`）。
- S100/S100P/S600：系统 DNN/UCP SDK（`libdnn`、`libhbucp`；头文件位于 `/usr/hobot/include` 与 `/usr/include/hobot`），以及本仓库 `utils/c_utils` 源码（构建自动加入）。
- SDK 位置覆盖（cache 变量）：`BYTETRACK_HOBOT_LIB_DIR`（默认 `/usr/hobot/lib`）、`BYTETRACK_HOBOT_INCLUDE_DIR`（默认 `/usr/hobot/include`）、`BYTETRACK_SYSTEM_HOBOT_INCLUDE_DIR`（默认 `/usr/include/hobot`）。

<a id="build"></a>
## 构建

target 必须显式指定；CMake 不探测 SoC，不做任何自动识别。`BYTETRACK_TARGET` 取 `host`、`x5`、`s100`、`s100p`、`s600` 之一，各目标使用独立构建目录，切换目标不会复用其他目标的目标文件：

```bash
# cwd：samples/vision/bytetrack/runtime/cpp
cmake -S . -B build/x5 -DBYTETRACK_TARGET=x5 -DCMAKE_BUILD_TYPE=Release
cmake --build build/x5 --target bytetrack -j
# 预期：build/x5/bytetrack
```

将 `-DBYTETRACK_TARGET=s100`（或 `s100p`、`s600`、`host`）同样使用即可；二进制位于 `build/<target>/bytetrack`。

启动脚本封装了同样的两条命令（在仓库根目录执行）：

```bash
bash samples/vision/bytetrack/runtime/cpp/run.sh --target x5 -- --help
```

<a id="run"></a>
## 运行

先准备模型与视频（可执行程序不会下载任何东西）：按[模型说明](../../model/README_cn.md)与[Sample 快速体验](../../README_cn.md#quickstart)操作。然后在仓库根目录执行：

```bash
# X5：使用默认检测变体（n-v7.0）进行视频跟踪
bash samples/vision/bytetrack/runtime/cpp/run.sh --target x5 -- video \
  --input samples/vision/bytetrack/test_data/track_test.mp4 \
  --model-path samples/vision/bytetrack/model/yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin \
  --output result_x5.mp4 --records tracks_x5.tsv --threads 1
# 预期："video: <N> frames, <M> track rows"、result_x5.mp4、tracks_x5.tsv
```

S 目标传入自己的模型行与目标名：

```bash
bash samples/vision/bytetrack/runtime/cpp/run.sh --target s100 -- video \
  --input samples/vision/bytetrack/test_data/track_test.mp4 \
  --model-path samples/vision/bytetrack/model/s100/yolov5x_672x672_nv12.hbm \
  --output result_s100.mp4 --records tracks_s100.tsv
```

其他 S 板将 `s100p`/`s600` 与对应的 `model/<target>/yolov5x_672x672_nv12.hbm` 替换即可。省略 `--output` 关闭渲染与视频编码；省略 `--records` 关闭 TSV 输出；两个写出器相互独立。

host 目标无需板端 SDK，仅从检测文件运行跟踪器：

```bash
bash samples/vision/bytetrack/runtime/cpp/run.sh --target host -- replay \
  --detections persons.btcpp --records tracks.tsv --threads 1
# 预期："replay: <N> frames, <M> track rows, final frame_id <F>"
```

<a id="parameters"></a>
## 参数

`bytetrack <mode> [options]`，mode 为 `replay` 或 `video`（`video` 仅板端构建存在；主机二进制会拒绝）。默认值为代码实际定义的值：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--detections FILE` | replay 必填 | BTCPP1 float32 检测序列 |
| `--input FILE` | video 必填 | 输入视频 |
| `--model-path FILE` | video 必填 | 检测模型（X5 `.bin` / S `.hbm`） |
| `--output FILE` | 无 | 标注输出视频（mp4v）；省略则不渲染 |
| `--records FILE` | 无 | 逐帧记录 TSV |
| `--score-thres F` | 0.25 | 检测器分数阈值，[0,1] |
| `--nms-thres F` | 0.45 | 检测器 NMS IoU 阈值，[0,1] |
| `--track-thresh F` | 0.3 | 跟踪高分检测阈值 |
| `--track-buffer N` | 60 | 丢失轨迹保留窗口，按 `frame_rate / 30` 缩放 |
| `--match-thresh F` | 0.8 | 第一轮关联最大匹配代价 |
| `--frame-rate N` | 30 | 用于缩放保留窗口的输入帧率 |
| `--mot20` | 关闭 | 源码可选匹配模式（不融合检测分数） |
| `--bpu-cores I` | -1 | BPU 核心：-1 任意，或固定索引 0..3 |
| `--priority I` | 0 | BPU 任务优先级，0..255 |
| `--max-frames N` | 0 | 处理帧数上限；0 表示全部 |
| `--threads N` | 1 | 跟踪器内部线程数，包含调用线程 |

输出路径（`--records` 或 `--output`）若与输入（`--detections`、`--input`、`--model-path`）指向同一文件——同一路径、符号链接别名或等价的未存在路径——会在任何文件打开之前被拒绝。`--records` 与 `--output` 也必须指向不同文件。

X5 使用默认的 `--priority 0` 和 `--bpu-cores -1`。S 系列的 C++ `--bpu-cores` 接收单个核心索引，Python CLI 则接收索引列表。

<a id="interface-lifecycle"></a>
## 接口与资源生命周期

`inc/tracking.hpp` 提供两个接口，均为 RAII——资源在构造函数中分配、析构函数中释放，没有 `init()`/`deinit()`。

- 配置：`TrackingConfig{track_thresh=.3, track_buffer=60, match_thresh=.8, frame_rate=30, mot20=false}`——`frame_rate` 按 `track_buffer/30` 缩放；越界值在构造函数抛出 `TrackingConfigError`。
- 模型：`ByteTracker(config, threads)` 校验配置并建立跟踪状态；`update(detections, img_h, img_w)` 推进一帧并返回自有 `Track` 快照（`track_id`、原图像素 XYXY 双精度 `tlbr`、`score`、`frame_id`）；另有 `frame_id()`、`persistent_track_count()`、`parallel_usage()`、`reset()`。`update` 对 NaN 坐标/分数或非正框/尺寸抛出 `TrackingInputError`。
- 模型：`ByteTrackModel::Options{model_path, score_thres=.25, nms_thres=.45, bpu_priority=0, bpu_core=-1, tracking, threads=1}`——仅板端构建。构造函数加载 YOLOv5 模型并分配 BPU 张量（失败抛出），析构函数释放。
- 数据流：`preprocess(const Frame&) -> Prepared`（letterbox/张量准备）、`infer(const Prepared&) -> Raw`（BPU 前向）、`postprocess(const Raw&, const Prepared&) -> std::vector<Track>`（解码行人、COCO 类 0、一次跟踪器更新）；`predict(const Frame&)` 将三个阶段各执行一次。`detect(const Frame&)` 返回检测列表但不推进跟踪器。`Frame` 持有 HWC BGR 字节；`Prepared`/`Raw` 持有按 const 引用传递的检测器负载。
- 线程：每条有序视频流一个实例；实例非线程安全。`--threads` 并行化帧内的逐轨迹 Kalman 预测与 IoU 行（任意线程数结果一致）；关联保持顺序。多路视频请为每路分配独立实例与线程。
- 视频 IO（`inc/cli.hpp`）：`VideoReader` 在构造函数打开输入；`VideoSink` 在第一次 `write` 时延迟打开输出（mp4v、以该帧尺寸），路径不可写时抛出；两者都在析构函数释放 OpenCV 句柄。

每条视频流使用独立实例并按帧序调用；空检测帧也调用 `update`。`reset()` 清除轨迹历史和帧号，保留递增的 ID 计数。ID 在各实例内独立，多路应用用 `(stream_id, track_id)` 标识目标。

<a id="results-interpretation"></a>
## 结果解释

成功时程序打印一行摘要——`video: N frames, M track rows` 或 `replay: N frames, M track rows, final frame_id F`——并退出 `0`。退出码 `2` 表示命令行错误（消息附带用法）；退出码 `1` 表示运行时错误（模型加载、视频打开、记录写入）。记录写入失败会被报告，不会静默丢弃。

`--records` 写出带表头 `frame lane track_id x1 y1 x2 y2 score` 的 TSV：帧号从 1 开始，CLI 的 `lane` 为 0，坐标为原图像素（`%.17g`）。无输出轨迹的帧没有 TSV 行，但仍推进跟踪状态。`--output` 写出带轨迹框与 `ID:<n>` 标签的 MP4，配色与 Python 渲染一致。

回放输入为小端二进制：8 字节 magic `BTCPP1\0\0`，四个 `uint32`（版本 1、帧数、图像高、图像宽）；每帧先写 `uint32` 检测数，再写每个检测的五个 `float32`（x1、y1、x2、y2、score）。坐标必须有限且框面积为正。

当前行为，完整视频检测路径：C++ 视频模式使用 YOLOv5 C++ 检测器的 letterbox 预处理（线性缩放、灰 127 填充）并保留浮点、未裁剪坐标，而 Python X5 路径使用直接最近邻缩放、裁剪和整数截断。这些预处理和坐标约定可能使同一视频产生不同的检测框和轨迹。需要共用检测器时，将检测结果直接传给 `ByteTracker::update`。

<a id="x5-cpu-performance"></a>
### X5 CPU 性能

以下为 2026-10-12 在 X5 上的跟踪计算数据，输入为相同的合成人员运动序列，包含遮挡、低分检测和空帧。每档预热 100 帧后处理 1200 帧，重复两轮。Release 使用 GCC 11.2、`-O3`，不使用 fast-math。统计不含检测、视频解码、画框和编码。

| 人数上限 | 单线程平均 / P95（ms/帧） | CPU 时间（ms/帧） | 30 FPS 单核预算 |
|---|---:|---:|---:|
| 5 | 0.041 / 0.051 | 0.042 | 0.12% |
| 20 | 0.175 / 0.221 | 0.175 | 0.52% |
| 50 | 0.629 / 0.844 | 0.630 | 1.89% |
| 100 | 1.912 / 2.656 | 1.912 | 5.73% |

同板、同输入的 Python 平均延迟依次为 4.359、12.825、29.997、60.227 ms/帧；C++ 单线程对应约快 105、73、48、31.5 倍。Python 环境为 3.10.12、NumPy 1.26.4、SciPy 1.10.1、lap 0.5.13、cython-bbox 0.1.5，BLAS/OpenMP 线程数为 1。

单核预算按 CPU 毫秒/帧 × FPS ÷ 10 计算。单线程进程 RSS 约 8.5–12.4 MiB，包含动态库、输入序列及计时记录；100 人档加载输入后的基线约 11.1 MiB，不能将整个 RSS 当作跟踪状态的独占内存。

100 人输入下，帧内线程数与独立视频流数的表现如下：

| 配置 | 每路平均延迟（ms/帧） | 每路帧 CPU 时间（ms） | 总吞吐（路帧/秒） |
|---|---:|---:|---:|
| 单路，1 线程 | 1.912 | 1.912 | 522 |
| 单路，2 线程 | 1.897 | 1.962 | 526 |
| 单路，4 线程 | 1.850 | 1.963 | 539 |
| 单路，8 线程 | 1.915 | 1.912 | 521 |
| 2 路，每路 1 线程 | 1.955 | 1.950 | 1020 |
| 4 路，每路 1 线程 | 2.140 | 2.134 | 1864 |

CPU 时间为整个进程计算窗口的 CPU 总时间除以总路帧数；多路总吞吐不等于单路帧率。100 人、8 线程配置未达到并行分块阈值，实际串行执行。四线程单路延迟仅下降约 3%，而四路独立跟踪的总吞吐约为单路的 3.57 倍。推荐保留默认 `--threads 1`，多路应用优先并行不同视频流。四路进程 RSS 约 15.7 MiB。

CPU 调频保持系统默认，运行时间较短的小人数档位容易受调频与调度影响。以上是纯跟踪吞吐，完整视频速度还取决于检测和视频输入输出。
