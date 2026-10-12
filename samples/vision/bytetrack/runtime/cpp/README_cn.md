[English](README.md) | 简体中文

# ByteTrack C++ 运行

<a id="overview"></a>
## C++ 推理

YOLOv5 在 BPU 上检测行人，ByteTrack 在 CPU 上关联轨迹。运行 `bytetrack video` 可处理视频，保存带框视频和轨迹记录；运行 `bytetrack replay` 可读取已有检测结果，只执行跟踪。

已有检测器时，可直接将检测结果传给 `ByteTracker::update`。视频模式的预处理和坐标格式见[结果解释](#results-interpretation)。

<a id="directory"></a>
## 目录结构

```text
cpp/
├── CMakeLists.txt  # 按板卡构建：host、x5、s100、s100p、s600
├── run.sh  # 构建并运行指定板卡程序
├── inc/
│   ├── tracking.hpp  # ByteTracker 与 ByteTrackModel 公开接口（不含 SDK 头）
│   ├── cli.hpp  # CLI 选项、回放读取、记录写出、视频读写
│   ├── parallel.hpp  # 帧内并行使用的线程调度
│   └── lapjv.h  # 内置线性分配求解器接口
├── src/
│   ├── tracking.cpp  # 轨迹关联与 YOLOv5 推理
│   ├── lapjv.cpp  # 内置 lap 0.5.12 求解器（原样编译）
│   ├── parallel.cpp  # ByteTracker 并行区间的线程池
│   ├── cli.cpp  # 参数解析、回放 IO、记录 TSV、视频 IO 与渲染
│   └── main.cpp  # 入口：解析 -> 构造模型 -> 逐帧 predict
├── THIRD_PARTY_LICENSES.md
└── licenses/  # lap-LICENSE
```

集成到 C++ 应用时，包含 `inc/tracking.hpp` 并链接 `bytetrack_core`。视频读写接口位于 `inc/cli.hpp`。

<a id="supported-boards"></a>
## 适用板卡

| 板卡 | C++ 入口 | 前提 |
| --- | --- | --- |
| X5 | `bytetrack video` / `replay` | OpenCV、系统 HB-DNN SDK、X5 YOLOv5 640×640 packed-NV12 `.bin` 模型（默认 `n-v7.0`，见[模型说明](../../model/README_cn.md)） |
| S100 | `bytetrack video` / `replay` | OpenCV、系统 DNN/UCP SDK、`s100/yolov5x_672x672_nv12.hbm` |
| S100P | `bytetrack video` / `replay` | OpenCV、系统 DNN/UCP SDK、`s100p/yolov5x_672x672_nv12.hbm` |
| S600 | `bytetrack video` / `replay` | OpenCV、系统 DNN/UCP SDK、`s600/yolov5x_672x672_nv12.hbm` |
| 主机（无板卡） | `bytetrack replay` | C++17 编译器；无需 SDK 与 OpenCV |

按[模型说明](../../model/README_cn.md)选择对应板卡的模型文件。

<a id="dependencies"></a>
## 依赖

- 工具链：支持 C++17 的板端 GCC 或对应板卡的 OE 交叉编译器。
- CMake ≥ 3.16，需要 `Threads`。
- 板端构建：OpenCV 开发库（视频解码/编码与渲染）。
- X5：系统 HB-DNN SDK（`libdnn`；头文件位于 `/usr/hobot/include` 与 `/usr/include/hobot/dnn`）。
- S100/S100P/S600：系统 DNN/UCP SDK（`libdnn`、`libhbucp`；头文件位于 `/usr/hobot/include` 与 `/usr/include/hobot`），以及本仓库 `utils/c_utils` 源码（构建自动加入）。
- SDK 位置覆盖（cache 变量）：`BYTETRACK_HOBOT_LIB_DIR`（默认 `/usr/hobot/lib`）、`BYTETRACK_HOBOT_INCLUDE_DIR`（默认 `/usr/hobot/include`）、`BYTETRACK_SYSTEM_HOBOT_INCLUDE_DIR`（默认 `/usr/include/hobot`）。

<a id="build"></a>
## 构建

通过 `BYTETRACK_TARGET` 选择 `host`、`x5`、`s100`、`s100p` 或 `s600`。以下命令构建 X5 程序：

```bash
# cwd：samples/vision/bytetrack/runtime/cpp
cmake -S . -B build/x5 -DBYTETRACK_TARGET=x5 -DCMAKE_BUILD_TYPE=Release
cmake --build build/x5 --target bytetrack -j
# 预期：build/x5/bytetrack
```

构建其他目标时，同时修改 `-B build/<target>` 和 `-DBYTETRACK_TARGET=<target>`。生成的程序位于 `build/<target>/bytetrack`。

启动脚本封装了同样的两条命令（在仓库根目录执行）：

```bash
bash samples/vision/bytetrack/runtime/cpp/run.sh --target x5 -- --help
```

<a id="run"></a>
## 运行

按[模型说明](../../model/README_cn.md)与[快速体验](../../README_cn.md#quickstart)准备模型和视频，然后在仓库根目录执行：

```bash
# X5：使用默认检测变体（n-v7.0）进行视频跟踪
bash samples/vision/bytetrack/runtime/cpp/run.sh --target x5 -- video \
  --input samples/vision/bytetrack/test_data/track_test.mp4 \
  --model-path samples/vision/bytetrack/model/yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin \
  --output result_x5.mp4 --records tracks_x5.tsv --threads 1
# 预期："video: <N> frames, <M> track rows"、result_x5.mp4、tracks_x5.tsv
```

S100 使用以下命令：

```bash
bash samples/vision/bytetrack/runtime/cpp/run.sh --target s100 -- video \
  --input samples/vision/bytetrack/test_data/track_test.mp4 \
  --model-path samples/vision/bytetrack/model/s100/yolov5x_672x672_nv12.hbm \
  --output result_s100.mp4 --records tracks_s100.tsv
```

S100P 和 S600 将命令中的 `s100` 替换为对应板卡名称，并使用相应模型文件。`--output` 用于保存带框视频，`--records` 用于保存轨迹记录，可分别选择。

host 目标无需板端 SDK，仅从检测文件运行跟踪器：

```bash
bash samples/vision/bytetrack/runtime/cpp/run.sh --target host -- replay \
  --detections persons.btcpp --records tracks.tsv --threads 1
# 预期："replay: <N> frames, <M> track rows, final frame_id <F>"
```

<a id="parameters"></a>
## 参数

命令格式为 `bytetrack <mode> [options]`。板端支持 `video` 和 `replay`，主机支持 `replay`。

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
| `--mot20` | 关闭 | MOT20 匹配模式（不融合检测分数） |
| `--bpu-cores I` | -1 | BPU 核心：-1 任意，或固定索引 0..3 |
| `--priority I` | 0 | BPU 任务优先级，0..255 |
| `--max-frames N` | 0 | 处理帧数上限；0 表示全部 |
| `--threads N` | 1 | 跟踪器内部线程数，包含调用线程 |

`--output` 和 `--records` 使用不同的文件路径，且不能覆盖输入视频、检测文件或模型。

X5 使用默认的 `--priority 0` 和 `--bpu-cores -1`。S 系列的 C++ `--bpu-cores` 接收单个核心索引，Python CLI 则接收索引列表。

<a id="interface-lifecycle"></a>
## 接口与资源生命周期

创建模型对象时加载资源，对象销毁时自动释放。按输入类型选择接口：

- 已有行人检测框：创建 `ByteTracker(config, threads)`，每帧调用 `update(detections, img_h, img_w)`。返回 `Track` 记录，包含目标 ID、原图坐标 `tlbr`、置信度和帧号。
- 输入视频帧：创建 `ByteTrackModel(options)`，每帧调用 `predict(frame)`。`options` 指定模型路径、检测阈值、跟踪参数和线程数。
- 只需要检测结果：调用 `ByteTrackModel::detect(frame)`，不更新轨迹。

`TrackingConfig` 的默认值为 `track_thresh=.3`、`track_buffer=60`、`match_thresh=.8`、`frame_rate=30`、`mot20=false`。参数越界时，创建跟踪器会抛出 `TrackingConfigError`；输入坐标或置信度为 NaN、框或图像尺寸无效时，`update` 抛出 `TrackingInputError`。

需要分步调用时，按 `preprocess(frame)` → `infer(prepared)` → `postprocess(raw, prepared)` 的顺序执行。`Frame` 保存 HWC 格式的 BGR 图像，`Prepared` 保存模型输入，`Raw` 保存推理输出。各返回对象持有自己的数据。

`--threads` 控制帧内卡尔曼预测和 IoU 计算的线程数，默认是 `1`。多路视频分别使用独立的模型实例和线程。

`VideoReader` 创建时打开视频；`VideoSink` 在首次 `write` 时打开输出文件，使用该帧的尺寸和 mp4v 编码。无法打开输入或输出文件时会抛出异常。

按视频顺序逐帧调用，即使某帧没有检测框，也应调用 `update`。切换视频时调用 `reset()` 清空轨迹并重置帧号，新目标 ID 继续递增。各 C++ 实例独立分配 ID；保存多路结果时，用 `(stream_id, track_id)` 区分目标。

<a id="results-interpretation"></a>
## 结果解释

运行完成后，程序打印 `video: N frames, M track rows` 或 `replay: N frames, M track rows, final frame_id F`，并返回退出码 `0`。命令行参数错误返回 `2`；模型加载、视频打开或记录写入失败返回 `1`。

`--records` 写出带表头 `frame lane track_id x1 y1 x2 y2 score` 的 TSV：帧号从 1 开始，CLI 的 `lane` 为 0，坐标为原图像素（`%.17g`）。无输出轨迹的帧没有 TSV 行，但仍推进跟踪状态。`--output` 写出带轨迹框与 `ID:<n>` 标签的 MP4，配色与 Python 渲染一致。

回放输入为小端二进制：8 字节 magic `BTCPP1\0\0`，四个 `uint32`（版本 1、帧数、图像高、图像宽）；每帧先写 `uint32` 检测数，再写每个检测的五个 `float32`（x1、y1、x2、y2、score）。坐标必须有限且框面积为正。

C++ 视频模式等比例缩放图像并以灰色 127 填充边缘（letterbox），输出浮点坐标。Python X5 模式直接缩放图像，将坐标裁剪到图像范围后取整。两种预处理方式可能产生不同的检测框和轨迹。共用检测结果时，直接调用 `ByteTracker::update`。

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

单核预算 = CPU 毫秒/帧 × FPS ÷ 10。单线程进程 RSS 约 8.5–12.4 MiB，包含动态库、输入序列、跟踪状态和计时记录；100 人档加载输入后的初始占用约为 11.1 MiB。

100 人输入下，帧内线程数与独立视频流数的表现如下：

| 配置 | 每路平均延迟（ms/帧） | 每路帧 CPU 时间（ms） | 总吞吐（路帧/秒） |
|---|---:|---:|---:|
| 单路，1 线程 | 1.912 | 1.912 | 522 |
| 单路，2 线程 | 1.897 | 1.962 | 526 |
| 单路，4 线程 | 1.850 | 1.963 | 539 |
| 单路，8 线程 | 1.915 | 1.912 | 521 |
| 2 路，每路 1 线程 | 1.955 | 1.950 | 1020 |
| 4 路，每路 1 线程 | 2.140 | 2.134 | 1864 |

CPU 时间按进程总 CPU 时间除以处理的总帧数计算；总吞吐为所有视频流每秒处理的帧数之和。100 人档中，单路 4 线程比单线程降低约 3% 延迟，4 路独立跟踪的总吞吐约为单路的 3.57 倍。多路应用可为每路保留默认 `--threads 1`，并行处理不同视频。4 路进程 RSS 约为 15.7 MiB。8 线程配置在此人数下仍使用串行计算。

CPU 调频保持系统默认，运行时间较短的小人数档位容易受调频与调度影响。以上是纯跟踪吞吐，完整视频速度还取决于检测和视频输入输出。
