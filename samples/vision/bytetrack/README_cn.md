[English](README.md) | 简体中文

# ByteTrack

<a id="overview"></a>
## 算法与来源

ByteTrack 通过关联高置信度和低置信度检测框，在视频帧之间跟踪目标。恢复低分检测有助于在行人被遮挡时保持身份连续。本示例组合 YOLOv5 行人检测器与 CPU BYTETracker。

参考：[ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864).

<a id="directory"></a>
## 目录结构

```text
bytetrack/
├── conversion/  # 导出与量化配置
├── evaluator/  # 评估程序与指标
├── model/  # 模型文件与下载脚本
├── runtime/  # 推理程序
├── test_data/  # 示例输入
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
└── requirements-host.txt  # 源码或数据文件
```

<a id="support-matrix"></a>
## 支持矩阵

| 目标 | 变体 | Python | C++ | 说明 |
|---|---|---|---|---|
| S100 | YOLOv5x 672 | 支持 | 支持 | 对应板卡 HBM |
| S100P | YOLOv5x 672 | 支持 | 支持 | 对应板卡 HBM |
| S600 | YOLOv5x 672 | 支持 | 支持 | 对应板卡 HBM |
| X5 | YOLOv5 640，默认 n-v7.0 | 支持 | 支持 | 复用 X5 YOLOv5 `.bin` 模型 |



tracker 有状态：同一个 `ByteTrackTask` 必须按顺序处理帧。`reset` 清除流历史和 frame index，但故意不把进程级 track ID 计数器归零。

<a id="prerequisites"></a>
## 环境前提

需要 Python 3、NumPy、SciPy、OpenCV、`lap==0.5.12` 和 `cython-bbox==0.1.5`。X5 使用系统镜像提供的 `hbm_runtime` 和 640×640 packed-NV12 `.bin` 模型；S 系列使用 `hbm_runtime` 和 672×672 split-NV12 `.hbm` 模型。准备依赖：

```bash
python3 -m pip install -r samples/vision/bytetrack/requirements-host.txt
```

<a id="quickstart"></a>
## 快速体验

在仓库根目录准备 X5 模型和输入视频，然后运行：

```bash
python3 -m samples.vision.bytetrack.model.download --target x5 --variant n-v7.0
curl -fL 'https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ByteTrack/track_test.mp4' \
  -o samples/vision/bytetrack/test_data/track_test.mp4
python3 -m samples.vision.bytetrack.runtime.python.main \
  --target x5 --variant n-v7.0 \
  --input samples/vision/bytetrack/test_data/track_test.mp4 \
  --output samples/vision/bytetrack/test_data/result_x5.mp4 \
  --records samples/vision/bytetrack/test_data/result_x5.jsonl
```

`--variant` 支持 `n-v7.0`、`s-v7.0`、`m-v7.0`、`l-v7.0`、`x-v7.0`，以及 `s-v2.0`、`m-v2.0`、`l-v2.0`、`x-v2.0`。下载和运行时选择同一个变体。

S100 使用以下命令：

```bash
python3 -m samples.vision.bytetrack.model.download --target s100 \
  --output-dir samples/vision/bytetrack/model
curl -L 'https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ByteTrack/track_test.mp4' \
  -o samples/vision/bytetrack/test_data/track_test.mp4
python3 -m samples.vision.bytetrack.runtime.python.main \
  --target s100 --asset-id s:bytetrack:s100/yolov5x_672x672_nv12.hbm \
  --model-path samples/vision/bytetrack/model/s100/yolov5x_672x672_nv12.hbm \
  --input samples/vision/bytetrack/test_data/track_test.mp4 \
  --output samples/vision/bytetrack/test_data/result_unified.mp4 \
  --records samples/vision/bytetrack/test_data/result_unified.jsonl
```

公开 `track_test.mp4` 的记录 SHA-256 为 `4bbe5bf11fe8967b28a900fd2add4949aba89b62076eaa03d0c55cdf7dd41397`；需要校验完整性时执行 `sha256sum track_test.mp4`（macOS：`shasum -a 256 track_test.mp4`）。成功推理会写出可解码 MP4 和可选逐帧 JSONL，打印帧数并退出 `0`。`run.sh` 不会安装或下载。

<a id="expected-results"></a>
## 预期结果

每帧输出零个或多个 person track，字段为 `track_id`、原图 `tlbr`、score 和 frame index，并写出指定视频。空检测仍推进 `frame_index` 并更新 tracker。若框完全落在 letterbox padding，clip 后可能出现零面积，XYAH 初始化会产生 NaN；task 在更新前剔除宽或高不正的 person 框。

适用条件与调参。`--score-thres`（默认 `0.25`）在 tracker 之前过滤检测框：检出框过少时调低它——调低 `--track-thresh` 找不回被 detector 丢弃的框。`--track-thresh`（`0.3`）只划分 tracker 输入：高于它的框进入第一次关联，(0.1, `track-thresh`) 区间的框与仍在跟踪的目标进行第二次关联，新轨迹只从得分不低于 `track_thresh + 0.1` 的第一次关联框初始化。若 track ID 频繁切换，可考虑调大 `--match-thresh`（`0.8`，第一次关联接受的最大代价——1 − IoU，默认模式与检测分数融合，`--mot20` 时不融合；越大允许越不相似的匹配）或加长 `--track-buffer`（`60`，丢失轨迹窗口，按 `frame_rate / 30` 缩放）。这些是调参方向，不是重新标定的阈值。当前流程只跟踪 COCO `person`；多类别跟踪需要每类一个 tracker 或扩展 tracker 使其感知类别（见 [evaluator 说明](evaluator/README_cn.md)）。

检测器保留 COCO `person` 类别 `0`。BYTETracker 先关联高分框，再用 IoU 将剩余轨迹与低分框关联；只有未匹配的高分框会初始化新轨迹。

![ByteTrack 检测示例条带](test_data/readme_img/image1.png)
![(a)/(b)/(c) 三行关联示意](test_data/readme_img/image.png)

<a id="entry-points"></a>
## 入口索引

- [`model/README_cn.md`](./model/README_cn.md)：X5 BIN、S 系列 HBM 与模型准备。
- [`runtime/python/README_cn.md`](./runtime/python/README_cn.md)：有状态 Task API 与完整 CLI。
- [`runtime/cpp/README_cn.md`](./runtime/cpp/README_cn.md)：X5/S100/S100P/S600 的 C++ 视频跟踪、主机检测回放、CPU 跟踪接口与线程配置。
- [`conversion/README_cn.md`](./conversion/README_cn.md)：仅 detector 的转换边界。
- [`evaluator/README_cn.md`](./evaluator/README_cn.md)：评测命令、输出与对照说明。

<a id="historical-performance"></a>
## 性能与资源占用

X5 多人跟踪可使用 [C++ Runtime](runtime/cpp/README_cn.md#x5-cpu-performance)：100 人档单线程平均约 1.91 ms，30 FPS 约占一个 CPU 核心的 5.7%。默认采用单线程，多路应用优先并行独立视频流。检测模型在 BPU 上执行，轨迹预测和关联在 CPU 上执行。以下 Python 数据供选择运行方式时参考。

### 跟踪计算

以下为 2026-10-12 的板端运行数据，仅统计 `BYTETracker.update`。每档使用相同的合成人员运动序列，包含低分检测、遮挡及空帧；预热 100 帧后记录 1200 帧，重复两轮。人数列为生成上限，活跃轨迹列为输出均值。延迟统计合并两轮数据。

| 板卡 | 人数上限 | 平均活跃轨迹 | 平均 / P95 延迟（ms） | CPU 时间（ms/帧） | 30 FPS 单核预算 | 60 FPS 单核预算 |
|---|---:|---:|---:|---:|---:|---:|
| X5 | 5 | 4.6 | 4.46 / 5.29 | 4.43 | 13.3% | 26.6% |
| X5 | 20 | 18.2 | 12.91 / 14.50 | 12.86 | 38.6% | 77.2% |
| X5 | 50 | 44.6 | 30.12 / 33.93 | 30.02 | 90.1% | 180.1% |
| X5 | 100 | 90.7 | 60.51 / 68.04 | 60.34 | 181.0% | 362.0% |
| S100 | 5 | 4.6 | 1.47 / 1.72 | 1.46 | 4.4% | 8.7% |
| S100 | 20 | 18.2 | 3.60 / 4.00 | 3.57 | 10.7% | 21.4% |
| S100 | 50 | 44.6 | 7.63 / 8.64 | 7.58 | 22.7% | 45.5% |
| S100 | 100 | 90.7 | 14.71 / 16.45 | 14.63 | 43.9% | 87.8% |

单核预算 = CPU 毫秒/帧 × FPS ÷ 10，以单核 100% 为基准。超过 100% 表示当前串行跟踪不能达到该帧率，并不表示会自动使用多个核心。整机归一化预算可再除以核心数：X5 为 8，S100 为 6；低整机占比不能消除单条跟踪流的延迟。

运行环境为 Ubuntu 22.04、Python 3.10.12、NumPy 1.26.4、lap 0.5.13、cython-bbox 0.1.5，BLAS/OpenMP 线程数为 1。X5 使用 SciPy 1.10.1，S100 使用 1.15.3；此处为这两套环境的表现，不能将差异全部归因于 CPU 硬件。CPU 频率上限均为 1.5 GHz，S100 部分采样为 1.125 GHz。S100 存在后台负载，频率与负载没有强制固定。安装依赖时仍按本 Sample 的 requirements 文件选择版本。

跟踪进程采样 RSS：X5 约 49–63 MiB，S100 约 69–85 MiB。加载解释器、依赖及输入序列后的基线分别为 46.8 / 68.9 MiB；RSS 包含这些内容和测量记录，不能当作算法独占内存或每条轨迹内存。各人数档位在同一进程顺序运行，分配器可能保留此前申请的内存。

### 视频处理链路

输入为本 Sample 的 `track_test.mp4`，1280×720、30 FPS。每轮预热 30 帧后处理 300 帧，共两轮；下表为两轮平均耗时。使用默认检测与跟踪阈值，OpenCV `mp4v` 编码并写入结果视频。

| 阶段 | X5：YOLOv5n v7.0 640（ms/帧） | S100：YOLOv5x 672（ms/帧） |
|---|---:|---:|
| 视频解码 | 7.99 | 2.92 |
| 检测预处理 | 2.52 | 2.59 |
| 模型 Runtime 调用 | 22.28 | 24.05 |
| 检测后处理 | 19.55 | 36.78 |
| 行人筛选与跟踪更新 | 6.60 | 3.01 |
| 画框 | 2.50 | 1.36 |
| 视频编码与写入 | 70.56 | 21.31 |
| 合计 | 132.02 | 92.03 |

平均活跃轨迹分别为 5.72 / 10.94，检测模型不同，总耗时不可用作相同检测任务的硬件对比。包含视频输出的运行吞吐约为 7.5 / 10.8 FPS；输入文件的 30 FPS 不代表处理达到 30 FPS。运行窗口的进程 CPU 占用约为 125% / 104% 单核等效，整机归一化约为 15.7% / 17.4%。CPU 统计窗口包含编码器关闭时的收尾开销，分段耗时不含此项。测量帧期间采样 RSS 峰值为 146.9 / 163.9 MiB，进程生命周期高水位为 146.9 / 191.6 MiB。以上含解码、画框和编码，不能算作 ByteTrack 单独占用。

X5 使用 HBRT 3.15.55.0，模型构建版本为 3.15.54.0；S100 使用 UCP 3.13.6。模型与 SDK 版本会影响运行表现。

### CPU 与 BPU 分工

保留 CPU 跟踪，不建议将整套 ByteTrack 直接迁入 BPU。当前关联包含小矩阵 Kalman 更新、IoU、LAPJV 分配和变长轨迹状态管理；迁移还需要处理 CPU/BPU 之间的数据传输、同步及算子适配。这里没有 BPU 版跟踪器的性能数据，不能假设迁移后一定更快。

X5 面向高密度人群或多路输入时，可选择 C++ 跟踪器；上述 Python 50 人档位的 P95 已超过 33.3 ms。S100 单路 30 FPS 可先保留当前实现；100 人、60 FPS 时约占 88% 单核，也需要考虑延迟波动。示例视频链路应先减少软件视频编码和检测后处理的开销。检测网络，以及以后按需加入的 ReID 网络，继续作为 BPU 加速对象。

<a id="license"></a>
## 许可

仓库 脚本 遵循 Apache-2.0。随附 tracker 源码树没有单独的 license 文件；其来源记录在 [`runtime/python/TRACKER_SOURCE_MAP.json`](./runtime/python/TRACKER_SOURCE_MAP.json)。上游 ByteTrack 与模型权重遵循各自许可。
