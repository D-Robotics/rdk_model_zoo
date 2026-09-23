[English](./README.md) | 简体中文

# ByteTrack

<a id="overview"></a>
## 算法与来源

ByteTrack 是有状态多目标跟踪器，通过高分和低分检测框关联目标。本 sample 在 S100/S100P/S600 上运行 YOLOv5x，保留 COCO `person` 类别 `0`，再更新 CPU BYTETracker。论文为 [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864)。

<a id="support-matrix"></a>
## 支持与验证矩阵

| target | variant | Python | C++ | 状态 |
|---|---|---|---|---|
| S100 | YOLOv5x 672 | supported-not-run | not-supported | 主机 tracker fixture；板测未运行 |
| S100P | YOLOv5x 672 | supported-not-run | not-supported | 主机 tracker fixture；板测未运行 |
| S600 | YOLOv5x 672 | supported-not-run | not-supported | 主机 tracker fixture；板测未运行 |
| X5 | — | not-supported | not-supported | 没有 ByteTrack 制品 |

tracker 有状态：同一个 `ByteTrackTask` 必须按顺序处理帧。`reset()` 清除流历史和 frame index，但故意不把进程级 track ID 计数器归零。

<a id="prerequisites"></a>
## 环境前提

主机 tracker 检查需要 Python、NumPy、SciPy、OpenCV、`lap==0.5.12` 和 `cython-bbox==0.1.5`。板端还需要目标 `hbm_runtime` 和精确 HBM。源视频不在 `test_data`，需显式从 `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ByteTrack/track_test.mp4` 准备。本轮没有下载模型或视频。

<a id="quickstart"></a>
## 快速体验

在仓库根目录显式准备 S100 模型和视频，然后运行：

```bash
python3 -m samples.vision.bytetrack.model.download --target s100 \\
  --output-dir samples/vision/bytetrack/model
curl -L 'https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ByteTrack/track_test.mp4' \\
  -o samples/vision/bytetrack/test_data/track_test.mp4
python3 -m samples.vision.bytetrack.runtime.python.main \\
  --target s100 --asset-id s:bytetrack:s100/yolov5x_672x672_nv12.hbm \\
  --model-path samples/vision/bytetrack/model/s100/yolov5x_672x672_nv12.hbm \\
  --input samples/vision/bytetrack/test_data/track_test.mp4 \\
  --output samples/vision/bytetrack/test_data/result_unified.mp4 \\
  --records samples/vision/bytetrack/test_data/result_unified.jsonl
```

前两条是显式准备命令，本轮未执行。成功推理会写出可解码 MP4 和可选逐帧 JSONL，打印帧数并退出 `0`。`run.sh` 不会安装或下载。

<a id="expected-results"></a>
## 预期结果

每帧输出零个或多个 person track，字段为 `track_id`、原图 `tlbr`、score 和 frame index，并写出指定视频。空检测仍推进 `frame_index` 并更新 tracker。若框完全落在 letterbox padding，clip 后可能出现零面积，源 XYAH 初始化会产生 NaN；统一 task 在更新前剔除宽或高不正的 person 框。evaluator 遇到源 NaN 会记录 error 并失败，不把它当成相等。

<a id="directory"></a>
## 目录职责

```text
.
├── model/                 # 显式 S HBM 准备
├── runtime/python/        # detector binding、tracker 状态、CLI、source map
├── conversion/            # 仅 detector 的转换边界和 OE 资源
├── evaluator/             # 新进程完整捕获/对照
├── test_data/              # 图片、标签、参考 GIF；视频需外部准备
└── tests/                 # CPU tracker、源对照、CLI、证据 fixture
```

<a id="entry-points"></a>
## 入口索引

- [`model/README_cn.md`](./model/README_cn.md)：三个 target 的 HBM 和显式准备。
- [`runtime/python/README_cn.md`](./runtime/python/README_cn.md)：有状态 Task API 与完整 CLI。
- [`conversion/README_cn.md`](./conversion/README_cn.md)：仅 detector 的转换边界。
- [`evaluator/README_cn.md`](./evaluator/README_cn.md)：新进程源/统一证据和严格 ID 比较。

<a id="historical-performance"></a>
## 源历史性能

源记录 RDK S100 tracker update 约 `2.37 ms`。论文报告 V100 GPU 上 `80.3 MOTA`、`77.3 IDF1`、约 `30 FPS`。这些是历史参考，不是本迁移的板端测量。

<a id="license"></a>
## 许可

仓库 wrapper 遵循 Apache-2.0。固定源 inventory 没有独立 tracker LICENSE；其来源和 `TRACKER_SOURCE_MAP.json` 均保留。上游 ByteTrack 与模型权重遵循各自许可。
