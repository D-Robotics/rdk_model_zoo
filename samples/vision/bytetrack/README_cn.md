[English](./README.md) | 简体中文

# ByteTrack

<a id="overview"></a>
## 算法与来源

ByteTrack 是有状态多目标跟踪器，通过高分和低分检测框关联目标。本 sample 在 S100/S100P/S600 上运行 YOLOv5x，保留 COCO `person` 类别 `0`，再更新 CPU BYTETracker。论文为 [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864)。

多目标跟踪（MOT）需要在视频帧间同时估计目标的边界框和身份。多数跟踪方法只关联高分检测框、丢弃低分检测框，导致被遮挡目标丢失、轨迹碎片化。ByteTrack 的 BYTE 策略（Tracking By associating Almost Every Detection Box）改为同时保留两组检测框：

- 保留高分和低分检测框；
- 第一次关联：高置信度检测框与已有轨迹匹配；
- 第二次关联：未匹配轨迹与低置信度检测框按 IoU 匹配；
- 新轨迹只从未匹配的高分检测框初始化。

固定 S 源 README 携带的检测示例（`test_data/readme_img/image1.png`，S pin `380e1a2`，sha256 `fdab9b40…`）是同一街区三帧的横向条带：彩色框带逐检测置信度（如左帧 0.94/0.92/0.83，中帧红三角旁低至 0.43），三角标记（左右帧为黄色、中间帧为红色）标注一名被跟随的行人。低分恢复的含义由下方三行图说明：该图以 `test_data/readme_img/image.png`（sha256 `032728fb…`）随固定源树携带，但未嵌入 pinned 源 README；其可见图注与论文动机示例一致——(a) detection boxes，其中被跟踪的较小行人分数为 t1 0.8、t2 0.4、t3 0.1（0.9 框属于另一名较高的前景行人）；(b) tracklets by associating high score detection boxes；(c) tracklets by associating every detection box，该行人的低分检测（虚线框，标注 0.4 与 0.1）被重新关联。

![ByteTrack detection example strip embedded by the source README](test_data/readme_img/image1.png)

![Three-row (a)/(b)/(c) association illustration bundled in the source tree, not embedded by the source README](test_data/readme_img/image.png)

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

前两条是显式准备命令，本轮未执行。成功推理会写出可解码 MP4 和可选逐帧 JSONL，打印帧数并退出 `0`。`run.sh` 不会安装或下载。

<a id="expected-results"></a>
## 预期结果

每帧输出零个或多个 person track，字段为 `track_id`、原图 `tlbr`、score 和 frame index，并写出指定视频。空检测仍推进 `frame_index` 并更新 tracker。若框完全落在 letterbox padding，clip 后可能出现零面积，源 XYAH 初始化会产生 NaN；统一 task 在更新前剔除宽或高不正的 person 框。evaluator 遇到源 NaN 会记录 error 并失败，不把它当成相等。

适用条件与调参（沿用源结果判读建议，并已对照本 sample 的 tracker 代码核实）。`--score-thres`（默认 `0.25`）在 tracker 之前过滤检测框：检出框过少时调低它——调低 `--track-thresh` 找不回被 detector 丢弃的框。`--track-thresh`（`0.3`）只划分 tracker 输入：高于它的框进入第一次关联，(0.1, `track-thresh`) 区间的框与仍在跟踪的目标进行第二次关联，新轨迹只从得分不低于 `track_thresh + 0.1` 的第一次关联框初始化。若 track ID 频繁切换，可考虑调大 `--match-thresh`（`0.8`，第一次关联接受的最大代价——1 − IoU，默认模式与检测分数融合，`--mot20` 时不融合；越大允许越不相似的匹配）或加长 `--track-buffer`（`60`，丢失轨迹窗口，按 `frame_rate / 30` 缩放）。这些是调参方向，不是重新标定的阈值。当前流程只跟踪 COCO `person`；多类别跟踪需要每类一个 tracker 或扩展 tracker 使其感知类别（见 [evaluator 说明](evaluator/README_cn.md)）。

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
