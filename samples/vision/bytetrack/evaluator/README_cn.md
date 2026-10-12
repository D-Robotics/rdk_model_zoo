[English](README.md) | 简体中文

# ByteTrack 评估器

<a id="dataset"></a>
## 数据集

本工具比较同一模型、同一视频在两种实现下的检测和跟踪输出。输入视频可按 [快速体验](../README_cn.md#quickstart) 下载，也可以使用自己的视频。

<a id="directory"></a>
## 目录结构

```text
evaluator/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── capture.py  # 记录逐帧推理与跟踪输出
└── compare.py  # 比较两次运行结果
```

<a id="environment"></a>
## 环境

在 S 系列板卡上准备 HBM 模型和视频，安装 `hbm_runtime`、OpenCV、NumPy、SciPy、`lap==0.5.12` 和 `cython-bbox==0.1.5`。

<a id="command"></a>
## 评估命令

准备模型和视频后，在仓库根目录执行。`--output-dir` 请使用尚未存在的目录：

```bash
python3 samples/vision/bytetrack/evaluator/compare.py \
  --target s100 \
  --asset-id s:bytetrack:s100/yolov5x_672x672_nv12.hbm \
  --model-path samples/vision/bytetrack/model/s100/yolov5x_672x672_nv12.hbm \
  --input samples/vision/bytetrack/test_data/track_test.mp4 \
  --output-dir /tmp/bytetrack-comparison \
  --max-frames 30
```

工具分别运行当前实现（`unified`）和原始 S 系列实现（`legacy`）。原始实现从 Git 历史读取，使用浅克隆时先获取对应提交：

```bash
git fetch origin d2d2a4e0a898697bdfe5f68a9740a8c7d7cad57d
```

运行结果分别保存到 `unified/` 和 `legacy/`，对比结果写入 `comparison.json`。全部项目一致时退出码为 `0`。

<a id="metrics"></a>
## 指标

比较项目和容差：

| 项目 | 要求 |
|---|---|
| 输入帧、模型输入、目标 ID | 完全一致 |
| 模型输出 | 形状和数据类型一致，`rtol=0, atol=1e-5` |
| 轨迹框坐标 | `atol=1e-4` |
| 轨迹置信度 | `atol=1e-5` |

比较包含没有检测结果的帧。NaN 或无穷值会记为错误。

<a id="outputs"></a>
## 输出

每次运行保存 `.npy` 格式的帧图像、模型输入和输出，以及 `capture.json` 中的轨迹、模型和视频信息。`comparison.json` 汇总比较结果；出错时保留错误日志。

<a id="reference-results"></a>
## 参考结果

| 参考 | 条件 | 数值 | 来源 |
|---|---|---:|---|
| ByteTrack 论文 MOTA | MOT17 test / V100 | 80.3 | 论文 |
| ByteTrack 论文 IDF1 | MOT17 test / V100 | 77.3 | 论文 |
| ByteTrack 论文吞吐 | V100 GPU | about 30 FPS | 论文 |

### 参考跟踪效果

以下为 ByteTrack 在 MOT17 两段视频上的跟踪效果：

![MOT17-01-SDP](../test_data/readme_img/MOT17-01-SDP.gif)

MOT17-01-SDP。

![MOT17-07-SDP](../test_data/readme_img/MOT17-07-SDP.gif)

MOT17-07-SDP。

### 跟踪参数

- `--score-thres`（默认 `0.25`）：检测置信度阈值。行人漏检时可调低。
- `--track-thresh`（默认 `0.3`）：高分检测阈值。高分框先与已有轨迹匹配，分数在 `0.1` 和该阈值之间的框用于第二轮匹配。未匹配框的分数达到 `track_thresh + 0.1` 时，可以创建新轨迹。
- `--match-thresh`（默认 `0.8`）：第一轮匹配的最大代价。调大可接受重叠较少的框；调小要求框更接近。默认结合 IoU 和检测置信度，`--mot20` 改为只使用 IoU。第二轮匹配的代价上限为 `0.5`。
- `--track-buffer`（默认 `60`）：丢失目标后保留轨迹的时长，以 30 FPS 下的帧数表示。实际帧数按 `--frame-rate`（默认 `30`）缩放。

本示例跟踪 COCO 行人类别（类别编号 `0`）。扩展到多类别时，可为每个类别创建独立跟踪器。

<a id="boundaries"></a>
## 适用范围

两次运行使用同一板卡上的同一份模型和视频。工具比较输出一致性，不计算 MOTA、IDF1 等数据集精度指标。
