[English](./README.md) | 简体中文

# PaddleOCR 两阶段文字检测与识别

<a id="overview"></a>
## 概述

本 sample 在 RDK 板卡上运行完整的两阶段 OCR 流程：DB 检测器在输入图像中
找出文字区域，逐个裁剪后由 CRNN+CTC 识别器解码为字符串。它是仓库中
合法多阶段推理的参照实现（见[阶段 I/O](./runtime/python/README.md#stage-io)）：
检测与识别是两个独立懒加载的运行时阶段，由显式 pipeline 组合，
检测→裁剪→识别的顺序全程可读。

维护两个经过审计的模型对。一对 = 检测器 + 识别器 + 词典，作为整体使用；
禁止跨对混用组件：

| 板卡 | 模型对 | 检测器输入 | 识别器输出 |
| --- | --- | --- | --- |
| RDK X5 | PP-OCRv3 英文 | 单个 packed NV12 张量（640×640） | F32 `[1,40,97,1]`：固定 96 字符字母表加 blank |
| RDK S100 | PP-OCRv6 | split NV12 `x_y`（640×640）+ `x_uv`（320×320） | F32 `[1,40,18710]`：随仓 UTF-8 词典加 blank 与末尾空格 |

旧 X5/S Python 入口与旧 S C++ 源码均转发到本 canonical 实现；它们仍是
可用的兼容入口，本 sample 不维护第二套实现。

<a id="support-matrix"></a>
## 支持矩阵

| 板卡 | Python 运行时 | C++ 运行时 |
| --- | --- | --- |
| X5 | supported-verified | not-supported（审计基线中无 X5 C++ 源码） |
| S100 | supported-verified | supported-verified |
| S100P | not-supported（未发布经审计的 OCR 模型对） | not-supported |
| S600 | supported-not-run | supported-not-run |

验证状态：Python 默认与保持长宽比两条管线及旧包装入口在两块 X5 板与
S100 上做过逐字节一致验证（2026-09-17 集成评审）；S100 C++ 完成构建运行，
渲染输出像素与源基线一致。S600 与 S100 共享源码与 SoC 探测，但板卡不可达
（SSH 未恢复），保持 `not-run`，不能用 S100 结果替代。S100P 在两侧发布
清单中均无匹配的审计模型对，sample 对其显式拒绝。

<a id="prerequisites"></a>
## 环境前提

- 与目标匹配、自带 `hbm_runtime` 包的板卡镜像（X5 镜像对应 `.bin`
  模型对，RDK S 镜像对应 `.hbm` 模型对）。
- Python 3.10 或更新，含 NumPy、OpenCV-Python、PyYAML；检测器返回
  至少一个框时另需 `pyclipper`（help/list/dry-run 模式不需要任何板端
  SDK 包）。
- 已准备的模型制品（见[入口](#entry-points)）；推理路径不联网下载。
- 以下命令均在完整检出的仓库根目录执行。

<a id="quickstart"></a>
## 快速体验

1. 列出清单支撑的模型对（无需 SDK；成功：打印两行带完整引用的结果）：

   ```bash
   python3 samples/vision/paddle_ocr/runtime/python/main.py \
     --list-models --target auto
   ```

2. 显式准备 X5 模型对——唯一可联网的操作（输入：清单 URL；
   输出：`--model-dir` 下两个 `.bin`；成功：退出码 0 且打印实测
   SHA-256）：

   ```bash
   python3 samples/vision/paddle_ocr/runtime/python/main.py --prepare \
     --target x5 \
     --det-asset-id x5:paddleocr:en_PP-OCRv3_det_640x640_nv12.bin \
     --rec-asset-id x5:paddleocr:en_PP-OCRv3_rec_48x320_rgb.bin \
     --model-dir /tmp/rdk-models
   ```

3. 运行 X5 模型对（cwd：仓库根目录；成功：退出码 0，stdout 打印识别
   字符串与多边形框）：

   ```bash
   python3 samples/vision/paddle_ocr/runtime/python/main.py \
     --target x5 \
     --det-asset-id x5:paddleocr:en_PP-OCRv3_det_640x640_nv12.bin \
     --rec-asset-id x5:paddleocr:en_PP-OCRv3_rec_48x320_rgb.bin \
     --det-model-path /tmp/rdk-models/en_PP-OCRv3_det_640x640_nv12.bin \
     --rec-model-path /tmp/rdk-models/en_PP-OCRv3_rec_48x320_rgb.bin \
     --test-img samples/vision/paddle_ocr/test_data/x5/paddleocr_test.jpg \
     --output-format json
   ```

4. S100 上整体切换 target、引用、路径与测试图（下列模型对使用 S 镜像
   `/opt/hobot/model/s100/basic` 下已有的制品）：

   ```bash
   python3 samples/vision/paddle_ocr/runtime/python/main.py \
     --target s100 \
     --det-asset-id s:paddle_ocr:s100/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm \
     --rec-asset-id s:paddle_ocr:s100/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm \
     --det-model-path /opt/hobot/model/s100/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm \
     --rec-model-path /opt/hobot/model/s100/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm \
     --test-img samples/vision/paddle_ocr/test_data/s100/gt_2322.jpg \
     --output-format json
   ```

5. 主机契约测试（无需板卡；成功：全部 OK，退出码 0）：

   ```bash
   python3 -m unittest discover -s samples/vision/paddle_ocr/tests -v
   ```

<a id="expected-results"></a>
## 预期结果

文本输出逐行打印识别字符串及其有序多边形框。JSON 输出包含 `target`、
`image_shape`、`detector_asset`、`recognizer_asset` 与对齐的
`boxes`/`texts`，例如：

```json
{
  "target": "x5",
  "boxes": [[[20, 30], [180, 30], [180, 70], [20, 70]]],
  "texts": ["RDK"]
}
```

框与文本保持检测器顺序；结果中的数组归结果所有。检测器输出为空时跳过
识别并返回空列表。检测器输出按观测到的 score map 直接阈值化（`0.5`）；
不添加未经验证的激活或精度声明。随仓测试图的具体打印内容属于模型对的
属性——已核验的对照记录见[评估](./evaluator/README.md#reference-results)；
数据集级精度在本 sample 为 **not-run**。

<a id="directory"></a>
## 目录

| 路径 | 职责 |
| --- | --- |
| `model/` | 制品引用与显式准备流程 |
| `runtime/python/` | canonical 两阶段 Python 运行时（上表全部目标） |
| `runtime/cpp/` | S 系列原生运行时（DB + CRNN/CTC + FreeType 渲染） |
| `conversion/` | 按 target 分开的导出/校准/编译配方 |
| `evaluator/` | 基于记录的检测/识别一致性评估器 |
| `test_data/` | 随仓测试材料：X5 图像、S100 图像与 PP-OCRv6 词典 |
| `tests/` | 主机契约测试（43 例） |

<a id="entry-points"></a>
## 入口

- 模型：[model/README.md](./model/README.md)——四条制品引用与
  `--prepare` 流程。
- Python 运行时：[runtime/python/README.md](./runtime/python/README.md)——
  完整参数表、集成示例与阶段 I/O 契约。
- C++ 运行时：[runtime/cpp/README.md](./runtime/cpp/README.md)——S 系列
  可执行文件的构建、运行、gflags 与生命周期。
- 转换：[conversion/README.md](./conversion/README.md)——PP-OCRv3
  `hb_mapper` 与 PP-OCRv6 `hb_compile` 配方。
- 评估：[evaluator/README.md](./evaluator/README.md)——标注记录评估
  及其边界。

<a id="license"></a>
## 许可

Sample 代码遵循仓库许可。模型制品经平台发布清单发布；PP-OCRv3 与
PP-OCRv6 权重为 PaddlePaddle 上游发布，其使用受对应上游许可约束。
S100 词典与 C++ 演示字体自审计源交付原样携带。
