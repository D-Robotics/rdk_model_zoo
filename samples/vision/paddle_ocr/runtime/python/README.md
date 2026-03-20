# PaddleOCR 文本检测与识别示例（Python）

本示例展示如何在 BPU 上使用量化后的 PaddleOCR 模型执行中文文本检测与识别。采用两阶段 OCR 流程：DB 算法检测文本区域，CRNN+CTC 识别文字内容。

> ⚠️ **平台说明**：本模型仅支持 **RDK S100** 平台。若使用 RDK S600，请参阅[注意事项](#注意事项)。

## 环境依赖

本样例需要以下 Python 依赖：

```bash
pip install numpy==1.26.4 opencv-python==4.11.0.86 pyclipper==1.3.0.post6 Pillow==9.0.1
```

## 目录结构

```text
.
├── paddle_ocr.py   # PaddleOCR 推理封装（检测与识别模型 + 辅助函数）
├── main.py         # 推理程序入口（参数解析与流程控制）
├── run.sh          # 示例运行脚本（自动安装依赖、下载模型、运行）
└── README.md       # 使用说明
```

## 参数说明

| 参数                  | 说明                                       | 默认值                                                                          |
|----------------------|---------------------------------------------|--------------------------------------------------------------------------------|
| `--det-model-path`   | 检测模型文件路径（.hbm 格式）               | `/opt/hobot/model/s100/basic/cn_PP-OCRv3_det_infer-deploy_640x640_nv12.hbm`  |
| `--rec-model-path`   | 识别模型文件路径（.hbm 格式）               | `/opt/hobot/model/s100/basic/cn_PP-OCRv3_rec_infer-deploy_48x320_rgb.hbm`    |
| `--test-img`         | 测试图片路径                                | `../../test_data/gt_2322.jpg`                                                  |
| `--label-file`       | 字符词典文件路径（每行一个字符）             | `../../test_data/ppocr_keys_v1.txt`                                            |
| `--threshold`        | 检测掩码二值化阈值（0.0–1.0）               | `0.5`                                                                          |
| `--ratio-prime`      | 轮廓扩张比例系数                            | `2.7`                                                                          |
| `--img-save-path`    | 结果图像保存路径                            | `result.jpg`                                                                   |
| `--priority`         | 模型调度优先级（0~255）                     | `0`                                                                            |
| `--bpu-cores`        | 使用的 BPU 核心编号列表                     | `[0]`                                                                          |

> **注意**：字体文件固定使用 `../../test_data/FangSong.ttf`，不作为命令行参数暴露。

## 快速运行

### 方式一：一键运行（推荐）

```bash
cd runtime/python/
./run.sh
```

脚本会自动完成：环境检测 → 模型下载 → 推理执行。

### 方式二：手动运行

- 使用默认参数

    ```bash
    python3 main.py
    ```

- 指定参数运行

    ```bash
    python3 main.py \
        --det-model-path /opt/hobot/model/s100/basic/cn_PP-OCRv3_det_infer-deploy_640x640_nv12.hbm \
        --rec-model-path /opt/hobot/model/s100/basic/cn_PP-OCRv3_rec_infer-deploy_48x320_rgb.hbm \
        --test-img ../../test_data/gt_2322.jpg \
        --label-file ../../test_data/ppocr_keys_v1.txt \
        --img-save-path result.jpg \
        --threshold 0.5 \
        --ratio-prime 2.7
    ```

### 输出结果

运行成功后，结果将保存至当前目录：

```text
[0] Prediction: 示例文字
[1] Prediction: 另一行文字
...
[Saved] Result saved to: result.jpg
```

- `result.jpg`：左侧为带检测框的原图，右侧为白色画布上的识别文字

## 接口说明

阅读[源码文档说明](../../../../../docs/source_reference/README.md)，根据说明查看源码参考文档。

## 注意事项

- **平台兼容性**：本模型仅支持 RDK S100 平台，**不支持 RDK S600**。若需在 S600 上运行，需使用 S600 工具链重新量化编译模型。
- 若模型文件不存在，`run.sh` 会自动从 D-Robotics 下载中心下载 S100 模型。
- 需安装 `pyclipper`（多边形扩张）和 `Pillow`（中文字体渲染）。
- 字体渲染使用仿宋字体（`FangSong.ttf`），路径固定为 `../../test_data/FangSong.ttf`。
- 识别结果使用纯 NumPy 实现的 CTC 贪婪解码，不依赖 PaddlePaddle。
