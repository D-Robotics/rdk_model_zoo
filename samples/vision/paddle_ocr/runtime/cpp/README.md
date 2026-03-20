# PaddleOCR 文本检测与识别示例（C++）

本示例展示如何在 BPU 上使用量化后的 PaddleOCR 模型执行中文文本检测与识别。采用两阶段 OCR 流程：DB 算法检测文本区域，CRNN+CTC 识别文字内容。

> ⚠️ **平台说明**：本模型仅支持 **RDK S100** 平台。若使用 RDK S600，请参阅[注意事项](#注意事项)。

## 环境依赖

```bash
sudo apt install libgflags-dev libpolyclipping-dev libfreetype6-dev
```

## 目录结构

```text
.
├── inc/
│   └── paddle_ocr.hpp     # PaddleOCR 模型封装接口与函数声明
├── src/
│   ├── paddle_ocr.cpp     # PaddleOCR 推理与前后处理实现
│   └── main.cpp           # 推理程序入口（参数解析与流程控制）
├── CMakeLists.txt         # CMake 构建配置
├── run.sh                 # 示例运行脚本（自动安装依赖、下载模型、编译、运行）
└── README.md              # 使用说明
```

## 编译工程

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

编译产物为 `build/paddle_ocr`。

## 参数说明

| 参数                | 说明                                    | 默认值                                                                          |
|--------------------|-----------------------------------------|--------------------------------------------------------------------------------|
| `--det_model_path` | 检测模型文件路径（.hbm 格式）            | `/opt/hobot/model/s100/basic/cn_PP-OCRv3_det_infer-deploy_640x640_nv12.hbm`  |
| `--rec_model_path` | 识别模型文件路径（.hbm 格式）            | `/opt/hobot/model/s100/basic/cn_PP-OCRv3_rec_infer-deploy_48x320_rgb.hbm`    |
| `--test_image`     | 测试图片路径                             | `../../../test_data/gt_2322.jpg`                                               |
| `--label_file`     | 字符词典文件路径（每行一个字符）          | `../../../test_data/ppocr_keys_v1.txt`                                         |
| `--threshold`      | 检测掩码二值化阈值（0.0–1.0）            | `0.5`                                                                          |
| `--ratio_prime`    | 轮廓扩张比例系数                         | `2.7`                                                                          |
| `--img_save_path`  | 结果图像保存路径                         | `result.jpg`                                                                   |
| `--font_path`      | 渲染识别文字使用的 TTF 字体路径           | `../../../test_data/FangSong.ttf`                                              |

> **注意**：`--det_model_path` 和 `--rec_model_path` 默认路径固定为 S100 模型路径，不随平台自动切换。

## 快速运行

### 方式一：一键运行（推荐）

```bash
cd runtime/cpp/
./run.sh
```

脚本会自动完成：依赖安装 → 模型下载 → 编译 → 推理执行。

### 方式二：手动运行

- 使用默认参数

    ```bash
    cd build/
    ./paddle_ocr
    ```

- 指定参数运行

    ```bash
    cd build/
    ./paddle_ocr \
        --det_model_path /opt/hobot/model/s100/basic/cn_PP-OCRv3_det_infer-deploy_640x640_nv12.hbm \
        --rec_model_path /opt/hobot/model/s100/basic/cn_PP-OCRv3_rec_infer-deploy_48x320_rgb.hbm \
        --test_image ../../../test_data/gt_2322.jpg \
        --label_file ../../../test_data/ppocr_keys_v1.txt \
        --font_path ../../../test_data/FangSong.ttf \
        --img_save_path result.jpg
    ```

### 输出结果

运行成功后，结果将保存至 `build/` 目录：

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
- 检测模型输入为 NV12 格式（Y + UV 双输入），大小 640×640。
- 识别模型输入为 float32 RGB NCHW 格式，大小 48×320。
- 依赖 `libpolyclipping-dev`（ClipperLib 多边形扩张）和 `libfreetype6-dev`（中文字体渲染）。
