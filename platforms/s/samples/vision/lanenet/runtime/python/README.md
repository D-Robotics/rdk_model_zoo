# LaneNet 车道线检测示例（Python）

本示例展示如何在 BPU 上使用量化后的 LaneNet 模型执行车道线检测。支持前处理、推理，以及实例分割与二值分割结果输出。

> ⚠️ **平台说明**：本模型仅支持 **RDK S100** 平台。若使用 RDK S600，请参阅 [平台兼容性说明](#注意事项)。

## 环境依赖

本样例无特殊环境需求，只需确保已安装以下依赖：

```bash
pip install numpy==1.26.4 opencv-python==4.11.0.86
```

## 目录结构

```text
.
├── lanenet.py          # LaneNet 推理封装与后处理实现
├── main.py             # 推理程序入口（参数解析与流程控制）
├── run.sh              # 示例运行脚本（自动下载模型并运行）
└── README.md           # 使用说明
```

## 参数说明

| 参数                    | 说明                                   | 默认值                                                    |
|------------------------|----------------------------------------|----------------------------------------------------------|
| `--model-path`         | 模型文件路径（.hbm 格式）               | `/opt/hobot/model/s100/basic/lanenet256x512.hbm`         |
| `--test-img`           | 测试图片路径                            | `../../test_data/lane.jpg`                               |
| `--instance-save-path` | 实例分割结果图像保存路径                 | `instance_pred.png`                                      |
| `--binary-save-path`   | 二值分割结果图像保存路径                 | `binary_pred.png`                                        |
| `--priority`           | 模型调度优先级（0~255）                  | `0`                                                      |
| `--bpu-cores`          | 使用的 BPU 核心编号列表                  | `[0]`                                                    |

> **注意**：`--model-path` 默认路径固定为 S100 模型路径，不随平台自动切换。

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
        --model-path /opt/hobot/model/s100/basic/lanenet256x512.hbm \
        --test-img ../../test_data/lane.jpg \
        --instance-save-path instance_pred.png \
        --binary-save-path binary_pred.png \
        --priority 0 \
        --bpu-cores 0
    ```

### 输出结果

运行成功后，结果将保存至当前目录：

```text
[Saved] Instance segmentation result: instance_pred.png
[Saved] Binary segmentation result:   binary_pred.png
```

- `instance_pred.png`：彩色实例分割掩码（RGB，uint8）
- `binary_pred.png`：二值车道线分割掩码（灰度，uint8）

## 接口说明

阅读[源码文档说明](../../../../../docs/source_reference/README.md)，根据说明查看源码参考文档。

## 注意事项

- **平台兼容性**：本模型（`lanenet256x512.hbm`）仅支持 RDK S100 平台，模型使用 S100 BPU 编译，**不支持 RDK S600**。若需在 S600 上运行车道线检测，需重新使用 S600 工具链编译对应模型。
- 若模型文件不存在，`run.sh` 会自动从 D-Robotics 下载中心下载 S100 模型。
- 测试图片需为路面/车道场景的 BGR 格式图像，推荐分辨率不低于 256×512。
