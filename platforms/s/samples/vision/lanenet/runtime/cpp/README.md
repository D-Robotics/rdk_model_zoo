# LaneNet 车道线检测示例（C++）

本示例展示如何在 BPU 上使用量化后的 LaneNet 模型执行车道线检测。支持前处理、推理，以及实例分割与二值分割结果输出。

> ⚠️ **平台说明**：本模型仅支持 **RDK S100** 平台。若使用 RDK S600，请参阅 [注意事项](#注意事项)。

## 环境依赖

```bash
sudo apt install libgflags-dev
```

## 目录结构

```text
.
├── inc/
│   └── lanenet.hpp     # LaneNet 模型封装接口与函数声明
├── src/
│   ├── lanenet.cpp     # LaneNet 推理与前后处理实现
│   └── main.cpp        # 推理程序入口（参数解析与流程控制）
├── CMakeLists.txt      # CMake 构建配置
├── run.sh              # 示例运行脚本（自动安装依赖、下载模型、编译、运行）
└── README.md           # 使用说明
```

## 编译工程

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

编译产物为 `build/lanenet`。

## 参数说明

| 参数                    | 说明                                   | 默认值                                                    |
|------------------------|----------------------------------------|----------------------------------------------------------|
| `--model_path`         | 模型文件路径（.hbm 格式）               | `/opt/hobot/model/s100/basic/lanenet256x512.hbm`         |
| `--test_img`           | 测试图片路径                            | `../../../test_data/lane.jpg`                            |
| `--instance_save_path` | 实例分割结果图像保存路径                 | `instance_pred.png`                                      |
| `--binary_save_path`   | 二值分割结果图像保存路径                 | `binary_pred.png`                                        |

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
    ./lanenet
    ```

- 指定参数运行

    ```bash
    cd build/
    ./lanenet \
        --model_path /opt/hobot/model/s100/basic/lanenet256x512.hbm \
        --test_img ../../../test_data/lane.jpg \
        --instance_save_path instance_pred.png \
        --binary_save_path binary_pred.png
    ```

### 输出结果

运行成功后，结果将保存至 `build/` 目录：

```text
[Saved] Instance segmentation result: instance_pred.png
[Saved] Binary segmentation result:   binary_pred.png
```

- `instance_pred.png`：彩色实例分割掩码（uint8 3通道）
- `binary_pred.png`：二值车道线分割掩码（uint8 单通道，白色为车道线）

## 接口说明

阅读[源码文档说明](../../../../../docs/source_reference/README.md)，根据说明查看源码参考文档。

## 注意事项

- **平台兼容性**：本模型（`lanenet256x512.hbm`）仅支持 RDK S100 平台，**不支持 RDK S600**。若需在 S600 上运行，需使用 S600 工具链重新编译模型。
- 若模型文件不存在，`run.sh` 会自动从 D-Robotics 下载中心下载 S100 模型。
- 测试图片需为路面/车道场景的 BGR 格式图像，推荐分辨率不低于 256×512。
