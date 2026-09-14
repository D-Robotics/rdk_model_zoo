# UnetMobileNet 语义分割示例（C++）

本示例展示如何在 BPU 上使用量化后的 UnetMobileNet 模型执行图像语义分割。支持前处理（直接 Resize + NV12）、BPU 推理、后处理（逐像素 argmax + 颜色叠加）以及结果保存。

## 环境依赖

```bash
sudo apt install -y libgflags-dev
```

> 也可直接运行 `run.sh`，脚本会自动检测并安装所需依赖。

## 目录结构

```text
.
├── inc/
│   └── unetmobilenet.hpp   # UnetMobileNetConfig 结构体、模型类及推理接口声明
├── src/
│   ├── unetmobilenet.cpp   # 推理实现（init/pre_process/infer/post_process）
│   └── main.cpp            # 推理程序入口（参数解析与流程控制）
├── CMakeLists.txt           # CMake 构建配置
├── run.sh                   # 示例一键运行脚本
└── README.md                # 使用说明
```

## 编译工程

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

> 或直接运行 `run.sh`，脚本会自动完成编译与运行。

## 参数说明

| 参数            | 说明                                                         | 默认值                                                                        |
|-----------------|--------------------------------------------------------------|-------------------------------------------------------------------------------|
| `--model_path`  | 模型文件路径（.hbm 格式）                                    | `/opt/hobot/model/<soc>/basic/unet_mobilenet_1024x2048_nv12.hbm`              |
| `--test_img`    | 测试图片路径                                                 | `../../../test_data/segmentation.png`                                         |
| `--alpha_f`     | 可视化融合系数（`0.0`=仅显示掩码，`1.0`=仅原图）             | `0.75`                                                                        |

> **注意**：C++ 参数名采用 snake_case（如 `--model_path`），与 Python 版本的 kebab-case（`--model-path`）有所不同。

## 快速运行

- 使用 `run.sh` 一键运行（推荐）
    ```bash
    ./run.sh
    ```
- 手动编译后运行
    ```bash
    mkdir -p build && cd build && cmake .. && make -j$(nproc)
    ./unetmobilenet \
        --model_path /opt/hobot/model/s100/basic/unet_mobilenet_1024x2048_nv12.hbm \
        --test_img ../../../test_data/segmentation.png \
        --alpha_f 0.75
    ```
- 查看结果

    运行成功后，分割掩码会与原图混合并保存到当前目录的 `result.jpg`：
    ```
    [Saved] Result saved to: result.jpg
    ```

## 接口说明

阅读[源码文档说明](../../../../../docs/source_reference/README.md)，根据说明查看源码参考文档；
