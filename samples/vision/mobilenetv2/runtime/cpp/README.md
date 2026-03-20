# MobileNetV2 图像分类示例（C++）

本示例展示如何在 BPU 上使用量化后的 MobileNetV2 模型执行图像分类，输出 Top-K 类别及对应置信度。

## 环境依赖
请先确保系统已安装以下依赖：
```bash
sudo apt update
sudo apt install libgflags-dev
```

## 目录结构
```bash
.
|-- inc                    # 头文件目录
|   `-- mobilenetv2.hpp    # MobileNetV2 模型封装类声明
|-- src                    # 源码目录
|   |-- main.cpp           # 推理程序入口（参数解析与流程控制）
|   `-- mobilenetv2.cpp    # MobileNetV2 推理与后处理实现
|-- CMakeLists.txt         # CMake 构建配置文件
|-- README.md              # C++ 推理示例使用说明
`-- run.sh                 # 示例运行脚本
```

## 编译工程
- 配置与编译
    ```bash
    mkdir build && cd build
    cmake ..
    make -j$(nproc)
    ```

## 参数说明
| 参数             | 说明                        | 默认值                                                               |
| --------------- | --------------------------- | -------------------------------------------------------------------- |
| `--model_path`  | 模型文件路径（.hbm 格式）    | `/opt/hobot/model/<soc>/basic/mobilenetv2_224x224_nv12.hbm`         |
| `--test_img`    | 测试图片路径                 | `../../../test_data/zebra_cls.jpg`                                   |
| `--label_file`  | 类别标签文件路径             | `../../../test_data/imagenet1000_labels.txt`                         |
| `--top_k`       | 输出 Top-K 分类结果数量      | `5`                                                                  |

> **注意**：`--model_path` 默认值中的 `<soc>` 会在编译时根据当前设备 SoC 自动注入（如 `s100`、`s600`）。

## 快速运行
- 运行模型
    - 使用脚本自动运行（自动安装依赖、下载模型、编译并运行）
        ```bash
        ./run.sh
        ```
    - 使用默认参数（在 `build/` 目录下执行）
        ```bash
        ./mobilenetv2
        ```
    - 指定参数运行
        ```bash
        ./mobilenetv2 \
            --model_path /opt/hobot/model/s100/basic/mobilenetv2_224x224_nv12.hbm \
            --test_img ../../../test_data/zebra_cls.jpg \
            --label_file ../../../test_data/imagenet1000_labels.txt \
            --top_k 5
        ```
- 查看结果

    运行成功后，将在终端打印 Top-K 分类结果：
    ```bash
    TOP-1: label=zebra, prob=0.992246
    TOP-2: label=tiger, Panthera tigris, prob=0.00404656
    TOP-3: label=hartebeest, prob=0.00133707
    TOP-4: label=tiger cat, prob=0.000722661
    TOP-5: label=impala, Aepyceros melampus, prob=0.000539704
    ```

## 接口说明
阅读[源码文档说明](../../../../../docs/source_reference/README.md)，根据说明查看源码参考文档；
