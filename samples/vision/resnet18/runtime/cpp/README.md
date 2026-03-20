# ResNet18 图像分类示例

本示例展示如何在 BPU 上使用量化后的 ResNet18 模型执行图像分类，输出 Top-K 类别及对应置信度。

## 环境依赖
请先确保系统已安装以下依赖：
```bash
sudo apt update
sudo apt install libgflags-dev
```

## 目录结构
```bash
.
|-- inc                # 头文件目录
|   `-- resnet18.hpp   # ResNet18 模型封装类声明
|-- src                # 源码目录
|   |-- main.cpp       # 推理程序入口（参数解析与流程控制）
|   `-- resnet18.cpp   # ResNet18 推理与后处理实现
|-- CMakeLists.txt     # CMake 构建配置文件
|-- README.md          # C++ 推理示例使用说明
`-- run.sh             # 示例运行脚本
```

## 编译工程
- 配置与编译
    ```bash
    mkdir build && cd build
    cmake ..
    make -j$(nproc)
    ```

## 参数说明
| 参数             | 说明                        | 默认值                                                          |
| --------------- | --------------------------- | --------------------------------------------------------------- |
| `--model_path`  | 模型文件路径（.hbm 格式）    | `/opt/hobot/model/<soc>/basic/resnet18_224x224_nv12.hbm`       |
| `--test_img`    | 测试图片路径                 | `../../../test_data/zebra_cls.jpg`                              |
| `--label_file`  | 类别标签文件路径             | `../../../test_data/imagenet1000_labels.txt`                    |
| `--top_k`       | 输出 Top-K 分类结果数量      | `5`                                                             |

> **注意**：`--model_path` 默认值中的 `<soc>` 会在编译时根据当前设备 SoC 自动注入（如 `s100`、`s600`）。

## 快速运行
- 运行模型
    - 使用脚本自动运行（自动安装依赖、下载模型、编译并运行）
        ```bash
        ./run.sh
        ```
    - 使用默认参数（在 `build/` 目录下执行）
        ```bash
        ./resnet18
        ```
    - 指定参数运行
        ```bash
        ./resnet18 \
            --model_path /opt/hobot/model/s100/basic/resnet18_224x224_nv12.hbm \
            --test_img ../../../test_data/zebra_cls.jpg \
            --label_file ../../../test_data/imagenet1000_labels.txt \
            --top_k 5
        ```
- 查看结果

    运行成功后，将在终端打印 Top-K 分类结果：
    ```bash
    Top-5 Classification Results:
    [0] zebra: 0.9981
    [1] cheetah, chetah, Acinonyx jubatus: 0.0004
    [2] impala, Aepyceros melampus: 0.0004
    [3] gazelle: 0.0003
    [4] prairie chicken, prairie grouse, prairie fowl: 0.0002
    ```

## 接口说明
阅读[源码文档说明](../../../../../docs/source_reference/README.md)，根据说明查看源码参考文档；
