# MobileNetV2 图像分类样例（Python）

本示例演示如何使用`HB_HBMRuntime`的 Python 接口部署`MobileNetV2`模型进行图像分类推理，输出 Top-K 类别及对应置信度。适用于搭载BPU芯片的 RDK 系列设备。

## 环境依赖
本样例无特殊环境需求，只需确保安装了以下依赖即可。
```bash
pip install numpy==1.26.4 opencv-python==4.11.0.86 scipy==1.15.3
```

## 目录结构
```text
.
├── README.md           # 示例说明文档，包含环境要求和使用方法
├── main.py             # 示例主入口，执行 MobileNetV2 图像分类推理
├── mobilenetv2.py      # MobileNetV2 模型封装，实现预处理、推理和后处理
└── run.sh              # 一键运行脚本，负责环境准备、模型下载和示例执行
```

## 参数说明
| 参数           | 说明                                                     | 默认值                                      |
|----------------|----------------------------------------------------------|---------------------------------------------|
| `--model-path` | 模型文件路径（.hbm 格式）                                 | `/opt/hobot/model/<soc>/basic/mobilenetv2_224x224_nv12.hbm`|
| `--test-img`   | 测试图片路径                                            | `../../test_data/zebra_cls.jpg`                |
| `--label-file` | 类别标签映射文件路径                                     | `../../test_data/imagenet1000_labels.txt` |
| `--priority`   | 模型优先级（0~255，越大优先级越高）                      | `0`                                         |
| `--bpu-cores`  | 推理使用的 BPU 核心编号列表（如 `--bpu-cores 0 1`）     | `[0]`                                       |

> **注意**：`--model-path` 默认值中的 `<soc>` 会在运行时根据当前设备自动识别（如 `s100`、`s600`）。

## 快速运行
- 运行模型
    - 使用脚本自动运行
        ```bash
        ./run.sh
        ```
    - 使用默认参数
        ```bash
        python main.py
        ```
    - 指定参数运行
        ```bash
        python main.py \
        --model-path /opt/hobot/model/s100/basic/mobilenetv2_224x224_nv12.hbm \
        --test-img ../../test_data/zebra_cls.jpg \
        --label-file ../../test_data/imagenet1000_labels.txt
        ```

- 查看结果
    ```bash
    Top-5 Classification Results:
    [0] zebra: 0.9922
    [1] tiger, Panthera tigris: 0.0040
    [2] hartebeest: 0.0013
    [3] tiger cat: 0.0007
    [4] impala, Aepyceros melampus: 0.0005
    ```

## 接口说明
阅读[源码文档说明](../../../../../docs/source_reference/README.md)，根据说明查看源码参考文档；
