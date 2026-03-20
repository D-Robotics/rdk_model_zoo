# ResNet18 图像分类样例

本示例演示如何使用`HB_HBMRuntime`的python接口部署`ResNet18`模型进行图像分类推理。适用于搭载BPU芯片的 RDK 系列设备。

## 环境依赖
本样例无特殊环境需求，只需确保安装了pydev中的环境依赖即可。
```bash
pip install -r ../../requirements.txt
```

## 目录结构
```text
.
├── README.md        # 示例说明文档，包含环境要求和使用方法
├── main.py          # 示例主入口，执行 ResNet18 图像分类推理
├── resnet18.py      # ResNet18 模型封装，实现预处理、推理和后处理
└── run.sh           # 一键运行脚本，负责环境准备、模型下载和示例执行
```

## 参数说明
| 参数           | 说明                                                     | 默认值                                      |
|----------------|----------------------------------------------------------|---------------------------------------------|
| `--model-path` | 模型文件路径（.hbm 格式）                                 | `/opt/hobot/model/<soc>/basic/resnet18_224x224_nv12.hbm`|
| `--test-img`   | 测试图片路径                                            | `../../test_data/zebra_cls.jpg`                |
| `--label-file` | 类别标签映射文件路径（dict 格式）                        | `../../test_data/imagenet1000_labels.txt` |
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
        --model-path /opt/hobot/model/s100/basic/resnet18_224x224_nv12.hbm \
        --test-img ../../test_data/zebra_cls.jpg \
        --label-file ../../test_data/imagenet1000_labels.txt
        ```

- 查看结果
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
