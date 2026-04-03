# ConvNeXt 图像分类 Python 示例

本示例展示如何在 BPU 上使用量化后的 ConvNeXt 模型执行图像分类任务。

## 环境依赖

本样例无特殊环境依赖，确保安装了 pydev 中的环境依赖即可。

```bash
pip install numpy opencv-python hbm-runtime scipy
```

## 目录结构

```text
.
├── main.py                # 推理入口脚本
├── convnext.py            # ConvNeXt 模型封装
├── run.sh                 # 一键运行脚本
├── README.md              # 使用说明 (英文)
└── README_cn.md           # 使用说明 (中文)
```

## 参数说明

| 参数           | 说明                                                     | 默认值                                      |
|----------------|----------------------------------------------------------|---------------------------------------------|
| `--model-path` | 模型文件路径（.bin 格式）                                  | `../../model/ConvNeXt_atto_224x224_nv12.bin` |
| `--test-img`   | 测试图片路径                                              | `../../test_data/cheetah.JPEG`              |
| `--label-file` | ImageNet 标签文件路径                                      | `../../../../datasets/imagenet/imagenet_classes.names` |
| `--topk`       | 显示 Top-K 结果                                           | `5`                                         |
| `--resize-type`| 缩放策略：0 为直接缩放，1 为等比例缩放并填充                 | `1`                                         |

## 快速运行

- **一键运行脚本**
    ```bash
    chmod +x run.sh
    ./run.sh
    ```

- **手动运行**
    - 使用默认参数
        ```bash
        python3 main.py
        ```
    - 指定参数运行
        ```bash
        python3 main.py \
            --model-path ../../model/ConvNeXt_atto_224x224_nv12.bin \
            --test-img ../../test_data/cheetah.JPEG \
            --topk 5
        ```

## 接口说明

- **ConvNeXtConfig**: 封装模型路径及推理参数。
- **ConvNeXt**: 包含完整的推理流水线（`pre_process`, `forward`, `post_process`, `predict`）。

阅读 [源码文档说明](../../../../../docs/source_reference/README.md)，根据说明查看源码参考文档。
