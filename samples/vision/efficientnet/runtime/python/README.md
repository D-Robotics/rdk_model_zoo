# EfficientNet-Lite0 图像分类示例

本示例展示如何在 BPU 上使用量化后的 EfficientNet-Lite0 模型执行图像分类任务。支持前处理、模型推理以及 Top-K 结果提取。

## 环境依赖
本样例无特殊环境需求，只需确保安装了 py_utils 中提到的环境依赖即可。
```bash
pip install numpy opencv-python scipy
```

## 目录结构
```text
.
├── efficientnet.py   # 模型封装类（包含 Config 和 Model）
├── main.py           # 主推理脚本
├── run.sh            # 一键运行脚本
└── README.md         # 使用说明
```

## 参数说明

| 参数           | 说明                                                     | 默认值                                      |
|----------------|----------------------------------------------------------|---------------------------------------------|
| `--model-path` | 模型文件路径（.hbm 格式）                                  | `/opt/hobot/model/s100/basic/efficientnet_lite0_224x224_nv12.hbm` |
| `--test-img`   | 测试图片路径                                              | `../../../../../datasets/imagenet/asset/scottish_deerhound.JPEG`  |
| `--label-file` | 类别标签路径（每行一个类别或字典格式）                       | `../../../../../datasets/imagenet/imagenet_classes.names`        |
| `--priority`  | 模型调度优先级（0~255）                                     | `0`                                         |
| `--bpu-cores` | 使用的 BPU 核心编号列表（如 `--bpu-cores 0 1`）              | `[0]`                                      |
| `--topk`      | 输出置信度最高的 K 个类别                                    | `5`                                         |
| `--resize-type` | 预处理缩放模式：0-直接缩放，1-保持比例(Letterbox)           | `1`                                         |


## 快速运行
- 运行模型
    - 使用默认参数
        ```bash
        bash run.sh
        ```
    - 指定参数运行
        ```bash
        python main.py \
            --model-path ../../model/efficientnet_lite0_224x224_nv12.hbm \
            --test-img ../../../../../datasets/imagenet/asset/scottish_deerhound.JPEG \
            --label-file ../../../../../datasets/imagenet/imagenet_classes.names \
            --topk 5 \
            --priority 0 \
            --bpu-cores 0
        ```
- 查看结果

    运行成功后，终端将输出 Top-K 的分类预测结果：
    ```bash
    Top-5 Results:
      Scottish deerhound, deerhound: 0.8234
      greyhound: 0.0123
      Saluki, gazelle hound: 0.0056
      whippet: 0.0034
      Irish wolfhound: 0.0021
    ```


## 接口说明
阅读[源码文档说明](../../../../../docs/source_reference/README.md)，根据说明查看源码参考文档；
本示例代码提供了详细的注释。为了获取最准确、最新的接口定义，请直接查阅源码中的文档字符串：
- **EfficientNetConfig** 与 **EfficientNet**: 详见 `efficientnet.py`

## 注意事项
 无