# PP-LiteSeg-STDC1 语义分割 Python 示例

本示例展示如何在 BPU 上使用量化后的 PP-LiteSeg-STDC1 模型执行实时语义分割任务。

## 目录结构

```text
.
├── main.py          # 推理入口脚本
├── pp_liteseg.py    # PP-LiteSeg 模型封装（PPLiteSegConfig + PPLiteSeg）
├── run.sh           # 一键运行脚本
├── README.md        # 使用说明（英文）
└── README_cn.md     # 使用说明（中文）
```

## 环境要求

- RDK X5 板端，固件 >= 3.5.0
- `hbm_runtime` Python 包（板端预装）
- `opencv-python`（`pip3 install opencv-python-headless`）

## 参数说明

| 参数             | 说明                             | 默认值                                               |
|------------------|----------------------------------|------------------------------------------------------|
| `--model-path`   | BPU `.bin` 模型路径              | `../../model/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin` |
| `--test-img`     | 测试图片路径                     | `../../test_data/street.jpg`                         |
| `--output`       | 输出结果图路径                   | `../../test_data/result.jpg`                         |
| `--alpha`        | 叠加混合透明度（0~1）            | `0.55`                                               |
| `--input-width`  | 模型输入宽度                     | `1024`                                               |
| `--input-height` | 模型输入高度                     | `512`                                                |

## 快速运行

- **一键运行脚本**
    ```bash
    chmod +x run.sh
    ./run.sh
    ```
    脚本会在模型不存在时自动下载。

- **手动运行**
    - 使用默认参数
        ```bash
        python3 main.py
        ```
    - 指定图片与输出路径
        ```bash
        python3 main.py \
            --model-path ../../model/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin \
            --test-img ../../test_data/street.jpg \
            --output ../../test_data/result.jpg
        ```

## 输出说明

结果图为三栏可视化：

```text
| 原图 | 叠加分割（alpha 混合）| 纯色分割图 |
```

叠加图右上角嵌有类别图例。

## 接口说明

- **PPLiteSegConfig**：封装模型路径及推理参数。
- **PPLiteSeg**：包含完整的推理流水线（`pre_process`、`forward`、`post_process`、`predict`、`visualize`）。

阅读 [源码文档说明](../../../../../docs/source_reference/README.md)，根据说明查看源码参考文档。
