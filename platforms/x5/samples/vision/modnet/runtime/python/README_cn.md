# MODNet 人像抠图 Python 示例

本示例展示如何在 BPU 上使用 MODNet 模型执行人像抠图任务。

## 目录结构

```text
.
├── main.py         # 推理入口脚本
├── modnet.py       # MODNet 模型封装
├── run.sh          # 一键运行脚本
├── README.md       # 使用说明 (英文)
└── README_cn.md    # 使用说明 (中文)
```

## 参数说明

| 参数                | 说明                                   | 默认值                             |
|---------------------|----------------------------------------|------------------------------------|
| `--model-path`      | 模型文件路径（.bin 格式）               | `../../model/modnet_512x512_rgb.bin` |
| `--test-img`        | 测试图片路径                            | `../../test_data/person.jpg`       |
| `--bg-img`          | 背景图像路径（用于合成）                 | `../../test_data/bg.jpg`           |
| `--matte-save-path` | Alpha matte 保存路径                   | `../../test_data/matte.png`        |
| `--img-save-path`   | 合成结果图像保存路径                     | `../../test_data/result.png`       |
| `--priority`        | 模型调度优先级（0~255）                 | `0`                                |
| `--bpu-cores`       | BPU 核心索引                           | `[0]`                              |
| `--ref-size`        | 目标输入分辨率（长边）                   | `512`                              |

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
            --test-img path/to/img.jpg \
            --bg-img path/to/bg.jpg
        ```

## 接口说明

- **MODNetConfig**: 封装模型路径及推理参数。
- **MODNet**: 包含完整的推理流水线（`pre_process`, `forward`, `post_process`, `predict`）。

阅读 [源码文档说明](../../../../../docs/source_reference/README.md)，根据说明查看源码参考文档。
