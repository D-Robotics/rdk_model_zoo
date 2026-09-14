[English](./README.md) | 简体中文

# CLIP Python 示例

本示例说明如何使用 BPU 图像 encoder 和 ONNX 文本 encoder 完成图文相似度匹配。

## 目录结构

```text
.
|-- bpe_simple_vocab_16e6.txt.gz
|-- clip_retrieval.py
|-- main.py
|-- README.md
|-- README_cn.md
|-- run.sh
`-- simple_tokenizer.py
```

## 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--image-model-path` | BPU 图像 encoder `.bin` 模型路径。 | `../../model/img_encoder.bin` |
| `--text-model-path` | ONNX 文本 encoder 模型路径。 | `../../model/text_encoder.onnx` |
| `--test-img` | 测试输入图片路径。 | `../../test_data/dog.jpg` |
| `--texts` | 逗号分隔的候选文本描述。 | `a diagram,a dog` |
| `--img-save-path` | 可视化结果图保存路径。 | `../../test_data/inference.png` |
| `--priority` | 图像 encoder 优先级，范围 `0~255`。 | `0` |
| `--bpu-cores` | 图像 encoder 使用的 BPU 核心索引。 | `0` |

## 快速运行

本 sample 的 ONNX 文本 encoder 和 tokenizer 需要 `onnxruntime`、`ftfy` 和
`regex`。若板端系统镜像未预装这些依赖，请先执行：

```bash
python3 -m pip install --user onnxruntime ftfy regex
```

```bash
chmod +x run.sh
./run.sh
```

## 手动运行

直接运行 `main.py` 前需要先准备默认模型。可以先执行一次 `./run.sh`，也可以在本目录下执行
`../../model/download.sh` 下载 `../../model/img_encoder.bin` 和 `../../model/text_encoder.onnx`。

- 使用默认参数运行：

```bash
python3 main.py
```

- 使用显式参数运行：

```bash
python3 main.py \
    --image-model-path ../../model/img_encoder.bin \
    --text-model-path ../../model/text_encoder.onnx \
    --test-img ../../test_data/dog.jpg \
    --texts "a diagram,a dog" \
    --img-save-path ../../test_data/inference.png
```

## 接口说明

- **CLIPConfig**：封装图像 encoder、文本 encoder 和 tokenizer 参数。
- **CLIPMatcher**：实现图像预处理、BPU 图像编码、ONNX 文本编码和余弦相似度匹配。
