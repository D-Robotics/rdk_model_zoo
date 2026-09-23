[English](./README.md) | 简体中文

# 模型制品 — CLIP X5 encoder 对

<a id="artifacts"></a>
## 制品清单

sample 需要两个独立制品：BPU 图像 encoder 和 CPU ONNX 文本 encoder。它们使用不同 asset ID 和路径，外部路径不能静默替换其他发布文件。

| 制品 | 格式 | Target | 阶段 | 来源 |
| --- | --- | --- | --- | --- |
| `img_encoder.bin` | BIN | x5 | image encoder | download |
| `text_encoder.onnx` | ONNX | x5 | text encoder | download |

manifest 精确 URL：

| Asset ID | 发布 URL |
| --- | --- |
| `x5:clip:img_encoder.bin` | <https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/clip/img_encoder.bin> |
| `x5:clip:text_encoder.onnx` | <https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/clip/text_encoder.onnx> |

<a id="preparation"></a>
## 准备步骤

从仓库根目录运行。target 是命名参数且只接受 `x5`；没有 variant 参数。`runtime/python/run.sh` 不会隐式执行下载。

```bash
# cwd：仓库根目录；来源：上表精确 URL
python3 samples/vision/clip/model/download.py --target x5
# 预期：samples/vision/clip/model/img_encoder.bin 和 text_encoder.onnx

# cwd：仓库根目录；等价 shell wrapper 和显式落盘目录
bash samples/vision/clip/model/download.sh --target x5 --output-dir /tmp/clip-model
# 预期：/tmp/clip-model/img_encoder.bin 和 /tmp/clip-model/text_encoder.onnx
```

manifest 将两个 SHA-256 均记录为未知。下载器打印观测 digest，并说明它们不能独立验证发布方来源。I/O 或选择错误退出 2。本轮未下载任一文件。

<a id="accompanying-files"></a>
## 伴随文件

| 文件 | 作用 | 必需 |
| --- | --- | --- |
| `download.py` | 下载两个精确 manifest 制品。 | 脚本准备时是；两个文件都存在时否。 |
| `download.sh` | `download.py` 的命名参数 wrapper。 | 否。 |
| `../runtime/python/bpe_simple_vocab_16e6.txt.gz` | 文本 tokenizer 使用的源 BPE 词表。 | 文本编码时是。 |

<a id="local-paths"></a>
## 本地路径

- 默认图像路径：`samples/vision/clip/model/img_encoder.bin`。
- 默认文本路径：`samples/vision/clip/model/text_encoder.onnx`。
- 传入 `--image-model-path` 时必须配对 `--image-asset-id x5:clip:img_encoder.bin`。
- 传入 `--text-model-path` 时必须配对 `--text-asset-id x5:clip:text_encoder.onnx`。

<a id="formats-checksums"></a>
## 格式与校验值

| 制品 | 格式 | SHA-256 | 数值来源 |
| --- | --- | --- | --- |
| `img_encoder.bin` | BIN | `sha256: null (unknown)` | `docs/release/x5/models.yaml` |
| `text_encoder.onnx` | ONNX | `sha256: null (unknown)` | `docs/release/x5/models.yaml` |

## 许可

准备代码遵循仓库 [LICENSE](../../../../LICENSE) 的 Apache-2.0。源模型制品沿用发布记录中的来源和许可证 metadata；不在此新增断言。
