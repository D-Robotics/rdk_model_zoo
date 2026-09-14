[English](./README.md) | 简体中文

# EfficientSAM-Tiny Python 运行示例

本目录提供 RDK X5 上 EfficientSAM-Tiny 完整 mask 推理的 Python demo。Image encoder 和固定 prompt decoder 都已量化为 `.bin`，并通过 `hbm_runtime` 运行。

## 流程

1. `pre_process`：将输入图 resize 到 `512x512`，生成 NCHW float32 featuremap `batched_images`。
2. `forward`：先运行 `efficient_sam_vitt_encoder_512x512_default_none.bin`，再将 `image_embeddings` 输入 `efficient_sam_vitt_decoder_fixedprompt_512_default.bin`。
3. `post_process`：选择预测 IoU 最高的 mask，将低分辨率 logits 上采样到 `512x512`，并以 `0` 为阈值二值化。
4. 保存 mask 叠加图和二值 mask 图。

## 依赖

- 带 `hbm_runtime` 的 RDK X5 板端环境。
- Python 包：`numpy`、`opencv-python`。

## 运行

```bash
cd samples/vision/efficient_sam/runtime/python
bash run.sh
```

`run.sh` 会检查所需 `.bin` 文件，模型缺失时调用 `../../model/download_model.sh` 下载。也支持直接运行 `python3 main.py`：

```bash
python3 main.py
```

可选参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--encoder-model-path` | `../../model/efficient_sam_vitt_encoder_512x512_default_none.bin` | 量化 encoder 模型路径。 |
| `--decoder-model-path` | `../../model/efficient_sam_vitt_decoder_fixedprompt_512_default.bin` | 量化 decoder 模型路径。 |
| `--test-img` | `../../test_data/dogs.jpg` | 输入图片路径。 |
| `--img-save-path` | `../../test_data/efficient_sam_full_mask_result.jpg` | mask 叠加结果图输出路径。 |
| `--mask-save-path` | `../../test_data/efficient_sam_binary_mask.png` | 二值 mask 输出路径。 |
| `--priority` | `0` | 可选 runtime 调度优先级。 |

RDK X5 只有一个 BPU 核，本 demo 不提供 BPU core 选择参数。

示例：

```bash
python3 main.py \
  --encoder-model-path ../../model/efficient_sam_vitt_encoder_512x512_default_none.bin \
  --decoder-model-path ../../model/efficient_sam_vitt_decoder_fixedprompt_512_default.bin \
  --test-img ../../test_data/dogs.jpg \
  --img-save-path ../../test_data/efficient_sam_full_mask_result.jpg \
  --mask-save-path ../../test_data/efficient_sam_binary_mask.png
```

## 输出

- 叠加结果图：`../../test_data/efficient_sam_full_mask_result.jpg`
- 二值 mask：`../../test_data/efficient_sam_binary_mask.png`
