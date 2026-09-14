[English](./README.md) | 简体中文

# MobileSAM Python 运行示例

本目录提供 RDK X5 上 MobileSAM 完整 mask 推理的 Python demo。TinyViT image encoder 和 box-prompt mask decoder 都已量化为 `.bin`，并通过 `hbm_runtime` 运行。

## 流程

1. `pre_process`：将输入图 resize 到 `512x512`，执行 MobileSAM mean/std 归一化，生成 NCHW float32 featuremap `normalized_images`。
2. `forward`：先运行 `mobile_sam_image_encoder_norm_512x512_allint16.bin`，再将 `image_embeddings` 和 box prompt 输入 `mobile_sam_decoder_512_box_default.bin`。
3. `post_process`：选择预测 IoU 最高的 mask，将低分辨率 logits 上采样到 `512x512`，并以 `0` 为阈值二值化。
4. 保存 mask 叠加图和二值 mask 图。

## 依赖

- 带 `hbm_runtime` 的 RDK X5 板端环境。
- Python 包：`numpy`、`opencv-python`。

## 运行

```bash
cd samples/vision/mobile_sam/runtime/python
bash run.sh
```

`run.sh` 会检查所需 `.bin` 文件，模型缺失时调用 `../../model/download_model.sh` 下载。也支持直接运行 `python3 main.py`：

```bash
python3 main.py
```

可选参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--encoder-model-path` | `../../model/mobile_sam_image_encoder_norm_512x512_allint16.bin` | 量化 encoder 模型路径。 |
| `--decoder-model-path` | `../../model/mobile_sam_decoder_512_box_default.bin` | 量化 decoder 模型路径。 |
| `--test-img` | `../../test_data/dogs.jpg` | 输入图片路径。 |
| `--box` | `185,120,380,445` | resize 到 `512x512` 后的 box prompt 坐标。 |
| `--img-save-path` | `../../test_data/mobile_sam_full_mask_result.jpg` | mask 叠加结果图输出路径。 |
| `--mask-save-path` | `../../test_data/mobile_sam_binary_mask.png` | 二值 mask 输出路径。 |
| `--priority` | `0` | 可选 runtime 调度优先级。 |

RDK X5 只有一个 BPU 核，本 demo 不提供 BPU core 选择参数。

示例：

```bash
python3 main.py \
  --encoder-model-path ../../model/mobile_sam_image_encoder_norm_512x512_allint16.bin \
  --decoder-model-path ../../model/mobile_sam_decoder_512_box_default.bin \
  --test-img ../../test_data/dogs.jpg \
  --box 185,120,380,445 \
  --img-save-path ../../test_data/mobile_sam_full_mask_result.jpg \
  --mask-save-path ../../test_data/mobile_sam_binary_mask.png
```

## 输出

- 叠加结果图：`../../test_data/mobile_sam_full_mask_result.jpg`
- 二值 mask：`../../test_data/mobile_sam_binary_mask.png`
