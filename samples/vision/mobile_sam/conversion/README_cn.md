English | [简体中文](./README_cn.md)

# MobileSAM 转换

本目录记录 MobileSAM 图像编码器和 box prompt 解码器的源转换能力。统一脚本通过 `--target x5`、`s100`、`s100p` 或 `s600` 选择目标；本次迁移没有运行转换。

<a id="source-model"></a>
## 源模型

源项目是 [ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM)，checkpoint 为 `weights/mobile_sam.pt`。固定源没有锁定上游 commit 或 checkpoint digest。编码器输入 `normalized_images` `(1,3,512,512)`、float32、RGB NCHW，输出 `image_embeddings` `(1,256,32,32)`、float32。解码器输入 `image_embeddings` `(1,256,32,32)` 和运行时 `boxes` `(1,4)` float32，输出 `low_res_masks` `(1,3,128,128)` 与 `iou_predictions`（最终 rank 必须以 decoder metadata 确认）。

预处理为 RGB 缩放到 512×512，再按通道应用 ImageNet 归一化：mean `[123.675,116.28,103.53]`，std `[58.395,57.12,57.375]`。源默认 box 是 512 像素坐标中的 `[185,120,380,445]`。box 是运行时输入，没有固定到 ONNX 图中。

<a id="toolchain-targets"></a>
## 工具链与目标

| 目标 | 配置目录 | march | 编译器 |
|---|---|---|---|
| `x5` | `configs/x5/` | `bayes-e` | `hb_mapper makertbin --model-type onnx` |
| `s100` | `configs/s100/` | `nash-e` | `hb_compile` |
| `s100p` | `configs/s100p/` | `nash-m` | `hb_compile` |
| `s600` | `configs/s600/` | `nash-p` | `hb_compile` |

X5 源配置使用 OE X5 环境和默认校准。S 配置使用 OE S 系列 3.7.0、`set_all_nodes_int16`、max 校准与 `max_percentile: 0.9999`。发布资产名称及 SHA 以 `docs/release/x5/models.yaml`、`docs/release/s/models.yaml` 为准；源 manifest 中 SHA 为未知（`null`）。

导出与浮点 embedding 生成依赖主机 PyTorch、ONNX、ONNX Runtime、NumPy、OpenCV。固定源没有锁定这些包的版本或上游仓库 revision，这仍是复现前提，不是经过验证的环境规格。在选定导出环境中检查导入，与板端推理环境分开：

```bash
# cwd: 本 conversion 目录；在选定主机导出环境中执行
python3 -c "import torch, onnx, onnxruntime, numpy, cv2; print(torch.__version__, onnx.__version__, onnxruntime.__version__)"
```

导出器实际调用 `ultralytics.models.sam.build.build_mobile_sam`，仅克隆 MobileSAM 仓库不会提供该包。源步骤是在导出环境执行 `python3 -m pip install ultralytics`。源没有固定版本，应记录实际版本，并在导出前核对 `build_mobile_sam`/`set_imgsz` API；本轮没有执行此安装命令。

<a id="export"></a>
## 导出 ONNX

X5 源提供上游 checkout helper。固定 S 源没有等价 helper，因此 S 目标需手工准备 checkout 和 checkpoint。

```bash
python3 scripts/download_assets.py --target x5 --workspace ./workspace
# S 目标：手工放置 ./workspace/MobileSAM 和
# ./workspace/MobileSAM/weights/mobile_sam.pt
```

X5 helper 在 `mobile_sam.pt` 不存在或小于 1,000,000 字节时下载它，并在此之前 clone 或复用 checkout。S 源没有等价 helper，因此源代码和 checkpoint 仍是手工前置条件。

从本目录导出两个图：

```bash
python3 scripts/export_encoder_onnx.py --target x5 \
  --repo ./workspace/MobileSAM \
  --weights ./workspace/MobileSAM/weights/mobile_sam.pt \
  --output ./mobile_sam_image_encoder_norm_512_op11.onnx
python3 scripts/export_decoder_onnx.py --target x5 \
  --repo ./workspace/MobileSAM \
  --checkpoint ./workspace/MobileSAM/weights/mobile_sam.pt \
  --output ./mobile_sam_decoder_512_box_op11.onnx \
  --box 185 120 380 445
```

S100/S100P/S600 只需替换 `--target`。导出默认尺寸为 512、opset 为 11；可显式指定 `--size`、`--opset` 和 decoder 的 `--box`。脚本不会自行下载 checkpoint。

<a id="calibration"></a>
## 校准

编码器校准读取代表性 RGB 图片，缩放至 512×512，转换为 NCHW，并使用上面的 ImageNet mean/std。X5 写 raw `.rgbchw`，S 写 `.npy`。producer 的数量是输出文件数量，不能证明输入彼此独立或具有代表性。

```bash
python3 scripts/prepare_calibration.py --target x5 \
  --src ./calibration_images --out . --num 30 --size 512
# ./calibration_data_norm_512/*.rgbchw
python3 scripts/dump_encoder_embedding.py --image ../test_data/dogs.jpg \
  --onnx ./mobile_sam_image_encoder_norm_512_op11.onnx --output ./encoder_embedding.bin

python3 scripts/prepare_decoder_calibration.py --target x5 \
  --embedding ./encoder_embedding.bin --out ./decoder_calibration \
  --num 30 --box 185 120 380 445
# ./decoder_calibration/calibration_embeddings/*.bin
# ./decoder_calibration/calibration_boxes/*.bin
```

S 目标需显式指定与 YAML 一致的校准根目录：

```bash
python3 scripts/prepare_calibration.py --target s100 \
  --src ./calibration_images --out ./calibration_data_norm_512 --num 30 --size 512
python3 scripts/dump_encoder_embedding.py --image ../test_data/dogs.jpg \
  --onnx ./mobile_sam_image_encoder_norm_512_op11.onnx --output ./encoder_embedding.bin
python3 scripts/prepare_decoder_calibration.py --target s100 \
  --embedding ./encoder_embedding.bin --out ./decoder_calibration \
  --num 30 --box 185 120 380 445
```

这会生成 `./calibration_data_norm_512/normalized_images/*.npy`、`./decoder_calibration/image_embeddings/*.npy` 和 `./decoder_calibration/boxes/*.npy`。dump helper 运行浮点 encoder ONNX 并使用源 ImageNet 变换，需要主机 `onnxruntime`；本次没有运行。此次迁移没有生成 embedding。

dump 示例在 ONNX 导出后使用已提交的 `../test_data/dogs.jpg`，输入文件可直接定位；实际量化应换成有代表性的校准图片。该 helper 源自 S 分支，按同一 tensor 协议可读取两类目标的浮点 encoder ONNX，本轮仅执行注入 ORT 的 fixture。`calibration_images/` 数据集由用户准备，仓库不提供。单份 embedding 缩放、框扰动只保留源演示配方，不等同于代表性校准数据集。提交配置与 runtime 仅覆盖 size 512；改变导出 `--size` 还需要对应的新配置与 runtime binding。

<a id="compile"></a>
## 编译

准备好目标 ONNX 和校准路径后运行：

```bash
python3 scripts/quantize.py --target x5
python3 scripts/quantize.py --target s100
python3 scripts/quantize.py --target s100p
python3 scripts/quantize.py --target s600
```

`--target <target> --config <path>` 可只编译一份 YAML。X5 调用 `hb_mapper makertbin --model-type onnx`；S 调用 `hb_compile --config`。X5 输出在 `bpu_model_output_norm_512_allint16/` 和 `bpu_model_output_decoder_default/`；S 输出在各目标编码器/解码器工作目录，文件名以 YAML 和 manifest 为准。本次没有运行编译。提交的 S 路径如下：

| 目标 | 角色 | ONNX | 校准目录 | 工作目录 | 输出前缀 |
|---|---|---|---|---|---|
| S100 | encoder | `mobile_sam_image_encoder_norm_512_op11.onnx` | `./calibration_data_norm_512/normalized_images` | `bpu_model_output_encoder_nashe` | `mobile_sam_image_encoder_norm_512x512_nashe` |
| S100 | decoder | `mobile_sam_decoder_512_op11.onnx` | `./decoder_calibration/image_embeddings;./decoder_calibration/boxes` | `bpu_model_output_decoder_nashe` | `mobile_sam_decoder_512_nashe` |
| S100P | encoder | `mobile_sam_image_encoder_norm_512_op11.onnx` | `./calibration_data_norm_512/normalized_images` | `bpu_model_output_encoder_nashm` | `mobile_sam_image_encoder_norm_512x512_nashm` |
| S100P | decoder | `mobile_sam_decoder_512_op11.onnx` | `./decoder_calibration/image_embeddings;./decoder_calibration/boxes` | `bpu_model_output_decoder_nashm` | `mobile_sam_decoder_512_nashm` |
| S600 | encoder | `mobile_sam_image_encoder_norm_512_op11.onnx` | `./calibration_data_norm_512/normalized_images` | `bpu_model_output_encoder_nashp` | `mobile_sam_image_encoder_norm_512x512_nashp` |
| S600 | decoder | `mobile_sam_decoder_512_op11.onnx` | `./decoder_calibration/image_embeddings;./decoder_calibration/boxes` | `bpu_model_output_decoder_nashp` | `mobile_sam_decoder_512_nashp` |

各配置与预期编译输出（相对本 conversion 工作目录）：

| Config | ONNX | Calibration | Compiler output |
| --- | --- | --- | --- |
| `configs/s100/mobile_sam_decoder_512_nashe_config.yaml` | `./mobile_sam_decoder_512_op11.onnx` | `./decoder_calibration/image_embeddings;./decoder_calibration/boxes` | `bpu_model_output_decoder_nashe/mobile_sam_decoder_512_nashe.hbm` |
| `configs/s100/mobile_sam_encoder_nashe_config.yaml` | `./mobile_sam_image_encoder_norm_512_op11.onnx` | `./calibration_data_norm_512/normalized_images` | `bpu_model_output_encoder_nashe/mobile_sam_image_encoder_norm_512x512_nashe.hbm` |
| `configs/s100p/mobile_sam_decoder_512_nashm_config.yaml` | `./mobile_sam_decoder_512_op11.onnx` | `./decoder_calibration/image_embeddings;./decoder_calibration/boxes` | `bpu_model_output_decoder_nashm/mobile_sam_decoder_512_nashm.hbm` |
| `configs/s100p/mobile_sam_encoder_nashm_config.yaml` | `./mobile_sam_image_encoder_norm_512_op11.onnx` | `./calibration_data_norm_512/normalized_images` | `bpu_model_output_encoder_nashm/mobile_sam_image_encoder_norm_512x512_nashm.hbm` |
| `configs/s600/mobile_sam_decoder_512_nashp_config.yaml` | `./mobile_sam_decoder_512_op11.onnx` | `./decoder_calibration/image_embeddings;./decoder_calibration/boxes` | `bpu_model_output_decoder_nashp/mobile_sam_decoder_512_nashp.hbm` |
| `configs/s600/mobile_sam_encoder_nashp_config.yaml` | `./mobile_sam_image_encoder_norm_512_op11.onnx` | `./calibration_data_norm_512/normalized_images` | `bpu_model_output_encoder_nashp/mobile_sam_image_encoder_norm_512x512_nashp.hbm` |
| `configs/x5/mobile_sam_decoder_512_box_default_config.yaml` | `./mobile_sam_decoder_512_box_op11.onnx` | `./decoder_calibration/calibration_embeddings; ./decoder_calibration/calibration_boxes` | `bpu_model_output_decoder_default/mobile_sam_decoder_512_box_default.bin` |
| `configs/x5/mobile_sam_image_encoder_norm_512x512_config.yaml` | `./mobile_sam_image_encoder_norm_512_op11.onnx` | `./calibration_data_norm_512` | `bpu_model_output_norm_512_allint16/mobile_sam_image_encoder_norm_512x512_allint16.bin` |

单配置命令必须同时显式指定匹配的 `--target`；脚本不从 YAML 推断编译器，省略 target 会默认 X5。例如：`python3 scripts/quantize.py --target s100 --config configs/s100/mobile_sam_decoder_512_nashe_config.yaml`。

以下输入字段逐字取自各编译配置，不能视为实测 SDK metadata；转换后仍须读取模型 metadata 校验。X5 的 type 字段依次为 runtime / train，S 为 input_type。

| Config | input_name | input_shape | type |
| --- | --- | --- | --- |
| `configs/s100/mobile_sam_decoder_512_nashe_config.yaml` | `image_embeddings;boxes` | `1x256x32x32;1x4` | `featuremap;featuremap / featuremap;featuremap` |
| `configs/s100/mobile_sam_encoder_nashe_config.yaml` | `normalized_images` | `1x3x512x512` | `featuremap / featuremap` |
| `configs/s100p/mobile_sam_decoder_512_nashm_config.yaml` | `image_embeddings;boxes` | `1x256x32x32;1x4` | `featuremap;featuremap / featuremap;featuremap` |
| `configs/s100p/mobile_sam_encoder_nashm_config.yaml` | `normalized_images` | `1x3x512x512` | `featuremap / featuremap` |
| `configs/s600/mobile_sam_decoder_512_nashp_config.yaml` | `image_embeddings;boxes` | `1x256x32x32;1x4` | `featuremap;featuremap / featuremap;featuremap` |
| `configs/s600/mobile_sam_encoder_nashp_config.yaml` | `normalized_images` | `1x3x512x512` | `featuremap / featuremap` |
| `configs/x5/mobile_sam_decoder_512_box_default_config.yaml` | `image_embeddings; boxes` | `1x256x32x32; 1x4` | `featuremap; featuremap / featuremap; featuremap` |
| `configs/x5/mobile_sam_image_encoder_norm_512x512_config.yaml` | `normalized_images` | `1x3x512x512` | `featuremap / featuremap` |

<a id="validation"></a>
## 转换后验证

本次迁移没有运行导出、校准、编译、模型下载或板测。接收制品前，必须读取真实 SDK metadata，要求两个子模型的输入/输出名称、rank、shape 和 native dtype 匹配；确认 decoder box 仍是 `(1,4)`，并与 512 像素预处理坐标系一致。运行时 cast 不能证明 native 量化 metadata。源没有数据集精度 harness；板测结果需记录图片、box、模型、SDK 和资源条件。

<a id="artifacts"></a>
## 制品

| 目标 | 编码器 | 解码器 | 预期位置 |
|---|---|---|---|
| X5 | `mobile_sam_image_encoder_norm_512x512_allint16.bin` | `mobile_sam_decoder_512_box_default.bin` | 显式准备或本地复制后的 `../model/` |
| S100 | `mobile_sam_image_encoder_norm_512x512_nashe.hbm` | `mobile_sam_decoder_512_nashe.hbm` | `../model/nash-e/` |
| S100P | `mobile_sam_image_encoder_norm_512x512_nashm.hbm` | `mobile_sam_decoder_512_nashm.hbm` | `../model/nash-m/` |
| S600 | `mobile_sam_image_encoder_norm_512x512_nashp.hbm` | `mobile_sam_decoder_512_nashp.hbm` | `../model/nash-p/` |

生成 ONNX、校准 tensor、量化 metadata 和编译模型保持在版本控制之外。准确文件名以 active manifest 和目标 YAML 为准。

<a id="known-gaps"></a>
## 已知缺口

- 上游源版本和 checkpoint digest 没有被固定源锁定。
- S 没有源 `download_assets.py`，获取 checkout 和 checkpoint 是手工前置条件。
- X5 与 S 使用不同校准文件格式和目标配置布局；统一 producer 按 target 分支并保留源配方。
- decoder 输出 rank 及所有编译制品 native dtype 需读取真实 SDK metadata 确认。
- 本次迁移未执行转换、下载或板端验证。

## 来源

源文件到统一文件的映射及 SHA-256 记录在 [`SOURCE_MAP.json`](./SOURCE_MAP.json)。带有 Apache-2.0 标识的源文件保留其来源信息，wrapper 遵循仓库许可证。
