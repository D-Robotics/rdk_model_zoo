English | [简体中文](./README_cn.md)

# EfficientSAM 转换

本目录记录 EfficientSAM ViT-Tiny 图像编码器和固定双正点解码器的转换能力。统一脚本通过 `--target` 选择 X5 或 RDK-S；本次迁移没有运行转换流程。

<a id="source-model"></a>
## 源模型

源项目是 [yformer/EfficientSAM](https://github.com/yformer/EfficientSAM)，使用 `weights/efficient_sam_vitt.pt`。固定源没有提供 checkpoint 版本或 SHA-256。X5 与 S 的上游 builder API 不同，因此 `scripts/export_encoder_onnx.py` 和 `scripts/export_decoder_onnx.py` 在 `--target` 后保留两个分支。

解码器导出时将两个正点 `(248, 210)`、`(302, 315)` 固定进 ONNX，不接收运行时 prompt。编码器契约为 `batched_images` `(1,3,512,512)`、float32、RGB NCHW，输出 `image_embeddings` `(1,256,32,32)`、float32。解码器读取该 embedding，输出 `low_res_masks` 与 `iou_predictions`。

<a id="toolchain-targets"></a>
## 工具链与目标

应在 x86 主机的对应工具链环境中运行。以下是源转换命令，本次迁移没有执行。

| 目标 | march/配置目录 | 源工具链 | 量化器 |
|---|---|---|---|
| `x5` | `configs/x5/` | `openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310` | `hb_mapper makertbin` |
| `s100` | `configs/s100/` | OE S 系列 3.7.0 | `hb_compile` / `nash-e` |
| `s100p` | `configs/s100p/` | OE S 系列 3.7.0 | `hb_compile` / `nash-m` |
| `s600` | `configs/s600/` | OE S 系列 3.7.0 | `hb_compile` / `nash-p` |

有效制品清单是 `docs/release/x5/models.yaml` 或 `docs/release/s/models.yaml`；其中发布模型的 SHA 字段均为 `null`（未知）。S 为三个 march 分别提供编码器和解码器 YAML，X5 提供两份 bayes-e YAML。

导出与浮点 embedding 生成依赖主机 PyTorch、ONNX、ONNX Runtime、NumPy、OpenCV。固定源没有锁定这些包的版本或上游仓库 revision，这仍是复现前提，不是经过验证的环境规格。在选定导出环境中检查导入，与板端推理环境分开：

```bash
# cwd: 本 conversion 目录；在选定主机导出环境中执行
python3 -c "import torch, onnx, onnxruntime, numpy, cv2; print(torch.__version__, onnx.__version__, onnxruntime.__version__)"
```

<a id="export"></a>
## 导出 ONNX

先进入 `samples/vision/efficient_sam/conversion`。X5 使用源 helper 准备上游树和默认 checkpoint；X5 上游 builder 只读取 `<repo>/weights/efficient_sam_vitt.pt`，传入不同的 `--checkpoint` 会被 exporter 显式拒绝。固定源没有 S 目标的下载 helper，因此 S 目标必须由使用者手工放置上游 checkout 和 checkpoint。

```bash
# 仅 X5；本迁移未执行
python3 scripts/download_assets.py --target x5 --workspace ./workspace

# S 目标：手工准备以下路径
# ./workspace/EfficientSAM
# ./workspace/EfficientSAM/weights/efficient_sam_vitt.pt
```

X5 helper 会先尝试 `git clone`；失败后下载上游 `main` 源 ZIP，并在私有暂存目录中只安全解压 `EfficientSAM-main/` 树，再移动到 workspace。随后在已有 `efficient_sam_vitt.pt` 不存在或小于 10,000,000 字节时下载 checkpoint。这些动作都必须由用户显式调用；上游 `main` ZIP 不是固定源版本。S 源没有等价 helper，因此没有自动获取源代码或 checkpoint 的路径。

导出编码器和固定 prompt 解码器：

```bash
python3 scripts/export_encoder_onnx.py --target x5 \
  --repo ./workspace/EfficientSAM \
  --checkpoint ./workspace/EfficientSAM/weights/efficient_sam_vitt.pt \
  --output ./efficient_sam_vitt_encoder_512_splitqkv_op11.onnx
python3 scripts/export_decoder_onnx.py --target x5 \
  --repo ./workspace/EfficientSAM \
  --checkpoint ./workspace/EfficientSAM/weights/efficient_sam_vitt.pt \
  --output ./efficient_sam_vitt_decoder_fixedprompt_512_op11.onnx
```

S100、S100P、S600 分别将 `--target` 改为对应值，输出名使用 `efficient_sam_vitt_encoder_512_op11.onnx` 和 `efficient_sam_vitt_decoder_512_op11.onnx`。导出器默认 opset 11、尺寸 512，也可用 `--size`、`--opset`、decoder 的 `--points` 显式覆盖。

<a id="calibration"></a>
## 校准

编码器校准必须使用真实 RGB 图片，输入为 `/255` 后的 float32 RGB CHW `(1,3,512,512)`；源 producer 要求至少 20 个输出文件。图片不足时 producer 会重复源路径补足，这不代表有 20 张独立代表性图片，因此校准集仍需单独审计。解码器校准必须从真实的 `(1,256,32,32)` float32 encoder embedding 开始，该 embedding 由浮点 ONNX 编码器或已编译编码器路径产生。固定 prompt 已在 decoder 内，不生成额外 prompt 张量。

```bash
python3 scripts/prepare_calibration.py --target x5 \
  --src ./calibration_images --out . --num 30 --size 512
# ./calibration_data_rgbchw_512/*.rgbchw
python3 scripts/dump_encoder_embedding.py --image ../test_data/dogs.jpg \
  --onnx ./efficient_sam_vitt_encoder_512_splitqkv_op11.onnx --output ./encoder_embedding.bin

python3 scripts/prepare_efficient_decoder_calibration.py --target x5 \
  --embedding ./encoder_embedding.bin --out ./decoder_calibration --num 30
# ./decoder_calibration/calibration_embeddings/*.bin
```

S 目标需显式指定与 YAML 一致的校准根目录：

```bash
python3 scripts/prepare_calibration.py --target s100 \
  --src ./calibration_images --out ./calibration_data --num 30 --size 512
python3 scripts/dump_encoder_embedding.py --image ../test_data/dogs.jpg \
  --onnx ./efficient_sam_vitt_encoder_512_op11.onnx --output ./encoder_embedding.bin
python3 scripts/prepare_efficient_decoder_calibration.py --target s100 \
  --embedding ./encoder_embedding.bin --out ./decoder_calibration --num 30
```

这会生成匹配 S YAML 的 `./calibration_data/batched_images/*.npy` 和 `./decoder_calibration/image_embeddings/*.npy`。dump helper 运行导出的浮点 ONNX 编码器并使用源 RGB `/255` 变换；它需要主机 `onnxruntime`，本次没有运行。此次迁移没有在主机或板端推理生成 embedding。

dump 示例在 ONNX 导出后使用已提交的 `../test_data/dogs.jpg`，输入文件可直接定位；实际量化应换成有代表性的校准图片。该 helper 源自 S 分支，按同一 tensor 协议可读取两类目标的浮点 encoder ONNX，本轮仅执行注入 ORT 的 fixture。`calibration_images/` 数据集由用户准备，仓库不提供。单份 embedding 缩放只保留源演示配方，不等同于代表性校准数据集。提交配置与 runtime 仅覆盖 size 512；改变导出 `--size` 还需要对应的新配置与 runtime binding。

<a id="compile"></a>
## 编译

在目标 ONNX 和校准文件存在后，从本目录运行：

```bash
python3 scripts/quantize.py --target x5
python3 scripts/quantize.py --target s100
python3 scripts/quantize.py --target s100p
python3 scripts/quantize.py --target s600
```

`--target <target> --config <path>` 可只编译一份已提交 YAML。X5 调用 `hb_mapper makertbin --model-type onnx`；S 调用 `hb_compile --config`。X5 YAML 使用源默认校准，没有显式声明全节点 int16；S YAML 使用 `set_all_nodes_int16`、max 校准、`0.9999` percentile 和对应 Nash march。提交的精确路径如下：

| 目标 | 角色 | ONNX | 校准目录 | 工作目录 | 输出前缀 |
|---|---|---|---|---|---|
| S100 | encoder | `efficient_sam_vitt_encoder_512_op11.onnx` | `./calibration_data/batched_images` | `bpu_model_output_encoder_nashe` | `efficient_sam_vitt_encoder_512x512_nashe` |
| S100 | decoder | `efficient_sam_vitt_decoder_512_op11.onnx` | `./decoder_calibration/image_embeddings` | `bpu_model_output_decoder_nashe` | `efficient_sam_vitt_decoder_512_nashe` |
| S100P | encoder | `efficient_sam_vitt_encoder_512_op11.onnx` | `./calibration_data/batched_images` | `bpu_model_output_encoder_nashm` | `efficient_sam_vitt_encoder_512x512_nashm` |
| S100P | decoder | `efficient_sam_vitt_decoder_512_op11.onnx` | `./decoder_calibration/image_embeddings` | `bpu_model_output_decoder_nashm` | `efficient_sam_vitt_decoder_512_nashm` |
| S600 | encoder | `efficient_sam_vitt_encoder_512_op11.onnx` | `./calibration_data/batched_images` | `bpu_model_output_encoder_nashp` | `efficient_sam_vitt_encoder_512x512_nashp` |
| S600 | decoder | `efficient_sam_vitt_decoder_512_op11.onnx` | `./decoder_calibration/image_embeddings` | `bpu_model_output_decoder_nashp` | `efficient_sam_vitt_decoder_512_nashp` |

各配置与预期编译输出（相对本 conversion 工作目录）：

| Config | ONNX | Calibration | Compiler output |
| --- | --- | --- | --- |
| `configs/s100/efficient_sam_decoder_nashe_config.yaml` | `./efficient_sam_vitt_decoder_512_op11.onnx` | `./decoder_calibration/image_embeddings` | `bpu_model_output_decoder_nashe/efficient_sam_vitt_decoder_512_nashe.hbm` |
| `configs/s100/efficient_sam_encoder_nashe_config.yaml` | `./efficient_sam_vitt_encoder_512_op11.onnx` | `./calibration_data/batched_images` | `bpu_model_output_encoder_nashe/efficient_sam_vitt_encoder_512x512_nashe.hbm` |
| `configs/s100p/efficient_sam_decoder_nashm_config.yaml` | `./efficient_sam_vitt_decoder_512_op11.onnx` | `./decoder_calibration/image_embeddings` | `bpu_model_output_decoder_nashm/efficient_sam_vitt_decoder_512_nashm.hbm` |
| `configs/s100p/efficient_sam_encoder_nashm_config.yaml` | `./efficient_sam_vitt_encoder_512_op11.onnx` | `./calibration_data/batched_images` | `bpu_model_output_encoder_nashm/efficient_sam_vitt_encoder_512x512_nashm.hbm` |
| `configs/s600/efficient_sam_decoder_nashp_config.yaml` | `./efficient_sam_vitt_decoder_512_op11.onnx` | `./decoder_calibration/image_embeddings` | `bpu_model_output_decoder_nashp/efficient_sam_vitt_decoder_512_nashp.hbm` |
| `configs/s600/efficient_sam_encoder_nashp_config.yaml` | `./efficient_sam_vitt_encoder_512_op11.onnx` | `./calibration_data/batched_images` | `bpu_model_output_encoder_nashp/efficient_sam_vitt_encoder_512x512_nashp.hbm` |
| `configs/x5/efficient_sam_vitt_decoder_fixedprompt_512_default_config.yaml` | `./efficient_sam_vitt_decoder_fixedprompt_512_op11.onnx` | `./decoder_calibration/calibration_embeddings` | `bpu_model_output_decoder_fixedprompt/efficient_sam_vitt_decoder_fixedprompt_512_default.bin` |
| `configs/x5/efficient_sam_vitt_encoder_featuremap_config.yaml` | `./efficient_sam_vitt_encoder_512_splitqkv_op11.onnx` | `./calibration_data_rgbchw_512` | `bpu_model_output_512_default_none/efficient_sam_vitt_encoder_512x512_default_none.bin` |

单配置命令必须同时显式指定匹配的 `--target`；脚本不从 YAML 推断编译器，省略 target 会默认 X5。例如：`python3 scripts/quantize.py --target s100 --config configs/s100/efficient_sam_decoder_nashe_config.yaml`。

以下输入字段逐字取自各编译配置，不能视为实测 SDK metadata；转换后仍须读取模型 metadata 校验。X5 的 type 字段依次为 runtime / train，S 为 input_type。

| Config | input_name | input_shape | type |
| --- | --- | --- | --- |
| `configs/s100/efficient_sam_decoder_nashe_config.yaml` | `image_embeddings` | `1x256x32x32` | `featuremap / featuremap` |
| `configs/s100/efficient_sam_encoder_nashe_config.yaml` | `batched_images` | `1x3x512x512` | `featuremap / featuremap` |
| `configs/s100p/efficient_sam_decoder_nashm_config.yaml` | `image_embeddings` | `1x256x32x32` | `featuremap / featuremap` |
| `configs/s100p/efficient_sam_encoder_nashm_config.yaml` | `batched_images` | `1x3x512x512` | `featuremap / featuremap` |
| `configs/s600/efficient_sam_decoder_nashp_config.yaml` | `image_embeddings` | `1x256x32x32` | `featuremap / featuremap` |
| `configs/s600/efficient_sam_encoder_nashp_config.yaml` | `batched_images` | `1x3x512x512` | `featuremap / featuremap` |
| `configs/x5/efficient_sam_vitt_decoder_fixedprompt_512_default_config.yaml` | `image_embeddings` | `1x256x32x32` | `featuremap / featuremap` |
| `configs/x5/efficient_sam_vitt_encoder_featuremap_config.yaml` | `batched_images` | `1x3x512x512` | `featuremap / featuremap` |

<a id="validation"></a>
## 转换后验证

本次迁移没有运行导出、校准、编译或板测，状态为 `not-run`。源历史 X5 量化记录保留在 [`QUANTIZATION_STATUS.md`](./QUANTIZATION_STATUS.md) 和 [`VALIDATION.md`](./VALIDATION.md)：编码器 cosine `0.968013`，decoder 的 `low_res_masks=0.965641`、`iou_predictions=0.997313`。这些是源历史值，不是本轮结果。S 源 evaluator 只有性能材料，没有数据集精度 harness。

声称制品可用前，必须读取每个目标真实 SDK metadata，核对输入/输出名称、rank、shape 和 native dtype。源中没有编译 HBM metadata，运行时转换 float 不能证明 native dtype。

<a id="artifacts"></a>
## 制品

| 目标 | 编码器 | 解码器 | 预期位置 |
|---|---|---|---|
| X5 | `efficient_sam_vitt_encoder_512x512_default_none.bin` | `efficient_sam_vitt_decoder_fixedprompt_512_default.bin` | 显式准备或本地复制后的 `../model/` |
| S100 | `nash-e/efficient_sam_vitt_encoder_512x512_nashe.hbm` | `nash-e/efficient_sam_vitt_decoder_512_nashe.hbm` | `../model/nash-e/` |
| S100P | `nash-m/efficient_sam_vitt_encoder_512x512_nashm.hbm` | `nash-m/efficient_sam_vitt_decoder_512_nashm.hbm` | `../model/nash-m/` |
| S600 | `nash-p/efficient_sam_vitt_encoder_512x512_nashp.hbm` | `nash-p/efficient_sam_vitt_decoder_512_nashp.hbm` | `../model/nash-p/` |

目标 YAML 和 active manifest 是输出前缀及运行时文件名的依据。ONNX、校准 tensor、量化 metadata 和编译模型不会提交在此目录。

<a id="known-gaps"></a>
## 已知缺口

- 上游 checkpoint 没有固定源版本或 digest。
- X5 上游 builder 只读取 `<repo>/weights/efficient_sam_vitt.pt`；不同 checkpoint 路径会被显式拒绝，S 则使用显式 `--checkpoint`。
- S 源没有 `download_assets.py`，源获取是手工前置条件。
- X5 与 S 的上游 builder API、量化器和配置布局不同；统一脚本显式分支，没有把它们伪装成字节相同的配方。
- 编译制品的 native dtype 以及 S 的输出空间 metadata，须等 binding 读取真实 SDK metadata 后确认。
- 本次迁移未执行转换、模型下载或板端验证。

## 来源

合并关系与 source SHA-256 映射记录在 [`SOURCE_MAP.json`](./SOURCE_MAP.json)。带有源 Apache-2.0 标识的代码和文档保留其来源信息，统一 wrapper 遵循仓库许可证。
