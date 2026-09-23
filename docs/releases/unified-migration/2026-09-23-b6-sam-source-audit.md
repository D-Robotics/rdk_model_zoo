# B6 SAM source protocol audit

这是一份只读源清点，覆盖固定 Git 对象中的 EfficientSAM 与 MobileSAM，以及其 `platforms` 副本。B5 尚未闭环，本文件不启动 B6 实现，不修改计划或台账。所有 board inference、SDK metadata 读取、模型下载和网络操作均未执行。

固定源对象：X5 为 `ac115717197920355fc390bb04299b20e6436864`，S 为 `380e1a2bf42041af54be6f34935e50197cfadff9`。逐文件 SHA-256、文件计数、模式和副本逐字节比较见 [`evidence/2026-09-23-b6-sam-source-inventory.json`](./evidence/2026-09-23-b6-sam-source-inventory.json)。

## 源副本清点

| 平台/样例 | 固定源 repo path | platforms 副本 path | 源文件数 | 副本文件数 | 逐字节相等 | Git 可执行文件 |
|---|---|---|---:|---:|---|---|
| X5 / EfficientSAM | `samples/vision/efficient_sam` | `platforms/x5/samples/vision/efficient_sam` | 26 | 26 | 26/26 | 无 |
| X5 / MobileSAM | `samples/vision/mobile_sam` | `platforms/x5/samples/vision/mobile_sam` | 24 | 24 | 24/24 | 无 |
| S / EfficientSAM | `samples/vision/efficient_sam` | `platforms/s/samples/vision/efficient_sam` | 28 | 28 | 28/28 | 无 |
| S / MobileSAM | `samples/vision/mobile_sam` | `platforms/s/samples/vision/mobile_sam` | 28 | 28 | 28/28 | 无 |

## Source facts

下表的 source path 均为完整 repo path。ONNX 的 float32 是导出协议；`.bin/.hbm` 编译产物的 native dtype 不能由这些源文件确认，不能把 runtime 的 `.astype(np.float32)` 当作 metadata 证明。

| source sample / platform | assets 与默认 | stages、输入与输出 | 数值变换、prompt、mask 与 batch | 量化/转换/evaluator 证据 |
|---|---|---|---|---|
| `samples/vision/efficient_sam` / X5 | `efficient_sam_vitt_encoder_512x512_default_none.bin` + `efficient_sam_vitt_decoder_fixedprompt_512_default.bin`；完整 manifest 证据 `docs/release/x5/models.yaml:85-101` | `pre_process → forward → post_process → predict`。Encoder `batched_images` `1×3×512×512` float32 NCHW → `image_embeddings` `1×256×32×32`；decoder 只接 `image_embeddings`，输出 `low_res_masks` `1×3×128×128`、`iou_predictions` `1×3×1×1`。运行时证据 `platforms/x5/samples/vision/efficient_sam/runtime/python/efficient_sam.py:73-136`；模型接口 `platforms/x5/samples/vision/efficient_sam/model/README.md:25-35` | BGR HWC→RGB，直接线性 resize 到 512 方形，CHW float32 `/255`；固定正点 `(248,210)`、`(302,315)` 在导出时烘焙，导出缩放证据 `platforms/x5/samples/vision/efficient_sam/conversion/scripts/export_decoder_onnx.py:31-44`。post 选最大 IoU，低分辨率 mask 线性上采样到 512 并阈值；不 inverse resize 回原图，不支持多对象/多图 batch。 | YAML 使用 default calibration、未声明 `set_all_nodes_int16`：`platforms/x5/samples/vision/efficient_sam/conversion/configs/efficient_sam_vitt_encoder_featuremap_config.yaml:8-20` 与 `.../efficient_sam_vitt_decoder_fixedprompt_512_default_config.yaml:8-20`。转换文件清单 `platforms/x5/samples/vision/efficient_sam/conversion/README.md:7-24`；evaluator 只有 `hrt_model_exec perf`、无数据集准确率：`platforms/x5/samples/vision/efficient_sam/evaluator/README.md:5-28`。 |
| `samples/vision/mobile_sam` / X5 | `mobile_sam_image_encoder_norm_512x512_allint16.bin` + `mobile_sam_decoder_512_box_default.bin`；`docs/release/x5/models.yaml:314-330` | 同样四阶段。Encoder `normalized_images` `1×3×512×512` float32 NCHW → `image_embeddings` `1×256×32×32`；decoder 接 embedding 与 box，代码实际传 box `[1,4,1,1]`，输出 `low_res_masks` `1×3×128×128`、`iou_predictions` `1×3×1×1`。`platforms/x5/samples/vision/mobile_sam/runtime/python/mobile_sam.py:75-144`；模型接口 `platforms/x5/samples/vision/mobile_sam/model/README.md:25-36` | RGB、512 方形、ImageNet mean `[123.675,116.28,103.53]`、std `[58.395,57.12,57.375]`；默认 box `[185,120,380,445]` 为 resized 512 坐标，可通过 CLI 修改。post 选最大 IoU、上采样到 512、阈值；没有原图 inverse resize、多对象或多图 batch。 | Encoder 配置明确 int16 节点和 `set_all_nodes_int16`，decoder 配置未显式写全节点 int16：`platforms/x5/samples/vision/mobile_sam/conversion/configs/mobile_sam_image_encoder_norm_512x512_config.yaml:7-20,199-212`、`.../mobile_sam_decoder_512_box_default_config.yaml:8-20`。转换文件 `platforms/x5/samples/vision/mobile_sam/conversion/README.md:7-23`；evaluator 只有分模型 perf：`platforms/x5/samples/vision/mobile_sam/evaluator/README.md:5-33`。 |
| `samples/vision/efficient_sam` / S | 每个 march 一对：S100/nash-e、S100P/nash-m、S600/nash-p，共 6 个 `.hbm`；完整 manifest `docs/release/s/models.yaml:228-260` | 四阶段与 X5 EfficientSAM 相同。Encoder `batched_images` `1×3×512×512` float32 → `image_embeddings` `1×256×32×32`；decoder 输入仅 embedding，输出名为 `low_res_masks`/`iou_predictions`。S runtime 的 named-model 包装、cast、选 mask 证据 `platforms/s/samples/vision/efficient_sam/runtime/python/efficient_sam.py:87-152`。S 源配置没有提交 HBM 输出空间 shape。 | RGB、512 方形、`/255`；固定两正点 `(248,210),(302,315)`，导出脚本 `platforms/s/samples/vision/efficient_sam/conversion/scripts/export_decoder_onnx.py:27-58,101-127`。运行时上采样到 512，无原图 inverse resize、无多对象/多图 batch。 | 三套 encoder/decoder YAML 均 `set_all_nodes_int16`、max calibration、`max_percentile: 0.9999`：`platforms/s/samples/vision/efficient_sam/conversion/configs/efficient_sam_encoder_nashe_config.yaml:7-28`、`.../efficient_sam_decoder_nashe_config.yaml:7-28`，nash-m/nash-p 同构。转换另有 `dump_encoder_embedding.py`，完整清单 `platforms/s/samples/vision/efficient_sam/conversion/README.md:7-22`；evaluator 仅 perf、无准确率 harness：`platforms/s/samples/vision/efficient_sam/evaluator/README.md:3-28`。 |
| `samples/vision/mobile_sam` / S | 每个 march 一对：S100/nash-e、S100P/nash-m、S600/nash-p，共 6 个 `.hbm`；`docs/release/s/models.yaml:323-355` | 四阶段。Encoder `normalized_images` `1×3×512×512` float32 → `image_embeddings` `1×256×32×32`；decoder 输入 embedding 与 `boxes` `[1,4]`，输出名为 `low_res_masks`/`iou_predictions`。S runtime box shape 和 named-model 包装证据 `platforms/s/samples/vision/mobile_sam/runtime/python/mobile_sam.py:89-164`。S 源配置没有提交 HBM 输出空间 shape。 | RGB、512 方形、同一组 ImageNet mean/std；默认 box `[185,120,380,445]` 是 resized 512 坐标。post 上采样到 512、无原图 inverse resize、无多对象/多图 batch。 | 三套 encoder/decoder YAML 均显式 `set_all_nodes_int16`、max calibration、`max_percentile: 0.9999`：`platforms/s/samples/vision/mobile_sam/conversion/configs/mobile_sam_encoder_nashe_config.yaml:7-25`、`.../mobile_sam_decoder_512_nashe_config.yaml:7-28`。转换清单和 embedding→decoder calibration 顺序 `platforms/s/samples/vision/mobile_sam/conversion/README.md:9-22,77-145`；evaluator 仅 perf：`platforms/s/samples/vision/mobile_sam/evaluator/README.md:3-28`。 |

## 可共享逻辑

- 两个样例都是真正的双模型 encoder→embedding→decoder 链，均使用 RGB 512 方形 float32 host tensor、`1×256×32×32` embedding、3 候选 mask/IoU 选择、512×512 二值 mask 后处理。
- S 的 board→march 映射可以共用：S100/nash-e、S100P/nash-m、S600/nash-p；每个目标仍必须绑定自己的 HBM 资产。
- 转换顺序可以共用：导出 encoder、取得真实 encoder embedding、准备 decoder calibration、按目标架构编译；S 的 `dump_encoder_embedding.py` 是这一顺序中的显式步骤。

## 不可合并逻辑

- EfficientSAM 的两正点在导出时成为 decoder 常量；MobileSAM 的单 box 是 runtime 输入，不能共用 decoder 输入或 prompt API。
- EfficientSAM 只做 RGB `/255`；MobileSAM 做 ImageNet mean/std，不能共用预处理函数。
- X5 MobileSAM runtime 实际把 box 传成 `[1,4,1,1]`，而 S MobileSAM 传 `[1,4]`；X5 YAML/ONNX 文档又写 `[1,4]`，必须由实际 binding metadata 决定兼容形状。
- X5 使用 bayes-e `.bin`，S 使用 Nash `.hbm`；S 有三套 march 配置，X5 没有 S 的 nash-e/m/p 资产。
- X5 `HB_HBMRuntime.run()` 直接接 tensor dict，S 需要按 model name 再包一层；不能把两个 runtime 调度调用机械合并。
- 四个源树当前只提供 Python runtime 文件，但这只是源文件事实，不构成 SDK、板卡或语言支持验收；不能据此声称“仅 Python 已支持”，也没有验证 C++ 支持。

## dtype/输出 shape 门禁

1. ONNX/export 文档声明的输入和输出是 float32；编译后的 `.bin/.hbm` native dtype 未由提交的模型 metadata 证明。runtime 中 `astype(np.float32)` 只是 CPU 侧转换，不能反推出 SDK native dtype。
2. X5 model README 明确写出 decoder 输出 `1×3×128×128` 与 `1×3×1×1`；S 的 source README/config 只给输出名和后处理路径，没有提交对应 HBM metadata。S 的空间输出 shape 在 binding 实际读取前必须保持 UNKNOWN。
3. 在 B6 实现前，binding 必须逐 asset/march 读取并核对 encoder/decoder 的输入名、rank、shape、native dtype、输出名、rank、shape、native dtype；尤其是 X5 MobileSAM 的 box shape。任何 shape/dtype 不一致都必须阻断运行，不能靠 `reshape` 或 `astype` 放行。
4. Manifest 中 `sha256: null` 代表资产哈希未知；不能把 source 文件 SHA 或 ONNX 量化 cosine 当作发布模型哈希。

## 源清点状态

本次只完成固定源对象、platforms 副本和协议事实清点。B5 未闭环，因此不启动 B6 实现；所有 board 测试均为 not-run，未下载模型，未读取 SDK metadata，未改计划或台账。

