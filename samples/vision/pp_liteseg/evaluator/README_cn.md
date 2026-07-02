# 评估说明

本目录记录 PP-LiteSeg-STDC1 在 RDK X5 上的验证说明。

## 数值一致性验证

使用 `hb_mapper infer` 或 `hb_verifier` 对比量化 ONNX 与生成 `.bin` 的输出。对语义分割模型，建议先以 cosine similarity 不低于 `0.95` 作为基础门槛，再进行数据集级 mIoU 评估。

## 数据集级评估

完整精度验证可使用原始 PaddleSeg 流程评估浮点模型，并使用自定义 RDK X5 推理流程在同一验证集上对比 mIoU。

## 性能验证

在 OpenExplorer 环境中使用 `hb_perf`，在板端使用 `hrt_model_exec perf`：

```bash
hb_perf ../conversion/ptq_yamls/pp_liteseg_stdc1_cityscapes_1024x512_nv12_output/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin
```
