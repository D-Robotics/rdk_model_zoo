[English](./README.md) | 简体中文

# DiffusionDrive 模型评估

运行板端示例后，可对比解码输出与浮点参考：

```bash
python3 compare_outputs.py \
  --reference-npz real_float_reference_outputs.npz \
  --board-npz ../runtime/python/diffusiondrive_outputs.npz
```

结果包含各张量 cosine/MAE 和 BEV argmax 像素一致率。完整 NAVSIM PDM Score 还需要 scene log、sensor blob、地图及 metric cache。
