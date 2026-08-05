English | [简体中文](./README_cn.md)

# DiffusionDrive Evaluation

After running the board sample, compare decoded outputs with a float reference:

```bash
python3 compare_outputs.py \
  --reference-npz real_float_reference_outputs.npz \
  --board-npz ../runtime/python/diffusiondrive_outputs.npz
```

The report includes tensor cosine/MAE and BEV argmax pixel agreement. Full NAVSIM PDM Score evaluation additionally requires scene logs, sensor blobs, maps, and metric cache.
