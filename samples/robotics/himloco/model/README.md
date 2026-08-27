[English](./README.md) | [简体中文](./README_cn.md)

# Model Files

The RDK X5 model is downloaded into `model/bayes-e/`. Model binaries are not
stored in the repository.

| Model | File | Input | Output | SHA256 |
| --- | --- | --- | --- | --- |
| HIMLoco Go2 | `himloco_go2_bayese_1x270.bin` | float32 `[1,270,1,1]` | float32 `[1,12,1,1]` | `7ce46ca2628f8bc236da0e8564180a1de92847bddf1ec00717ce7aa93e8c3e6a` |

## Download

```bash
bash download_model.sh
```

The script skips an existing file and verifies its SHA256 before returning.
