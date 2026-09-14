[English](./README.md) | [简体中文](./README_cn.md)

# 模型文件

RDK X5 模型下载到 `model/bayes-e/`，模型二进制文件不直接存放在仓库中。

| 模型 | 文件 | 输入 | 输出 | SHA256 |
| --- | --- | --- | --- | --- |
| HIMLoco Go2 | `himloco_go2_bayese_1x270.bin` | float32 `[1,270,1,1]` | float32 `[1,12,1,1]` | `7ce46ca2628f8bc236da0e8564180a1de92847bddf1ec00717ce7aa93e8c3e6a` |

## 下载

```bash
bash download_model.sh
```

脚本会跳过已存在的文件，并在返回前验证 SHA256。
