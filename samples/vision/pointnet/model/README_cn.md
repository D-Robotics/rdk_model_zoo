[English](./README.md) | 简体中文

# PointNet 模型文件

本目录存放 PointNet 椅子部件分割 sample 的 HBM 模型文件。

## 模型检查

```bash
cd samples/vision/pointnet/model
bash download_model.sh s100
```

模型文件路径：

```text
samples/vision/pointnet/model/s100/pointnet.hbm
```

`download_model.sh` 用于检查模型文件是否存在；当前 sample 已提供该 HBM 模型文件。

## 模型说明

| 文件 | 说明 |
| ---- | ---- |
| `s100/pointnet.hbm` | PointNet HBM 模型，输入为归一化椅子点云，输出为四类部件的逐点分割结果。 |

## License

本目录遵循 [Apache 2.0 License](../../../../LICENSE)。
