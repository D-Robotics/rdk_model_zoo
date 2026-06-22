English | [简体中文](./README_cn.md)

# PointNet Model Files

This directory stores the HBM model file for the PointNet chair part segmentation sample.

## Model Check

```bash
cd samples/vision/pointnet/model
bash download_model.sh s100
```

Model file path:

```text
samples/vision/pointnet/model/s100/pointnet.hbm
```

`download_model.sh` checks whether the model file exists. This sample already provides the HBM model file.

## Model Artifact

| File | Description |
| ---- | ----------- |
| `s100/pointnet.hbm` | PointNet HBM model for normalized chair point cloud input and four-part per-point segmentation output. |

## License

This directory is licensed under the [Apache 2.0 License](../../../../LICENSE).
