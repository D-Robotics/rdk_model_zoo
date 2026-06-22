[English](./README.md) | 简体中文

# 模型文件说明

本目录用于存放 YOLOv13 iMoonLab Detect 的参考 `.hbm` 模型和下载脚本。

## 下载模型

```bash
cd model
bash download_model.sh
```

脚本会将参考模型下载到 `./s100/` 目录。

## 参考模型列表

以下参考模型地址来自原始 YOLOv13 iMoonLab demo 的 `source/reference_hbm_models/README.md`：

- `yolo13n_detect_nashe_640x640_nv12.hbm`
- `yolo13s_detect_nashe_640x640_nv12.hbm`
- `yolo13l_detect_nashe_640x640_nv12.hbm`
- `yolo13x_detect_nashe_640x640_nv12.hbm`

## 说明

- 当前目录提供的公开参考模型均为 `nashe` 版本。
- 输入格式为 NV12，运行时按 Y plane 与 UV plane 两个输入 tensor 送入。
- Python runtime 默认使用 `yolo13n_detect_nashe_640x640_nv12.hbm`。

## License

本目录内容遵循仓库顶层 `LICENSE`。
