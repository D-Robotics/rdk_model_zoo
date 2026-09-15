# 模型下载

```bash
bash download_model.sh --platform x5 --family yolov8 --task detect --dry-run
bash download_model.sh s600 yolov8 cls n --dry-run
bash fulldownload.sh --platform s100 --dry-run
```

去掉 `--dry-run` 执行下载；`--all` 获取目标平台全部已发布组合。URL 与发布 Manifest 对照校验。YOLOv9 分割只有 X5/S100/S100P 的 c/e，没有 S600。X5 产物平铺，S 保留 nash-e/m/p 子目录。旧下载入口保存到原平台 model 目录；新运行入口使用本目录，可通过 `--model-path` 复用旧下载。
