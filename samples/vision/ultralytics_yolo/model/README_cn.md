# 模型下载

```bash
bash download_model.sh --platform x5 --family yolov8 --task detect --dry-run
bash download_model.sh s600 yolov8 cls n --dry-run
bash fulldownload.sh --platform s100 --dry-run
```

去掉 `--dry-run` 执行下载；`--all` 获取目标平台全部已发布组合。URL 与发布 Manifest 对照校验。YOLOv9 分割只有 X5/S100/S100P 的 c/e，没有 S600。X5 产物平铺，S 保留 nash-e/m/p 子目录。旧下载入口保存到原平台 model 目录；新运行入口使用本目录，可通过 `--model-path` 复用旧下载。

## YOLO26

```bash
bash download_model.sh --platform s600 --family yolo26 --task cls --model-size n --dry-run
bash fulldownload.sh --platform x5 --family yolo26 --dry-run
```

`--all` 搭配 `--family yolo26` 只列出该平台 25 个 YOLO26 资产；不指定 family 则列出全部系列。共用 X5 全量列表为 92 个，旧 X5 `ultralytics_yolo/model/fulldownload.sh` 仍是原来 67 个；旧 YOLO26 全量脚本为 25 个。服务器 URL 沿用现有发布地址。
