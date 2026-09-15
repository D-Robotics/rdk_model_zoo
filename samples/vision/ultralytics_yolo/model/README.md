# Published model downloads

```bash
bash download_model.sh --platform x5 --family yolov8 --task detect --dry-run
bash download_model.sh s600 yolov8 cls n --dry-run
bash fulldownload.sh --platform s100 --dry-run
```

Remove `--dry-run` to download. `--all` selects the complete published inventory for a platform. URLs remain those in release manifests; the host test compares every generated URL with the catalog. YOLOv9 segmentation exists only at c/e on X5/S100/S100P, not S600. X5 keeps flat .bin files, S keeps nash-e/m/p subdirectories. Legacy platform download wrappers save to their original model directory. Canonical runtime downloads to this directory; use `--model-path` to reuse an old download.

## YOLO26

```bash
bash download_model.sh --platform s600 --family yolo26 --task cls --model-size n --dry-run
bash fulldownload.sh --platform x5 --family yolo26 --dry-run
```

`--all --family yolo26` selects 25 assets per platform; omit family for the full inventory. The canonical X5 full inventory contains 92 assets; the old X5 generic full-download wrapper retains its original 67, and the old YOLO26 full-download wrapper selects 25. Published server URLs are unchanged.
