# Published model downloads

```bash
bash download_model.sh --platform x5 --family yolov8 --task detect --dry-run
bash download_model.sh s600 yolov8 cls n --dry-run
bash fulldownload.sh --platform s100 --dry-run
```

Remove `--dry-run` to download. `--all` selects the complete published inventory for a platform. URLs remain those in release manifests; the host test compares every generated URL with the catalog. YOLOv9 segmentation exists only at c/e on X5/S100/S100P, not S600. X5 keeps flat .bin files, S keeps nash-e/m/p subdirectories. Legacy platform download wrappers save to their original model directory. Canonical runtime downloads to this directory; use `--model-path` to reuse an old download.
