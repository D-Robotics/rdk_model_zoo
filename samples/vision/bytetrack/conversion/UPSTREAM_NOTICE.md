# Upstream model notice

The independently rebuilt S detector uses Ultralytics YOLOv5 v7 source commit `915bbf294bb74c859f0b41f1c23bc395014ea679` and official trained `yolov5x.pt` SHA-256 `9f27a794fa0308e2606f90565164571d6f0a0ba18a3ce2e5e5d323b71c157859`.

- Source: https://github.com/ultralytics/yolov5/tree/915bbf294bb74c859f0b41f1c23bc395014ea679
- Checkpoint: https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt
- Model/source license: GPL-3.0, reproduced in `LICENSE-YOLOv5`.
- Conversion exporter, normalized real-image calibration producer and exact target recipes are in this directory. Original frozen Workbench producer and effective configuration hashes remain in each accepted target record.

ByteTrack association and the detector are separate components. The bundled tracker/license notices and the repository's sample-code license are preserved; they do not replace the detector model's GPL-3.0 license. R01 artifacts were retired before publication; only the independently source-normalized R02 targets may be registered as downloads.
