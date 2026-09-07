# RDK Model Zoo S v1.1.0

Benchmark data refresh for the S100 / S100P / S600 line. No board tests were run; values are transcribed from the `rdk_s` sample READMEs and evaluator documents.

## Changed

- `docs/release/benchmarks.yaml` expanded from 58 to 473 records: Ultralytics YOLO (368), YOLO26 (90), SigLIP (16), YOLO26 Depth (20), YOLOv13 (8), plus per-device S100/S100P/S600 rows.
- `docs/release/models.yaml` adds ACT and Pi0 (rdk_LeRobot_tools submodules).
- Removed erroneous `cls-640` classification rows (copy-paste of the 640x640/80-class detection config).

## Known limitations

- Several S samples are runtime demos without a published board benchmark; those remain unmeasured in this release.
