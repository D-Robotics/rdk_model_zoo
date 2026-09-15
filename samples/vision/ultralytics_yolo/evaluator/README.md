# Evaluation

The four `eval_yolo_det.py`, `eval_yolo_seg.py`, `eval_yolo_pose.py`, `eval_yolo_cls.py` scripts share platform selection and current runtime classes. Run `--help` for parameters. COCO requires pycocotools, image directory and annotation JSON; classifiers need ImageNet `--val-txt` or synset `--label-file`. No dependency is installed automatically.

COCO decoders use standard COCO class-index/category-ID ordering, including when annotations are a subset. Custom class order is not supported by this evaluator. Without annotations, prediction dumps require numeric image filenames for stable IDs. `--limit` evaluates only selected image IDs. An empty detection result writes an empty JSON and reports that metric computation was skipped; it is not a passing accuracy result. NMS defaults to 0.70 for evaluators on both platforms; this differs from the S runtime CLI default. Thresholds and dataset must accompany any reported score.

Original Benchmark tables remain under the platform sample READMEs and manifests. This merge does not claim to regenerate their metrics. Board accuracy, latency and C++ compilation are still pending release validation.
