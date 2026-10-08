English | [简体中文](README_cn.md)

# Batch COCO evaluation

These scripts generate COCO-format predictions and evaluate detection, instance segmentation and pose results with `pycocotools` on COCO2017 validation data. See the [dataset preparation guide](en_COCO2017val.md).

- `eval_batch_python.py` and `eval_batch_cpp.py` run the selected sample evaluator for `.bin` models in a directory.
- `eval_pytorch_generate_labels*.py` and their batch wrappers generate predictions from PyTorch models.
- `eval_pycocotools.py`, `eval_pycocotools_seg.py` and `eval_pycocotools_pose.py` evaluate bounding boxes, segmentation masks and keypoints, respectively.

Run a scoring script in an environment with `pycocotools` and a prepared ground-truth annotation file. The detection evaluator accepts either a prediction JSON file or a directory of JSON files:

```bash
python3 tools/batch_eval_pycocotools/eval_pycocotools.py \
  --truth /path/to/instances_val2017.json \
  --json /path/to/predictions.json
```

Prediction generation requires the selected model's runtime, dependencies and assets. Check the selected script's `--help` for its parameters before running it.
