English | [简体中文](./README_cn.md)

# Python Inference

Dependencies: the board's `hbm_runtime`, NumPy and OpenCV. No Torch, ONNX Runtime
or automatic pip installation is required on the board.

```bash
bash run.sh --size n
bash run.sh --size x --test-img /path/image.jpg --output result.jpg
python3 main.py --size s --march nash-m --output result.jpg
```

`run.sh` downloads the selected model and checks its hash. Direct `main.py`
expects the artifacts already in `../../model/<march>/`. Paths to bundled files
are resolved from the script, not the current working directory.

Options: `--size`, `--march`, `--model-path`, `--metadata`, `--test-img`,
`--output`, `--json-output`, `--score-thres`, `--max-det`, `--multi-label`.
A requested march and model metadata must match the board detected from
`soc_name` and `board_type`. S600 and other boards are rejected.

The wrapper returns original-image xyxy boxes, scores, class IDs and full-image
binary masks. Single-label top-k is the default; `--multi-label` permits several
classes at one anchor. Neither mode performs NMS. Resizing uses centered
letterbox with padding 114, matching the conversion pipeline.
