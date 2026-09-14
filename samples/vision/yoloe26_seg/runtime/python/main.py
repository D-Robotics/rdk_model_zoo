# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Run a YOLOE-26 PF image sample on S100 or S100P."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from yoloe26seg import (SAMPLE, YoloE26Seg, YoloE26SegConfig, SIZES, MARCHES,
                       hbm_name, model_stem, detect_march)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=SIZES, default="n")
    parser.add_argument("--march", choices=MARCHES, help="Must match the detected board")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--test-img", type=Path, default=SAMPLE / "test_data" / "office_desk.jpg")
    parser.add_argument("--output", type=Path, default=Path("result.jpg"))
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--score-thres", type=float, default=.25)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--multi-label", action="store_true")
    args = parser.parse_args()
    march = detect_march()
    if args.march and args.march != march:
        parser.error(f"Requested march {args.march} differs from board {march}")
    folder = SAMPLE / "model" / march
    filename = hbm_name(args.size, march)
    model_path = args.model_path or folder / filename
    metadata = args.metadata or folder / f"{model_stem(args.size)}.json"
    for path in (model_path, metadata, args.test_img):
        if not path.is_file():
            parser.error(f"File not found: {path}. Run model/download_model.sh {march} {args.size} first.")
    image = cv2.imread(str(args.test_img))
    if image is None:
        parser.error(f"Cannot decode image: {args.test_img}")
    model = YoloE26Seg(YoloE26SegConfig(str(model_path), str(metadata), args.score_thres,
                                  args.max_det, not args.multi_label))
    if model.metadata["size"] != args.size:
        parser.error("--size differs from model metadata")
    raw = model.forward(image)
    boxes, scores, labels, masks = model.post_process(raw, image.shape)
    names = model.metadata["names"]
    records = []
    for box, score, label, mask in zip(boxes, scores, labels, masks):
        color = tuple(int(x) for x in np.random.default_rng(int(label)).integers(40, 240, 3))
        image[mask] = (image[mask] * .6 + np.array(color) * .4).astype(np.uint8)
        x1, y1, x2, y2 = np.round(box).astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{names[label]} {score:.3f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
        records.append({"box": box.tolist(), "score": float(score), "class_id": int(label),
                        "name": names[label]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), image):
        raise RuntimeError(f"Cannot write {args.output}")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(records)} detections: {args.output}")


if __name__ == "__main__":
    main()
