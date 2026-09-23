"""Small OpenCV visualization helpers for YOLOWorld."""
from pathlib import Path
import cv2
import numpy as np

def draw_results(image, result, class_names):
    output = image.copy()
    for box, score, cid in zip(result.boxes, result.scores, result.class_ids):
        x1, y1, x2, y2 = map(int, box); color = (0, 180, 255)
        cv2.rectangle(output, (x1,y1), (x2,y2), color, 2)
        name = class_names[int(cid)] if 0 <= int(cid) < len(class_names) else str(int(cid))
        cv2.putText(output, f"{name}: {float(score):.3f}", (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
    return output

def save_image(path, image):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image): raise OSError(f"Could not save {path}")
