# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Source-compatible score annotation and explicit output file I/O."""
from pathlib import Path
import cv2


def draw_scores(image, texts, result):
    """Return a copy annotated with each prompt's ranked cosine similarity."""
    canvas = image.copy()
    for rank, index in enumerate(result.order, start=1):
        text = f'Rank {rank}: {texts[int(index)]} | similarity: {result.scores[int(index)]:.4f}'
        cv2.putText(canvas, text, (10, 40+(rank-1)*40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    return canvas


def save_image(path, image):
    """Write exactly the caller's visualization path or raise an explicit error."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f'Failed to save image to {path}')
