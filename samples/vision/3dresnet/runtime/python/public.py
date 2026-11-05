# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Public Kinetics-400 R3D-18 variant: source video pixels or prepared F32 clip."""
from pathlib import Path
import argparse, json
import cv2, numpy as np
from utils.py_utils.converted_featuremap import Runner, select
SAMPLE = Path(__file__).resolve().parents[2]
VARIANT = 'tv-r3d18-k400-16x112-featuremap-int16-cpuptq-o1-20261105'
MEAN = [.43216, .394666, .37645]
STD = [.22803, .22145, .216989]


def preprocess(path):
    """Sample a middle 16-frame window at up to 15 fps, never repeat a still."""
    import torch
    from torchvision.models.video import R3D_18_Weights
    cap = cv2.VideoCapture(str(path)); fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok: break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally: cap.release()
    if len(frames) < 16 or not np.isfinite(fps) or fps <= 0:
        raise ValueError('A genuine decoded video with at least 16 frames and known fps is required')
    step = min(max(fps/15., 1.), (len(frames)-1)/15.)
    start = max(0, (len(frames)-1-15*step)/2)
    indices = np.rint(start + np.arange(16)*step).astype(int)
    if len(set(indices.tolist())) != 16:
        raise ValueError('Temporal input must contain 16 distinct video frames')
    clip = torch.from_numpy(np.stack([frames[i] for i in indices])).permute(0,3,1,2)
    value = R3D_18_Weights.KINETICS400_V1.transforms()(clip).unsqueeze(0).numpy()
    return np.ascontiguousarray(value, dtype=np.float32), {
        'frame_indices': indices.tolist(), 'decoded_frames': len(frames), 'source_fps': fps,
        'effective_fps': fps/step, 'spatial': 'torchvision official bilinear resize128x171, center112',
        'frame_content_sha256': [__import__('hashlib').sha256(frames[i].tobytes()).hexdigest() for i in indices]}


def predictions(logits, labels, top_k=5):
    if logits.shape != (1,400) or logits.dtype != np.float32 or not np.isfinite(logits).all() or len(labels) != 400:
        raise ValueError('Expected all 400 finite F32 Kinetics logits and exact source labels')
    score = logits[0].astype(np.float64); score = np.exp(score-score.max()); score /= score.sum()
    return [{'class_id': int(i), 'label': labels[i], 'score': float(score[i])}
            for i in np.argsort(-score)[:top_k]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', choices=['s100'], default='s100'); p.add_argument('--variant', default=VARIANT)
    p.add_argument('--asset-id'); p.add_argument('--model-path', type=Path)
    p.add_argument('--clip', type=Path); p.add_argument('--video', type=Path)
    p.add_argument('--backend', choices=['board','source-onnx'], default='board'); p.add_argument('--source-onnx', type=Path)
    p.add_argument('--dry-run', action='store_true'); a = p.parse_args()
    selection = select(SAMPLE, a.target, a.variant, asset_id=a.asset_id, model_path=a.model_path)
    if a.dry_run: print(json.dumps({'asset_id': selection.asset.reference, 'profile': selection.profile})); return 0
    if bool(a.video) == bool(a.clip): raise ValueError('Choose exactly one genuine video or prepared normalized F32 clip')
    value = preprocess(a.video)[0] if a.video else np.load(a.clip, allow_pickle=False)
    outputs = Runner(selection, backend=a.backend, source_onnx=a.source_onnx)(value)
    labels = json.loads((SAMPLE/'model/kinetics400_labels.json').read_text())
    print(json.dumps({'backend': a.backend, 'asset_id': selection.asset.reference, 'predictions': predictions(outputs['logits'], labels)}))
    return 0


if __name__ == '__main__': raise SystemExit(main())
