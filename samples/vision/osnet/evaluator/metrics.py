"""Market1501 single-query evaluation, preserving the source experiment protocol."""

from pathlib import Path
import re


def records(folder):
    result = []
    for path in sorted(Path(folder).glob("*.jpg")):
        match = re.match(r"(-?\d+)_c(\d)", path.name)
        if not match:
            raise ValueError(f"Not a Market1501 image name: {path.name}")
        pid, camera = map(int, match.groups())
        if pid != -1:
            result.append((path, pid, camera))
    return result


def retrieval(query, gallery, qf, gf, *, normalize=True):
    import numpy as np

    qf, gf = np.asarray(qf, np.float32), np.asarray(gf, np.float32)
    if qf.shape != (len(query), 512) or gf.shape != (len(gallery), 512):
        raise ValueError("Feature rows must match the ordered query/gallery manifests and have 512 columns")
    if not np.isfinite(qf).all() or not np.isfinite(gf).all():
        raise ValueError("Features must be finite")
    if normalize:
        qf = qf / np.maximum(np.linalg.norm(qf, axis=1, keepdims=True), 1e-12)
        gf = gf / np.maximum(np.linalg.norm(gf, axis=1, keepdims=True), 1e-12)
    gp = np.array([r[1] for r in gallery])
    gc = np.array([r[2] for r in gallery])
    aps, rank1 = [], []
    for index, (_, pid, camera) in enumerate(query):
        distance = np.sum((gf - qf[index]) ** 2, axis=1)
        order = np.argsort(distance, kind="stable")
        order = order[~((gp[order] == pid) & (gc[order] == camera)) & (gp[order] != -1)]
        matches = gp[order] == pid
        if not matches.any():
            continue
        precision = np.cumsum(matches) / np.arange(1, len(matches) + 1)
        aps.append(float(precision[matches].mean()))
        rank1.append(float(matches[0]))
    if not aps:
        raise ValueError("No query has a valid cross-camera gallery match")
    return {"rank1_percent": 100 * float(np.mean(rank1)), "mAP_percent": 100 * float(np.mean(aps)),
            "valid_queries": len(aps), "normalized": normalize, "reranking": False,
            "distance": "squared Euclidean", "samples": len(query) + len(gallery)}
