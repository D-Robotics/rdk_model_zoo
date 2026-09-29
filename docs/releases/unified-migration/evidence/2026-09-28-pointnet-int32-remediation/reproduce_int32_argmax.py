# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""POINTNET-R2 reproducer: int32 logits lose ordering before argmax.

Exact fixture from the independent review
(docs/releases/unified-migration/evidence/2026-09-28-pointnet-independent-review/int32-argmax.json):
binding accepts an int32 SCALE output with scalar scale=1 and zero_point=0; raw
scores [16777216, 16777217, 0, 0] must select class 1, but a float32 dequant
path rounds 16777217 down to 16777216, manufactures a tie, and argmax returns
class 0. Synthetic host tensor case through the real PointNetTask.post_process;
no board, SDK, model file or quantization run.

Usage (cwd = repository root):
    .venv/bin/python docs/releases/unified-migration/evidence/2026-09-28-pointnet-int32-remediation/reproduce_int32_argmax.py
Exit 0 = post_process selects class 1; exit 1 = defect reproduced (actual 0).
"""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import types


def main() -> int:
    from samples.vision.pointnet.runtime.python.model_binding import bind_model, resolve_selection
    from samples.vision.pointnet.runtime.python.pointnet import PointNetTask

    meta = dict(model_names=['pointnet'], model_name='pointnet',
                input_names=['point'], input_shapes={'point': [1, 3, 1]},
                input_dtypes={'point': 'float32'}, output_names=['pred'],
                output_shapes={'pred': [1, 1, 4]}, output_dtypes={'pred': 'int32'},
                output_quants={'pred': types.SimpleNamespace(
                    quant_type=types.SimpleNamespace(name='SCALE'),
                    scale=np.array([1.0], np.float32),
                    zero_point=np.array([0]), axis=2)})
    raw = np.array([[[16777216, 16777217, 0, 0]]], np.int32)
    task = PointNetTask(lambda tensors: raw, bind_model(resolve_selection('s100'), meta))
    actual = task.post_process(raw).tolist()
    raw_after = raw.tolist()
    result = {
        'fixture': 'int32 SCALE scale=[1.0] zero_point=[0]; raw=[[[16777216,16777217,0,0]]]',
        'expected': [1],
        'actual': actual,
        'raw_after_post_process': raw_after,
        'pass': actual == [1] and raw_after == raw.tolist(),
    }
    print(json.dumps(result, indent=2))
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
