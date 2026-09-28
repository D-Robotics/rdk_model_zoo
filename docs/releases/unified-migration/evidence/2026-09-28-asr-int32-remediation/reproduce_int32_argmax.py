# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""ASR-R1 reproducer: int32 logits round into a blank-token tie before argmax.

Exact fixture from the independent review
(docs/releases/unified-migration/evidence/2026-09-28-asr-independent-review/int32-argmax.json):
the binding accepts an int32 SCALE output with scalar scale=1 and zero_point=0;
raw scores 16777216 (blank ID 0) and 16777217 (token ID 1) inside [1,1,3503]
must decode to token ID 1, but the float32 dequant/decode path rounds 16777217
down to 16777216, manufactures a tie, and transcribe returns the empty string
instead of vocabulary[1].  Synthetic host tensor case through the real
bind_model and ASR.post_process; no board, SDK, model file or quantization run.

Usage (cwd = repository root):
    ../rdk_model_zoo/.venv/bin/python docs/releases/unified-migration/evidence/2026-09-28-asr-int32-remediation/reproduce_int32_argmax.py
Exit 0 = transcribe selects token ID 1; exit 1 = defect reproduced (empty text).
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np


def main() -> int:
    from samples.speech.asr.runtime.python.asr import ASR
    from samples.speech.asr.runtime.python.model_binding import bind_model, resolve_selection

    vocabulary = ("<pad>",) + tuple(f"token{i}" for i in range(1, 3503))
    meta = dict(
        model_names=["asr"], model_name="asr",
        input_names=["audio"], input_shapes={"audio": [1, 30000]},
        input_dtypes={"audio": "float32"}, output_names=["logits"],
        output_shapes={"logits": [1, 1, 3503]}, output_dtypes={"logits": "int32"},
        output_quants={"logits": types.SimpleNamespace(
            quant_type="SCALE", scale=np.array([1.0], np.float32),
            zero_point=np.array([0]), axis=2)})
    raw = np.zeros((1, 1, 3503), np.int32)
    raw[0, 0, 0] = 16777216
    raw[0, 0, 1] = 16777217
    task = ASR(lambda tensors: raw, bind_model(resolve_selection("s100"), meta), vocabulary)
    pair = np.array([16777216, 16777217], np.int32).astype(np.float32)
    actual_ctc = task.post_process(raw)
    actual_legacy = ASR(
        task.runner, task.binding, vocabulary, decode_mode="legacy"
    ).post_process(raw)
    result = {
        "fixture": (
            "int32 SCALE scale=[1.0] zero_point=[0]; raw[0,0,0]=16777216, "
            "raw[0,0,1]=16777217 in [1,1,3503]"
        ),
        "float32_false_tie_premise": bool(pair[0] == pair[1]),
        "expected": "token1",
        "actual_ctc": actual_ctc,
        "actual_legacy": actual_legacy,
        "pass": actual_ctc == "token1" and actual_legacy == "token1",
    }
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
