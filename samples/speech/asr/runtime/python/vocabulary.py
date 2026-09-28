# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Load the exact published 3503-token mapping outside inference stages."""

import hashlib
import json
from pathlib import Path
from samples.speech.asr.runtime.python.decoding import validate_vocabulary

SHA256 = "33fea3444869c2cd2433f59da079b04ce91515f946d21fe1b0ff3825398bcec7"


def load_vocabulary(path):
    raw = Path(path).expanduser().read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("Vocabulary bytes differ from the published ASR token mapping")
    mapping = json.loads(raw)
    if (
        not isinstance(mapping, dict)
        or len(mapping) != 3503
        or any(type(i) is not int for i in mapping.values())
        or set(mapping.values()) != set(range(3503))
    ):
        raise ValueError("Vocabulary IDs must be unique and contiguous from 0 to 3502")
    tokens = [""] * 3503
    for token, i in mapping.items():
        tokens[i] = token
    return validate_vocabulary(tokens)
