# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Greedy CTC in float32 or exact comparison precision, plus an explicit
archived concatenate-only comparison mode."""

from numbers import Integral
import numpy as np


def validate_vocabulary(vocabulary):
    if (
        not isinstance(vocabulary, (list, tuple))
        or not vocabulary
        or any(not isinstance(token, str) or not token for token in vocabulary)
        or vocabulary[0] != "<pad>"
        or len(set(vocabulary)) != len(vocabulary)
    ):
        raise ValueError("Expected unique nonempty ordered tokens with <pad> at ID 0")
    return tuple(vocabulary)


def decode_ids(ids, vocabulary, mode="ctc"):
    tokens = validate_vocabulary(vocabulary)
    if mode not in ("ctc", "legacy"):
        raise ValueError("decode-mode must be ctc or legacy")
    output = []
    previous = None
    for token in ids:
        if (
            isinstance(token, (bool, np.bool_))
            or not isinstance(token, Integral)
            or not 0 <= token < len(tokens)
        ):
            raise ValueError("Token IDs must be integers within the vocabulary")
        if token != 0 and (mode == "legacy" or token != previous):
            output.append(tokens[token])
        previous = token
    return "".join(output)


def decode_logits(logits, vocabulary, mode="ctc"):
    tokens = validate_vocabulary(vocabulary)
    if (
        not isinstance(logits, np.ndarray)
        or logits.ndim != 3
        or logits.shape[0] != 1
        or logits.shape[1] < 1
        or logits.shape[2] != len(tokens)
        or logits.dtype != np.float32
        or not np.isfinite(logits).all()
    ):
        raise ValueError("Expected finite float32 logits [1,T,vocabulary_size], T > 0")
    return decode_ids(np.argmax(logits[0], axis=-1), tokens, mode)


def decode_exact_logits(logits, vocabulary, mode="ctc"):
    """Greedy decode of float64 logits carrying exact integer affine scores.

    Integer SCALE outputs dequantized with ``dequantize_tensor(dtype="float64")``
    keep distinct raw scores distinct through argmax; the float32
    :func:`decode_logits` rounds adjacent int32 magnitudes such as 2**24 and
    2**24 + 1 into artificial ties.  Genuinely equal scores still tie to the
    lowest ID exactly like :func:`decode_logits`.
    """
    tokens = validate_vocabulary(vocabulary)
    if (
        not isinstance(logits, np.ndarray)
        or logits.ndim != 3
        or logits.shape[0] != 1
        or logits.shape[1] < 1
        or logits.shape[2] != len(tokens)
        or logits.dtype != np.float64
        or not np.isfinite(logits).all()
    ):
        raise ValueError("Expected finite float64 logits [1,T,vocabulary_size], T > 0")
    return decode_ids(np.argmax(logits[0], axis=-1), tokens, mode)
