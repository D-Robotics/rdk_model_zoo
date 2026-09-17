# Copyright (c) 2026 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure, finite post-processing for the PaddleOCR CTC recognizers.

The two audited recognizers expose different output widths, but both use the
same best-path CTC rule.  This module deliberately has no runtime or image
processing imports so it is usable in host tests and in ``--help`` paths.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def ctc_decode_indices(
    indices: np.ndarray,
    tokens: Sequence[str],
    *,
    raw: bool = False,
    blank_symbol: str = "",
) -> str:
    """Decode an already-argmaxed CTC index sequence.

    This is the shared primitive used by the canonical recognizer and the
    legacy compatibility converter.  ``raw=True`` intentionally keeps every
    class, including blanks, for callers that expose the historical diagnostic
    string; normal decoding applies the usual blank and repeat collapse.
    """

    values = np.asarray(indices)
    if values.ndim != 1:
        raise ValueError(f"CTC indices must be one-dimensional, got {values.shape}.")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"CTC indices must be integers, got {values.dtype}.")
    token_table = tuple(tokens)
    if not token_table or token_table[0] != "blank":
        raise ValueError("CTC token table must contain 'blank' at class index 0.")
    if any(not isinstance(token, str) for token in token_table):
        raise ValueError("CTC token table entries must be strings.")

    result: list[str] = []
    previous = -1
    for value in values:
        index = int(value)
        if index < 0 or index >= len(token_table):
            raise ValueError(
                f"CTC class index {index} is outside the token table of length "
                f"{len(token_table)}."
            )
        if raw:
            result.append(blank_symbol if index == 0 else token_table[index])
        elif index != 0 and index != previous:
            result.append(token_table[index])
        previous = index
    return "".join(result)


def ctc_greedy_decode(scores: np.ndarray, tokens: Sequence[str]) -> str:
    """Decode one CTC score tensor using the source-compatible best path.

    ``scores`` may be ``[T,V]`` or ``[1,T,V]``.  Class zero is the blank.  A
    blank resets the repeated-token state, matching both audited Python
    wrappers; no softmax or other activation is inserted.

    Args:
        scores: Float32 score/logit tensor with one sequence.
        tokens: Token strings indexed by the final class dimension.

    Returns:
        The decoded Unicode text.

    Raises:
        ValueError: If rank, class count, dtype, or finite-value contracts are
            not satisfied.
    """

    values = np.asarray(scores)
    if values.ndim == 3:
        if values.shape[0] != 1:
            raise ValueError(f"CTC decoding accepts one batch, got {values.shape}.")
        values = values[0]
    if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] <= 0:
        raise ValueError(f"CTC scores must have shape (T,V) or (1,T,V), got {values.shape}.")
    if values.dtype != np.dtype("float32"):
        raise ValueError(f"CTC scores must be float32, got {values.dtype}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("CTC scores contain NaN or infinity.")

    token_table = tuple(tokens)
    if len(token_table) != values.shape[1]:
        raise ValueError(
            f"CTC class count {values.shape[1]} does not match {len(token_table)} tokens."
        )
    if not token_table or token_table[0] != "blank":
        raise ValueError("CTC token table must contain 'blank' at class index 0.")
    if any(not isinstance(token, str) for token in token_table):
        raise ValueError("CTC token table entries must be strings.")

    indices = np.argmax(values, axis=1)
    return ctc_decode_indices(indices, token_table)


__all__ = ["ctc_decode_indices", "ctc_greedy_decode"]
