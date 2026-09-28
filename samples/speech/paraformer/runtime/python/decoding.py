"""Paraformer greedy token text; deliberately not a CTC decoder."""

import numpy as np


def validate_vocabulary(vocabulary):
    """Freeze the ordered vocabulary required by the 8404-class decoder."""
    if (
        not isinstance(vocabulary, (list, tuple))
        or len(vocabulary) != 8404
        or any(not isinstance(token, str) or not token for token in vocabulary)
        or len(set(vocabulary)) != 8404
    ):
        raise ValueError("Expected 8404 unique nonempty ordered vocabulary tokens")
    return tuple(vocabulary)


def decode_logits(logits, token_count, vocabulary):
    """Return text and all selected IDs, including filtered special-token IDs."""
    tokens = validate_vocabulary(vocabulary)
    if (
        not isinstance(logits, np.ndarray)
        or logits.shape != (1, 100, 8404)
        or logits.dtype != np.float32
        or not np.isfinite(logits).all()
    ):
        raise ValueError("Expected finite float32 decoder logits [1,100,8404]")
    if type(token_count) is not int or not 0 <= token_count <= 100:
        raise ValueError("token_count must be an integer in [0,100]")
    ids = tuple(int(i) for i in np.argmax(logits[0, :token_count], axis=-1))
    words = [tokens[i] for i in ids]
    text = "".join(
        token.replace("@@", "")
        for token in words
        if not (token.startswith("<") and token.endswith(">"))
    )
    return text, ids
