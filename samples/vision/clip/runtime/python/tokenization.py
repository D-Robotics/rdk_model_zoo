# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tokenizer initialization and source-compatible fixed-length token arrays."""
from pathlib import Path
import numpy as np
from samples.vision.clip.runtime.python.simple_tokenizer import SimpleTokenizer


class PromptTokenizer:
    """Load the preserved BPE vocabulary once, outside the inference task."""
    def __init__(self, bpe_path=None):
        path = Path(bpe_path) if bpe_path is not None else Path(__file__).with_name('bpe_simple_vocab_16e6.txt.gz')
        self.tokenizer = SimpleTokenizer(str(path))

    def __call__(self, texts, *, truncate=False):
        """Return I32[N,77], preserving source cleaning, SOT/EOT and overflow."""
        if not isinstance(texts, (list, tuple)) or not texts or any(not isinstance(t, str) for t in texts):
            raise ValueError('texts must be a nonempty list or tuple of strings.')
        start = self.tokenizer.encoder['<|startoftext|>']
        end = self.tokenizer.encoder['<|endoftext|>']
        result = np.zeros((len(texts), 77), dtype=np.int32)
        for row, text in enumerate(texts):
            tokens = [start, *self.tokenizer.encode(text), end]
            if len(tokens) > 77:
                if not truncate:
                    raise RuntimeError(f'Input text is too long for context length 77: {text}')
                tokens = tokens[:77]
                tokens[-1] = end
            result[row, :len(tokens)] = tokens
        return result
