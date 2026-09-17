"""Safe label-file parsing shared by the canonical entrypoint and adapters."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Mapping


def load_labels(path: Path) -> Mapping[int, str]:
    """Read an ImageNet literal mapping or one-label-per-line text file.

    ``ast.literal_eval`` preserves the legacy ImageNet dictionary format while
    avoiding execution of arbitrary Python from a label file.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"label file not found: {path}")
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return {}

    if content.startswith(("{", "[", "(")):
        try:
            parsed = ast.literal_eval(content)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"unsupported label literal in {path}; expected a dict/list "
                "or one label per line"
            ) from exc
        if isinstance(parsed, dict):
            labels: dict[int, str] = {}
            for key, value in parsed.items():
                try:
                    index = int(key)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"label dictionary key {key!r} is not an integer"
                    ) from exc
                labels[index] = str(value)
            return labels
        if isinstance(parsed, (list, tuple)):
            return {index: str(value) for index, value in enumerate(parsed)}
        raise ValueError(
            f"unsupported label literal type {type(parsed).__name__}; "
            "expected a dict/list or one label per line"
        )

    labels = {}
    for line in content.splitlines():
        text = line.strip()
        if text:
            labels[len(labels)] = text
    return labels


__all__ = ["load_labels"]
