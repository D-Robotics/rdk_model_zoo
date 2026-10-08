"""Read classification label files and validate class-index coverage.

Supports full class-name sequences and sparse class-index mappings without
importing numerical libraries or a board SDK.
"""


from __future__ import annotations

import ast
from pathlib import Path
from numbers import Integral
from typing import Mapping, Sequence


def load_labels(path: Path) -> Mapping[int, str]:
    """Read a literal mapping/sequence or one-label-per-line UTF-8 file.

    Args:
        path: Local label-file path. Dictionary literals, lists, and tuples are
            parsed with ast.literal_eval; no Python code is executed.

    Returns:
        Mapping[int, str]: Integer class IDs mapped to string names. Sequence
        items and nonempty plain-text lines receive consecutive IDs from zero.
        An empty file produces an empty mapping.

    Raises:
        FileNotFoundError: The path is not a file.
        OSError: The file cannot be read.
        UnicodeError: The contents are not valid UTF-8.
        ValueError: Literal syntax, container type, or a dictionary key is invalid.
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


def validate_labels(
    labels: Mapping[int, str] | Sequence[str] | None, class_count: int
) -> Mapping[int, str] | Sequence[str] | None:
    """Validate label indexes or sequence length against a class count.

    Args:
        labels: None, a sparse mapping with integral keys in [0, class_count),
            or a sequence containing exactly class_count names. Strings and
            bytes are not accepted as sequences of names.
        class_count: Positive number of output classes, validated by the caller.

    Returns:
        Mapping[int, str] | Sequence[str] | None: The original labels object,
        unchanged. Name values are not converted or validated here.

    Raises:
        ValueError: A mapping key is invalid or sequence length differs from class_count.
        TypeError: Labels are neither None, a mapping, nor a supported sequence.
    """

    if labels is None:
        return None
    if isinstance(labels, Mapping):
        for key in labels:
            if not isinstance(key, Integral) or not 0 <= int(key) < class_count:
                raise ValueError(
                    f"Label key {key!r} is not a valid class index for the bound "
                    f"{class_count}-class output.")
        return labels
    if isinstance(labels, Sequence) and not isinstance(labels, (str, bytes)):
        if len(labels) != class_count:
            raise ValueError(
                f"{len(labels)} labels do not match the bound {class_count}-class "
                "output; labels must cover every class exactly.")
        return labels
    raise TypeError(
        "labels must be a mapping of class index to name, or a sequence of names.")


__all__ = ["load_labels", "validate_labels"]
