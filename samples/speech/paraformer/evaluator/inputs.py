"""Strict feature manifests for evaluation; hashes cover the bytes actually read."""

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re

import numpy as np

from samples.speech.paraformer.runtime.python.input_io import validate_id


@dataclass(frozen=True)
class FeatureEntry:
    source: dict
    path: Path
    expected_digest: str | None


def read_manifest(path, max_utts=0):
    """Validate every record before selecting a prefix; zero selects all records."""
    if type(max_utts) is not int or max_utts < 0:
        raise ValueError("max-utts must be nonnegative; zero means all")
    path = Path(path).resolve()
    raw = path.read_bytes()
    records = json.loads(raw)
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest must be a nonempty JSON list")
    entries, seen = [], set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each manifest entry must be an object")
        key = validate_id(record.get("utt_id"))
        if key in seen:
            raise ValueError(f"Duplicate utt_id: {key}")
        seen.add(key)
        if not isinstance(record.get("text"), str):
            raise ValueError(f"{key}: reference text must be a string")
        length = record.get("feat_length")
        if type(length) is not int or not 1 <= length <= 400:
            raise ValueError(f"{key}: feat_length must be an integer in [1, 400]")
        if "original_frames" in record:
            original = record["original_frames"]
            if (
                type(original) is not int
                or original < 1
                or min(original, 400) != length
            ):
                raise ValueError(f"{key}: original_frames contradicts feat_length")
        if "truncated" in record:
            truncated = record["truncated"]
            if (
                type(truncated) is not bool
                or (
                    "original_frames" in record
                    and truncated != (record["original_frames"] > 400)
                )
                or (truncated and length != 400)
            ):
                raise ValueError(f"{key}: invalid truncation declaration")
        filename = record.get("feature_file", f"feats/{key}.npy")
        if not isinstance(filename, str) or not filename.strip() or "\0" in filename:
            raise ValueError(f"{key}: feature_file must be a nonempty path")
        feature_path = Path(filename)
        if not feature_path.is_absolute():
            feature_path = path.parent / feature_path
        digest = record.get("feature_sha256")
        if "feature_sha256" in record:
            if not isinstance(digest, str) or not re.fullmatch(
                r"[0-9a-fA-F]{64}", digest
            ):
                raise ValueError(f"{key}: invalid feature_sha256")
            digest = digest.lower()
        entries.append(FeatureEntry(dict(record), feature_path.resolve(), digest))
    selected = entries[:max_utts] if max_utts else entries
    return tuple(selected), hashlib.sha256(raw).hexdigest()


def load_feature(entry):
    """Read and hash once, then reject malformed NPY without coercion or pickle."""
    raw = entry.path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if entry.expected_digest is not None and digest != entry.expected_digest:
        raise ValueError(f"{entry.source['utt_id']}: feature digest mismatch")
    with BytesIO(raw) as stream:
        value = np.load(stream, allow_pickle=False)
        if not isinstance(value, np.ndarray):
            value.close()
            raise ValueError("Feature must be a single NPY array")
        if stream.tell() != len(raw):
            raise ValueError("Feature contains trailing data")
    if (
        value.shape != (1, 400, 560)
        or value.dtype != np.dtype("float32")
        or not np.isfinite(value).all()
    ):
        raise ValueError("Feature must be finite float32 [1,400,560]")
    return np.array(value, copy=True), digest
