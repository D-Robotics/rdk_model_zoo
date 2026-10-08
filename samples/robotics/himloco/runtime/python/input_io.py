"""Source-indexed observation files and provenance, separate from policy math.

NumPy loads lazily inside :func:`load_observation` so importing this module
(the manifest validation path) stays NumPy-free for host listing/dry-run.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class InputRecord:
    source_index: int
    path: Path
    expected_sha256: str | None


def discover_inputs(path):
    path = Path(path).resolve()
    files = sorted(path.glob("*.bin")) if path.is_dir() else [path]
    if not files:
        raise ValueError("No observation BIN files found")
    indexed = {}
    for file in files:
        if (
            not file.is_file()
            or file.suffix != ".bin"
            or not file.stem.isascii()
            or not file.stem.isdecimal()
        ):
            raise ValueError(f"Expected a numerically named observation BIN: {file}")
        index = int(file.stem)
        if index in indexed:
            raise ValueError(f"Duplicate source index {index}")
        indexed[index] = file
    directory = path if path.is_dir() else path.parent
    manifest_path = directory.parent / "runtime-input-manifest.json"
    manifest_info = None
    records = {}
    if manifest_path.is_file():
        data = manifest_path.read_bytes()
        manifest = json.loads(data)
        contract = manifest.get("input_contract", {})
        if any(
            contract.get(k) != v
            for k, v in {
                "name": "obs_history",
                "shape": [1, 270],
                "dtype": "float32",
                "bytes_per_file": 1080,
            }.items()
        ):
            raise ValueError("Observation manifest physical contract mismatch")
        entries = manifest.get("records")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Manifest requires nonempty records")
        indices = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Invalid manifest record")
            name = entry.get("file")
            index = entry.get("source_index")
            digest = entry.get("sha256")
            if (
                not isinstance(name, str)
                or Path(name).is_absolute()
                or ".." in Path(name).parts
                or type(index) is not int
                or index < 0
                or index in indices
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
                or entry.get("bytes") != 1080
            ):
                raise ValueError("Invalid or duplicate manifest identity/digest")
            location = (manifest_path.parent / name).resolve()
            if (
                not location.is_relative_to(manifest_path.parent.resolve())
                or location in records
            ):
                raise ValueError(
                    "Manifest paths must be unique and within its directory"
                )
            indices.add(index)
            records[location] = (index, digest)
        manifest_info = {
            "path": str(manifest_path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "source": manifest.get("source"),
            "source_sha256": manifest.get("source_sha256"),
        }
    result = []
    for index, file in sorted(indexed.items()):
        expected = None
        if manifest_info is not None:
            if file not in records or records[file][0] != index:
                raise ValueError(f"Input identity not in manifest: {file}")
            expected = records[file][1]
        result.append(InputRecord(index, file, expected))
    return tuple(result), manifest_info


def load_observation(record):
    import numpy as np

    data = record.path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != 1080:
        raise ValueError("Observation must contain exactly 1080 bytes")
    if record.expected_sha256 is not None and digest != record.expected_sha256:
        raise ValueError(f"Observation digest mismatch: {record.path}")
    values = np.frombuffer(data, dtype="<f4").astype(np.float32).reshape(1, 270)
    if not np.isfinite(values).all():
        raise ValueError("Observation contains NaN/Inf")
    return values, digest
