"""Identity checks for the audited OSNet experiment's model artifacts."""

import hashlib
import json
from pathlib import Path


def load_manifest():
    return json.loads(Path(__file__).with_name("artifacts.json").read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_artifact(path, variant):
    record = load_manifest()["artifacts"][variant]
    if sha256(path) != record["sha256"]:
        raise ValueError(f"Model does not match the audited {variant} artifact: {path}")
    return record
