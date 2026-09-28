"""Audio/manifest file handling, kept outside feature and inference math."""

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class InputItem:
    entry: dict
    audio_path: Path


def validate_id(value):
    if (
        not isinstance(value, str)
        or not value
        or value in (".", "..")
        or value.strip() != value
        or any(char in value for char in ("/", "\\", "\0"))
    ):
        raise ValueError("utt_id must be a nonempty filename stem, not a path")
    return value


def load_manifest(path, audio_dir, max_utts=0):
    if type(max_utts) is not int or max_utts < 0:
        raise ValueError("max-utts must be nonnegative; zero means all")
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest must be a nonempty JSON list")
    seen = set()
    items = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each manifest entry must be an object")
        key = validate_id(record.get("utt_id"))
        if key in seen or ("text" in record and not isinstance(record["text"], str)):
            raise ValueError(
                "Manifest IDs must be unique and reference text must be a string"
            )
        seen.add(key)
        items.append(InputItem(dict(record), Path(audio_dir) / f"{key}.wav"))
    selected = items[:max_utts] if max_utts else items
    for item in selected:
        if not item.audio_path.is_file():
            raise ValueError(f"Missing selected WAV: {item.audio_path}")
    return tuple(selected)


def read_audio(path):
    import soundfile as sf

    return sf.read(Path(path), dtype="float32")


def write_json(path, payload):
    """Atomically finish JSON inside a newly created per-run directory."""
    path = Path(path)
    temporary = path.with_name(path.name + ".part")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
