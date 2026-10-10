"""Validate a completed calibration workspace before external compiler use."""

import hashlib
import io
import json
from pathlib import Path
import numpy as np
import yaml

from utils.py_utils.assets import sha256_file
from samples.speech.paraformer.conversion.calibration import CALIBRATION, checked_tensor
from samples.speech.paraformer.conversion.configuration import STAGES, make_config
from samples.speech.paraformer.runtime.python.cli import LOCAL_DIGESTS


def verify_prepared(root):
    root = Path(root).expanduser().resolve()
    raw = (root / "preparation.json").read_bytes()
    report = json.loads(raw)
    if (
        report.get("schema") != "rdk-model-zoo/paraformer-calibration/v1"
        or report.get("status") != "prepared"
        or report.get("target") != "s100"
        or report.get("march") != "nash-e"
        or report.get("cif_valid_frame_mask") is not False
    ):
        raise ValueError("Expected a completed S100 calibration preparation")
    records = report.get("records")
    if (
        not isinstance(records, list)
        or not records
        or report.get("sample_count_selected") != len(records)
    ):
        raise ValueError("Expected the complete nonempty calibration record set")
    if (
        report.get("cmvn_sha256") != LOCAL_DIGESTS["am.mvn"]
        or sha256_file(root / "source/am.mvn") != report["cmvn_sha256"]
        or sha256_file(root / "source/export-report.json")
        != report.get("export_report_sha256")
    ):
        raise ValueError("Source support snapshot identity mismatch")
    expected_files = {f"{index:06d}.npy" for index in range(len(records))}
    for name in CALIBRATION:
        entries = list((root / "calibration" / name).iterdir())
        if {p.name for p in entries} != expected_files or any(
            not p.is_file() for p in entries
        ):
            raise ValueError(f"Unexpected calibration files for {name}")
    for index, record in enumerate(records):
        filename = f"{index:06d}.npy"
        if record.get("filename") != filename or set(record.get("arrays", {})) != set(
            CALIBRATION
        ):
            raise ValueError("Calibration ordering or tensor set mismatch")
        for name, (shape, dtype) in CALIBRATION.items():
            entry = record["arrays"][name]
            data = (root / "calibration" / name / filename).read_bytes()
            if (
                entry.get("shape") != list(shape)
                or entry.get("dtype") != dtype
                or hashlib.sha256(data).hexdigest() != entry.get("sha256")
            ):
                raise ValueError(f"Calibration identity mismatch: {name}/{filename}")
            array = np.load(io.BytesIO(data), allow_pickle=False)
            checked_tensor(array, name)
    configs = {}
    for stage in STAGES:
        if (
            sha256_file(root / f"source/{stage}.onnx")
            != report["models"][stage]["sha256"]
        ):
            raise ValueError(f"Stage snapshot mismatch: {stage}")
        path = root / f"configs/{stage}.yaml"
        data = path.read_bytes()
        config = yaml.safe_load(data)
        if hashlib.sha256(data).hexdigest() != report["configs"][
            stage
        ] or config != make_config(stage, report.get("jobs")):
            raise ValueError(f"Compiler config differs from source contract: {stage}")
        configs[stage] = config
    return report, hashlib.sha256(raw).hexdigest(), configs
