"""Prepare real-WAV calibration and three nash-e compiler configs; never compile implicitly."""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
from importlib.metadata import version
from pathlib import Path
import shutil
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from utils.py_utils.assets import sha256_file
from samples.speech.paraformer.runtime.python.input_io import write_json
from samples.speech.paraformer.conversion.calibration import (
    CALIBRATION,
    select_wavs,
    intermediates,
)
from samples.speech.paraformer.conversion.configuration import STAGES, make_config
from samples.speech.paraformer.conversion.export import check_signature

SAMPLE = Path(__file__).resolve().parents[1]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument(
        "--wav-dir",
        type=Path,
        required=True,
        help="Recursively scanned real 16 kHz WAVs",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New preparation workspace"
    )
    parser.add_argument("--cmvn-file", type=Path, default=SAMPLE / "model/am.mvn")
    parser.add_argument(
        "--sample-count", type=int, default=50, help="Sorted WAV prefix (default: 50)"
    )
    parser.add_argument(
        "--threads", type=int, default=4, help="ONNX CPU threads (default: 4)"
    )
    parser.add_argument(
        "--jobs", type=int, default=32, help="Recorded OE compiler jobs (default: 32)"
    )
    return parser


def inspect_export(directory):
    """Hash-bound completed exporter output; inspect the actual embedded graphs."""
    import onnx

    directory = Path(directory).expanduser().resolve()
    data = (directory / "export-report.json").read_bytes()
    report = json.loads(data)
    if (
        report.get("schema") != "rdk-model-zoo/paraformer-export/v1"
        or report.get("status") != "completed"
        or set(report.get("stages", {})) != set(STAGES)
    ):
        raise ValueError("Expected a completed three-stage export report")
    models = {}
    for stage in STAGES:
        source = directory / f"{stage}.onnx"
        recorded = report["stages"][stage]
        digest = sha256_file(source)
        if recorded.get("path") != source.name or recorded.get("sha256") != digest:
            raise ValueError(f"Export graph digest/name mismatch: {stage}")
        graph = onnx.load(source, load_external_data=False)
        if any(
            t.data_location == onnx.TensorProto.EXTERNAL
            for t in graph.graph.initializer
        ):
            raise ValueError(
                "Expected self-contained exported graph, not external tensor files"
            )
        onnx.checker.check_model(graph)
        check_signature(graph, stage)
        models[stage] = {"source": str(source), "sha256": digest}
        del graph
        gc.collect()
    return models, data


def prepare(args):
    import numpy as np
    import yaml
    import soundfile as sf
    from samples.speech.paraformer.runtime.python.frontend import ParaformerFrontend
    from samples.speech.paraformer.conversion.onnx_stage import OnnxStage

    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise ValueError("Output workspace must be new")
    if args.threads < 1 or args.jobs < 1:
        raise ValueError("threads and jobs must be positive")
    paths = select_wavs(args.wav_dir, args.sample_count)
    models, export_bytes = inspect_export(args.export_dir)
    frontend = ParaformerFrontend(args.cmvn_file)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "rdk-model-zoo/paraformer-calibration/v1",
        "status": "preparing",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "target": "s100",
        "march": "nash-e",
        "compiler": "not-run",
        "board": "not-run",
        "dataset_accuracy": "not-run",
        "sample_count_requested": args.sample_count,
        "sample_count_selected": len(paths),
        "random_seed": frontend.random_seed,
        "environment": {
            name: version(name)
            for name in ("numpy", "torch", "funasr", "onnxruntime", "soundfile")
        },
        "threads": args.threads,
        "jobs": args.jobs,
        "cif_valid_frame_mask": False,
        "cmvn_sha256": sha256_file(args.cmvn_file),
        "export_report_sha256": hashlib.sha256(export_bytes).hexdigest(),
        "models": models,
        "records": [],
        "configs": {},
        "warnings": (
            ["Fewer than source reference 50 WAVs; no calibration quality acceptance"]
            if len(paths) < 50
            else []
        ),
    }
    try:
        (output / "source").mkdir()
        (output / "source/export-report.json").write_bytes(export_bytes)
        shutil.copyfile(args.cmvn_file, output / "source/am.mvn")
        if sha256_file(output / "source/am.mvn") != report["cmvn_sha256"]:
            raise ValueError("CMVN changed while copying")
        for stage, model in models.items():
            snapshot = output / f"source/{stage}.onnx"
            shutil.copyfile(model["source"], snapshot)
            if sha256_file(snapshot) != model["sha256"]:
                raise ValueError(f"Model changed while copying: {stage}")
        runners = {
            stage: OnnxStage(output / f"source/{stage}.onnx", stage, args.threads)
            for stage in ("encoder", "predictor")
        }
        for name in CALIBRATION:
            (output / "calibration" / name).mkdir(parents=True)
        for index, path in enumerate(paths):
            report["current_audio"] = str(path)
            data = path.read_bytes()
            audio, rate = sf.read(io.BytesIO(data), dtype="float32")
            prepared = frontend.pre_process(audio, rate)
            arrays = intermediates(
                prepared.tensor,
                runners["encoder"].forward,
                runners["predictor"].forward,
            )
            filename = f"{index:06d}.npy"
            record = {
                "audio_path": str(path),
                "audio_sha256": hashlib.sha256(data).hexdigest(),
                "sample_rate": rate,
                "sample_count": prepared.sample_count,
                "valid_frames": prepared.valid_frames,
                "original_frames": prepared.original_frames,
                "truncated": prepared.truncated,
                "filename": filename,
                "arrays": {},
            }
            for name, array in arrays.items():
                destination = output / "calibration" / name / filename
                np.save(destination, array, allow_pickle=False)
                record["arrays"][name] = {
                    "sha256": sha256_file(destination),
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "min": float(array.min()),
                    "max": float(array.max()),
                }
            report["records"].append(record)
            write_json(output / "preparation.json", report)
        (output / "configs").mkdir()
        for stage in STAGES:
            path = output / "configs" / f"{stage}.yaml"
            path.write_text(
                yaml.safe_dump(make_config(stage, args.jobs), sort_keys=False),
                encoding="utf-8",
            )
            report["configs"][stage] = sha256_file(path)
        for stage, model in models.items():
            if sha256_file(output / f"source/{stage}.onnx") != model["sha256"]:
                raise ValueError(f"Snapshot changed during calibration: {stage}")
        report.pop("current_audio", None)
        report["status"] = "prepared"
    except Exception as error:
        report.update(status="preparation_failed", error=str(error))
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "preparation.json", report)
    return report


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        report = prepare(args)
    except Exception as error:
        print(f"Preparation failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "samples": len(report["records"]),
                "compiler": "not-run",
                "output": str(args.output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
