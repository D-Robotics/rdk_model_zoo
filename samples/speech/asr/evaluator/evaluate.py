# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Score saved transcripts, independently of model or board execution."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from samples._shared.text_metrics import score_transcripts


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        raw = args.predictions.expanduser().read_bytes()
        data = json.loads(raw)
        if (
            not isinstance(data, dict)
            or data.get("schema") != "rdk-model-zoo/asr-transcripts/v1"
        ):
            raise ValueError("Expected ASR transcript schema")
        provenance = data.get("provenance")
        if not isinstance(provenance, dict) or any(
            not isinstance(provenance.get(k), str) or not provenance[k].strip()
            for k in ("dataset", "model", "decode_mode")
        ):
            raise ValueError("Provenance requires dataset/model/decode_mode strings")
        report = score_transcripts(data.get("records"))
        report.update(
            schema="rdk-model-zoo/asr-metrics/v1",
            provenance=provenance,
            predictions_sha256=hashlib.sha256(raw).hexdigest(),
            inference_executed=False,
            provenance_independently_verified=False,
        )
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
