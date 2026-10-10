# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Check the bundled audio transcript against its independently published prefix.

The screenshot prints only the prefix. This check cannot establish a complete
reference transcript or dataset CER. Optional second-model text is a crosscheck,
not ground truth, and has no effect on the published-prefix pass criterion.
"""
import argparse
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SAMPLE = Path(__file__).resolve().parents[1]
PUBLISHED_PREFIX = "我是来自阿里云的大规模"


def evaluate_report(report, audio_sha256):
    """Return preset prefix/completion checks for one saved native ASR report.

    Args:
        report: Native asr-run/v1 mapping with full text and chunk records.
        audio_sha256: Exact expected bundled audio digest.

    Returns:
        dict: Individual checks, pass status, and declared prefix-only scope.

    Raises:
        ValueError: Native report schema or text/chunk structure is invalid.
    """
    if not isinstance(report, dict) or report.get("schema") != "rdk-model-zoo/asr-run/v1":
        raise ValueError("Expected native ASR run report")
    text = report.get("text")
    chunks = report.get("chunks")
    if not isinstance(text, str) or not isinstance(chunks, list) or not chunks:
        raise ValueError("Expected text and nonempty independently decoded chunks")
    if any(not isinstance(chunk, dict) or not isinstance(chunk.get("text"), str) for chunk in chunks):
        raise ValueError("Each chunk must contain text")
    checks = {"completed": report.get("status") == "completed",
              "bundled_audio": report.get("audio_sha256") == audio_sha256,
              "chunk_text_consistent": text == "".join(chunk["text"] for chunk in chunks),
              "published_prefix": "".join(text.split()).startswith(PUBLISHED_PREFIX),
              "word_delimiter_decoded": "|" not in text or report.get("decode_mode") != "ctc",
              "no_special_tokens": not any(token in text for token in ("<pad>", "<s>", "</s>", "<unk>"))}
    return {"passed": all(checks.values()), "checks": checks, "text": text,
            "published_prefix": PUBLISHED_PREFIX,
            "scope": "basic transcription prefix and decoder protocol; remaining characters unverified; no dataset accuracy"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    """Score a saved report and return zero/pass or one/fail.

    Args:
        argv: Optional CLI arguments; None reads sys.argv.

    Returns:
        int: Zero when the declared prefix and decoder protocol checks pass.

    Raises:
        ValueError: Run-report content is invalid.
        OSError: Input or output files cannot be read or written.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-report", type=Path, required=True)
    parser.add_argument("--crosscheck-text-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate_report(json.loads(args.run_report.read_text()), digest(SAMPLE / "test_data/chi_sound.wav"))
    result.update(schema="rdk-model-zoo/asr-basic-check/v1", run_report_sha256=digest(args.run_report),
                  prefix_source={"path": str(SAMPLE / "test_data/readme_img/print.jpg"),
                                 "sha256": digest(SAMPLE / "test_data/readme_img/print.jpg"),
                                 "method": "Codex visual transcription of published screenshot before inference"})
    if args.crosscheck_text_file:
        text = args.crosscheck_text_file.read_text().strip()
        result["independent_model_crosscheck"] = {"text": text, "sha256": digest(args.crosscheck_text_file),
            "sequence_match_ratio": SequenceMatcher(None, "".join(result["text"].split()), "".join(text.split())).ratio(),
            "ground_truth": False, "affects_pass_criterion": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
