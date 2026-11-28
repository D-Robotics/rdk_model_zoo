# Copyright (c) 2026 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wav2Vec2 float vs. BPU quantized accuracy comparison.

This script measures the accuracy loss introduced by BPU INT8 quantization
(and possibly the layout/compile transforms) by comparing the CTC logits
produced by:

1. The floating-point reference (a float ONNX model run with onnxruntime,
   or previously dumped float logits in ``.npz``), and
2. The quantized ``.bin`` model running on the BPU.

Metrics computed per audio sample:

- **cosine similarity** between flattened float and quantized logits.
  This is the standard quantization-loss metric used by the RDK toolchain;
  a value ``>= 0.95`` usually indicates an acceptable quantization loss.
- **argmax match rate** over time frames (overall and on non-blank frames).
  This directly predicts whether greedy CTC decoding will change; if it is
  1.0, the decoded text is guaranteed to be identical.
- **token-level CER** between the CTC-decoded token sequences, plus the
  decoded text of both models for human inspection.
- Numerical error statistics (MAE / max absolute error).

Two typical usage flows:

- **On the conversion PC** (no ``hbm_runtime`` available): pass only
  ``--float-onnx`` and ``--save-float-logits`` to dump the float reference
  once::

      python3 compare_accuracy.py --float-onnx ../../model/wav2vec2_base.onnx \\
          --test-audio ../../test_data/test_audio.wav \\
          --save-float-logits ../../test_data/float_logits.npz

- **On the board**: pass the dumped ``.npz`` and the compiled ``.bin``::

      python3 compare_accuracy.py \\
          --float-logits ../../test_data/float_logits.npz \\
          --bin-model ../../model/wav2vec2_base_960h_1x65535_x5.bin \\
          --test-audio ../../test_data/test_audio.wav

The board CPU can also run the float ONNX directly, but inference may be
slow for a 15 s input.
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import json
import logging
import os
from pathlib import Path
import sys

import numpy as np

# Direct entry points share the calibration audio preprocessing.
sys.path.append(str(Path(__file__).resolve().parents[5] / "utils/py_utils"))
import audio_utils
import wav2vec2

_LOGGER = logging.getLogger("Wav2Vec2_Acc")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../model"))
TEST_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../test_data"))

DEFAULT_TEST_AUDIO = os.path.join(TEST_DATA_DIR, "test_audio.wav")
DEFAULT_FLOAT_ONNX = os.path.join(MODEL_DIR, "wav2vec2_base.onnx")
DEFAULT_BIN_MODEL = os.path.join(MODEL_DIR, "wav2vec2_base_960h_1x65535_x5.bin")
DEFAULT_REPORT = os.path.join(TEST_DATA_DIR, "wav2vec2_accuracy_report.json")


def run_float_onnx(onnx_path: str, audio: np.ndarray) -> np.ndarray:
    """Run the float ONNX model with onnxruntime.

    Args:
        onnx_path: Path to the float ONNX model (opset 11 export).
        audio: Mono float32 waveform.

    Returns:
        CTC logits as a 2-D ``(T, V)`` float32 array.
    """

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError("onnxruntime is required to run the float reference. "
                          "Install it with: pip install onnxruntime") from exc

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"Float ONNX model not found: {onnx_path}")

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    shape = sess.get_inputs()[0].shape
    onnx_input = audio_utils.wav2vec2_input(audio,
                                            int(np.prod(shape))).reshape(shape)
    outputs = sess.run(None, {input_name: onnx_input})
    return as_2d(outputs[0])


def load_float_logits(npz_path: str, audio_name: str) -> np.ndarray:
    """Load previously dumped float logits for one audio sample.

    Args:
        npz_path: Path to the ``.npz`` created with ``--save-float-logits``.
        audio_name: Audio file name whose logits should be loaded.

    Returns:
        CTC logits as a 2-D ``(T, V)`` float32 array.
    """

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Float logits file not found: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as data:
        key = audio_name.rsplit(".", 1)[0]
        if key not in data:
            raise KeyError(f"Audio '{audio_name}' not found in {npz_path}. "
                           f"Available keys: {sorted(data.files)}")
        return as_2d(data[key])


def as_2d(logits: np.ndarray) -> np.ndarray:
    """Convert CTC output to a time-by-vocabulary matrix.

    Args:
        logits: Tensor with optional singleton batch or channel axes.

    Returns:
        A float32 matrix indexed by frame and token ID.

    Raises:
        ValueError: If the output cannot be interpreted as a CTC matrix.
    """

    arr = np.squeeze(logits)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected logits rank after squeeze: {arr.ndim} "
                         f"(shape {np.asarray(logits).shape})")
    return arr.astype(np.float32)


def ctc_greedy_decode(logits: np.ndarray, blank_id: int = 0) -> list[int]:
    """Collapse frame predictions and remove the CTC blank.

    Args:
        logits: Matrix indexed by frame and vocabulary entry.
        blank_id: ID to discard after collapsing consecutive repeats.

    Returns:
        Decoded token IDs in order.
    """

    token_ids = np.argmax(logits, axis=-1)
    return [
        int(t) for t, _ in itertools.groupby(token_ids) if int(t) != blank_id
    ]


def edit_distance(a: list[int], b: list[int]) -> int:
    """Compute the Levenshtein distance between token sequences.

    Args:
        a: First sequence of token IDs.
        b: Second sequence of token IDs.

    Returns:
        The minimum number of insertions, deletions and substitutions.
    """

    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1]) + 1
    return d[len(a)][len(b)]


def compute_cer(reference: list[int], hypothesis: list[int]) -> float:
    """Compute the token-level error rate.

    Args:
        reference: Expected token IDs.
        hypothesis: Predicted token IDs.

    Returns:
        Edits per reference token. An empty reference scores zero for an
        empty hypothesis and one otherwise.
    """

    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(reference, hypothesis) / len(reference)


def cosine_similarity(fp: np.ndarray, quant: np.ndarray) -> float:
    """Compute cosine similarity over all frames and vocabulary entries.

    Args:
        fp: Floating-point reference logits.
        quant: Quantized-model logits aligned to the reference.

    Returns:
        The cosine similarity, with identical zero tensors assigned one.
    """

    a = fp.astype(np.float64).ravel()
    b = quant.astype(np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.dot(a, b) / denom)


def argmax_match_rate(fp: np.ndarray, quant: np.ndarray,
                      blank_id: int) -> tuple[float, float]:
    """Measure frame-level agreement between the two models.

    Args:
        fp: Reference logits by frame and vocabulary entry.
        quant: Quantized-model logits in the same layout.
        blank_id: CTC blank ID used to select reference non-blank frames.

    Returns:
        Overall agreement and agreement on reference non-blank frames.
    """

    fp_ids = np.argmax(fp, axis=-1)
    q_ids = np.argmax(quant, axis=-1)
    agree = (fp_ids == q_ids)
    overall = float(np.mean(agree))

    non_blank = fp_ids != blank_id
    if non_blank.any():
        non_blank_rate = float(np.mean(agree[non_blank]))
    else:
        non_blank_rate = 1.0
    return overall, non_blank_rate


def align_frames(fp: np.ndarray,
                 quant: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Truncate both CTC outputs to their common frame count.

    Args:
        fp: Reference time-by-vocabulary matrix.
        quant: Quantized-model time-by-vocabulary matrix.

    Returns:
        The two matrices restricted to matching frames.

    Raises:
        ValueError: If the vocabulary sizes differ.
    """

    t_min = min(fp.shape[0], quant.shape[0])
    if fp.shape[1] != quant.shape[1]:
        raise ValueError(
            f"Vocabulary size mismatch: float {fp.shape[1]} vs "
            f"quantized {quant.shape[1]}. The float ONNX and the .bin model "
            f"do not match.")
    return fp[:t_min], quant[:t_min]


def compare_one(fp_logits: np.ndarray,
                quant_logits: np.ndarray,
                blank_id: int,
                audio_name: str,
                vocab=None) -> dict:
    """Compare reference and quantized CTC outputs for one recording.

    Args:
        fp_logits: Reference logits by frame and token.
        quant_logits: Quantized-model logits by frame and token.
        blank_id: CTC blank token ID.
        audio_name: Recording name to include in the report.
        vocab: Optional ID-to-token mapping for transcription.

    Returns:
        Per-file metrics, token IDs and transcripts for both models.
    """

    fp, quant = align_frames(fp_logits, quant_logits)

    cos = cosine_similarity(fp, quant)
    overall_match, non_blank_match = argmax_match_rate(fp, quant, blank_id)

    fp_ids = ctc_greedy_decode(fp, blank_id)
    q_ids = ctc_greedy_decode(quant, blank_id)
    cer = compute_cer(fp_ids, q_ids)

    diff = fp - quant
    return {
        "file": audio_name,
        "float_shape": list(fp.shape),
        "quant_shape": list(quant.shape),
        "cosine_similarity": round(cos, 6),
        "argmax_match_rate_overall": round(overall_match, 6),
        "argmax_match_rate_non_blank": round(non_blank_match, 6),
        "token_cer": round(cer, 6),
        "text_identical": fp_ids == q_ids,
        "mae": float(np.mean(np.abs(diff))),
        "max_abs_error": float(np.max(np.abs(diff))),
        "float_token_ids": fp_ids,
        "quant_token_ids": q_ids,
        "float_text": wav2vec2.token_ids_to_text(fp_ids, vocab),
        "quant_text": wav2vec2.token_ids_to_text(q_ids, vocab),
    }


def print_one(result: dict) -> None:
    """Log accuracy metrics and both transcripts for one recording.

    Args:
        result: Per-file metrics returned by compare_one().
    """

    _LOGGER.info("=" * 64)
    _LOGGER.info("sample: %s", result["file"])
    _LOGGER.info("float logits shape %s | quant logits shape %s",
                 result["float_shape"], result["quant_shape"])
    _LOGGER.info("cosine similarity     = %.6f", result["cosine_similarity"])
    _LOGGER.info("argmax match rate     = %.4f (overall) / %.4f (non-blank)",
                 result["argmax_match_rate_overall"],
                 result["argmax_match_rate_non_blank"])
    _LOGGER.info("token CER             = %.4f%s", result["token_cer"],
                 " (text identical)" if result["text_identical"] else "")
    _LOGGER.info("MAE = %.6f | max abs error = %.6f", result["mae"],
                 result["max_abs_error"])
    _LOGGER.info("float text : %r", result["float_text"])
    _LOGGER.info("quant text : %r", result["quant_text"])


def print_summary(results: list[dict], cos_threshold: float,
                  report_path: str | None) -> None:
    """Log aggregate metrics and optionally write a JSON report.

    Args:
        results: Per-file metrics for at least one recording.
        cos_threshold: Minimum mean cosine similarity for a passing verdict.
        report_path: JSON output path, or None to skip writing the report.
    """

    n = len(results)
    avg_cos = sum(r["cosine_similarity"] for r in results) / n
    avg_match = sum(r["argmax_match_rate_non_blank"] for r in results) / n
    avg_cer = sum(r["token_cer"] for r in results) / n
    identical = sum(1 for r in results if r["text_identical"])

    passed = avg_cos >= cos_threshold
    _LOGGER.info("=" * 64)
    _LOGGER.info("SUMMARY over %d sample(s)", n)
    _LOGGER.info("avg cosine similarity      = %.6f", avg_cos)
    _LOGGER.info("avg non-blank argmax match = %.6f", avg_match)
    _LOGGER.info("avg token CER              = %.6f", avg_cer)
    _LOGGER.info("text-identical samples     = %d/%d", identical, n)
    verdict = "PASS" if passed else "FAIL"
    _LOGGER.info("verdict: %s (avg cosine %.6f %s threshold %.4f)", verdict,
                 avg_cos, ">=" if passed else "<", cos_threshold)

    if not passed:
        _LOGGER.warning(
            "Quantization loss may be too large. Consider re-checking the "
            "calibration data, compile options, or the exported ONNX.")

    if report_path:
        summary = {
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_samples": n,
            "avg_cosine_similarity": round(avg_cos, 6),
            "avg_non_blank_argmax_match": round(avg_match, 6),
            "avg_token_cer": round(avg_cer, 6),
            "text_identical": f"{identical}/{n}",
            "verdict": verdict,
            "cos_threshold": cos_threshold,
            "results": results,
        }
        report_dir = os.path.dirname(report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        _LOGGER.info("Detailed report saved to %s", report_path)


def main() -> None:
    """Compare model outputs or save reference logits for a later comparison."""
    logging.basicConfig(
        level=logging.INFO,
        format=("[%(name)s] [%(asctime)s.%(msecs)03d] "
                "[%(levelname)s] %(message)s"),
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Compare float vs BPU-quantized Wav2Vec2 accuracy.")
    parser.add_argument("--float-onnx",
                        type=str,
                        default=None,
                        help="Path to the float ONNX model (opset 11). "
                        "Ignored when --float-logits is given.")
    parser.add_argument("--float-logits",
                        type=str,
                        default=None,
                        help="Path to a .npz file with dumped float logits "
                        "(faster than running onnxruntime on the board).")
    parser.add_argument("--bin-model",
                        type=str,
                        default=None,
                        help="Path to the BPU quantized .bin model. If "
                        "omitted, only float logits are produced/saved.")
    parser.add_argument("--test-audio",
                        type=str,
                        default=DEFAULT_TEST_AUDIO,
                        help="Single audio file to test.")
    parser.add_argument("--audio-dir",
                        type=str,
                        default=None,
                        help="Directory of audio files (overrides "
                        "--test-audio, results are averaged).")
    parser.add_argument("--save-float-logits",
                        type=str,
                        default=None,
                        help="Save float logits to this .npz for later "
                        "board-side comparison.")
    parser.add_argument("--blank-id",
                        type=int,
                        default=0,
                        help="CTC blank token ID.")
    parser.add_argument("--cos-threshold",
                        type=float,
                        default=0.95,
                        help="Cosine similarity threshold for PASS verdict.")
    parser.add_argument("--report-path",
                        type=str,
                        default=DEFAULT_REPORT,
                        help="Path to save the JSON report.")
    parser.add_argument("--priority",
                        type=int,
                        default=0,
                        help="BPU model scheduling priority (0~255).")
    parser.add_argument("--bpu-cores",
                        nargs="+",
                        type=int,
                        default=[0],
                        help="BPU core indices.")
    parser.add_argument("--vocab-path",
                        type=str,
                        default=None,
                        help="Custom token-to-ID JSON for the CTC checkpoint.")
    args = parser.parse_args()

    if not args.float_onnx and not args.float_logits:
        parser.error("either --float-onnx or --float-logits must be provided")

    # Collect audio files
    if args.audio_dir:
        if not os.path.isdir(args.audio_dir):
            parser.error(f"audio directory not found: {args.audio_dir}")
        extensions = (".wav", ".mp3", ".flac")
        audio_files = sorted(
            os.path.join(args.audio_dir, f)
            for f in os.listdir(args.audio_dir)
            if f.lower().endswith(extensions))
        if not audio_files:
            parser.error(f"no audio files found in {args.audio_dir}")
    else:
        if not os.path.exists(args.test_audio):
            parser.error(f"audio file not found: {args.test_audio}")
        audio_files = [args.test_audio]

    # Quantized BPU model (optional: skipped for float-only dump on PC)
    model = None
    if args.bin_model:
        if not os.path.exists(args.bin_model):
            parser.error(f"bin model not found: {args.bin_model}")
        _LOGGER.info("Loading BPU quantized model: %s", args.bin_model)
        model = wav2vec2.Wav2Vec2(
            wav2vec2.Wav2Vec2Config(
                model_path=args.bin_model,
                blank_id=args.blank_id,
                vocab_path=args.vocab_path,
            ))
        model.set_scheduling_params(priority=args.priority,
                                    bpu_cores=args.bpu_cores)

    saved_logits: dict[str, np.ndarray] = {}
    results: list[dict] = []

    for audio_path in audio_files:
        audio_name = os.path.basename(audio_path)
        _LOGGER.info("Processing audio: %s", audio_path)
        audio = audio_utils.load_waveform(audio_path)

        # Float reference
        if args.float_logits:
            fp_logits = load_float_logits(args.float_logits, audio_name)
        else:
            fp_logits = run_float_onnx(args.float_onnx, audio)

        if args.save_float_logits:
            saved_logits[audio_name.rsplit(".", 1)[0]] = fp_logits

        if model is None:
            _LOGGER.info("No --bin-model given; skipped quantized inference.")
            continue

        # Quantized BPU inference (same preprocessing as main.py)
        outputs = model.forward(model.pre_process(audio))
        quant_logits = as_2d(outputs[model.model_name][model.output_names[0]])

        result = compare_one(fp_logits, quant_logits, args.blank_id, audio_name,
                             model.vocab)
        print_one(result)
        results.append(result)

    if args.save_float_logits and saved_logits:
        save_dir = os.path.dirname(args.save_float_logits)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(args.save_float_logits, **saved_logits)
        _LOGGER.info("Float logits saved to %s", args.save_float_logits)

    if results:
        print_summary(results, args.cos_threshold, args.report_path)


if __name__ == "__main__":
    main()
