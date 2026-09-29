"""Validate a native report against selected artifacts and prepared inputs."""

import math
from pathlib import Path

from samples.speech.paraformer.runtime.python.model_binding import (
    bind_model,
    VOCABULARY_DIGEST,
)


def validate_report(
    report, selections, digests, entries, manifest, manifest_sha, vocabulary
):
    expected = {
        "schema": "rdk-model-zoo/paraformer-native-run/v1",
        "status": "completed",
        "execution_backend": "native-sdk",
        "target": "s100",
        "manifest_sha256": manifest_sha,
        "vocabulary_sha256": VOCABULARY_DIGEST,
        "manifest_path": str(manifest),
    }
    if not isinstance(report, dict) or any(
        report.get(k) != v for k, v in expected.items()
    ):
        raise ValueError("Native report identity/backend/status mismatch")
    if (
        report.get("inference_attempted") is not True
        or report.get("inference_executed") is not True
    ):
        raise ValueError("Native inference was not completed")
    models = report.get("models")
    if not isinstance(models, list) or len(models) != 3:
        raise ValueError("Expected three native model records")
    for record, selection, digest in zip(models, selections, digests, strict=True):
        if not isinstance(record, dict) or any(
            record.get(k) != v
            for k, v in {
                "stage": selection.stage,
                "asset_id": selection.asset.reference,
                "path": str(selection.model_path.resolve()),
                "sha256": digest,
            }.items()
        ):
            raise ValueError("Native model identity mismatch")
        meta = record.get("metadata")
        if (
            not isinstance(meta, dict)
            or not isinstance(meta.get("model_name"), str)
            or not meta["model_name"]
        ):
            raise ValueError("Missing native model metadata")
        converted = {
            "model_name": meta["model_name"],
            "model_names": [meta["model_name"]],
        }
        roles = {}
        for side in ("input", "output"):
            tensors = meta.get(side + "s")
            if not isinstance(tensors, list) or not tensors:
                raise ValueError("Missing tensor metadata")
            names, shapes, dtypes, role_names = [], {}, {}, {}
            for tensor in tensors:
                if not isinstance(tensor, dict):
                    raise ValueError("Invalid tensor metadata")
                name, role = tensor.get("name"), tensor.get("role")
                shape, strides, size = (
                    tensor.get("shape"),
                    tensor.get("strides"),
                    tensor.get("allocation_bytes"),
                )
                if (
                    not isinstance(name, str)
                    or not name
                    or name in names
                    or not isinstance(role, str)
                    or role in role_names
                ):
                    raise ValueError("Duplicate or invalid native tensor names/roles")
                if (
                    not isinstance(shape, list)
                    or not shape
                    or any(type(n) is not int or n <= 0 for n in shape)
                    or not isinstance(strides, list)
                    or len(strides) != len(shape)
                    or type(size) is not int
                    or size <= 0
                ):
                    raise ValueError("Invalid tensor allocation geometry")
                span = 4
                for n, stride in reversed(list(zip(shape, strides, strict=True))):
                    if (
                        type(stride) is not int
                        or stride < span
                        or stride % 4
                        or stride * n > size
                    ):
                        raise ValueError("Invalid native tensor strides")
                    span = stride * n
                names.append(name)
                shapes[name] = tuple(shape)
                dtypes[name] = tensor.get("dtype")
                role_names[role] = name
            converted.update(
                {
                    side + "_names": names,
                    side + "_shapes": shapes,
                    side + "_dtypes": dtypes,
                }
            )
            roles[side] = role_names
        binding = bind_model(selection, converted)
        if roles["input"] != dict(binding.inputs) or roles["output"] != dict(
            binding.outputs
        ):
            raise ValueError("Native semantic tensor roles disagree with model binding")
    records = report.get("records")
    if not isinstance(records, list) or len(records) != len(entries) or not records:
        raise ValueError("Native utterance count mismatch")
    for record, entry in zip(records, entries, strict=True):
        if not isinstance(record, dict) or record.get("source_record") != entry:
            raise ValueError("Native source record mismatch")
        if (
            not isinstance(record.get("feature_path"), str)
            or Path(record["feature_path"]).resolve()
            != (manifest.parent / entry["feature_file"]).resolve()
        ):
            raise ValueError("Native feature path mismatch")
        values = {
            "utt_id": entry["utt_id"],
            "feature_sha256": entry["feature_sha256"].lower(),
            "valid_frames": entry["feat_length"],
            "original_frames": entry["original_frames"],
            "truncated": entry["truncated"],
        }
        if (
            any(record.get(k) != v for k, v in values.items())
            or any(
                type(record.get(k)) is not int
                for k in ("valid_frames", "original_frames")
            )
            or type(record.get("truncated")) is not bool
        ):
            raise ValueError("Native feature identity/geometry mismatch")
        if ("text" in entry and record.get("reference_text") != entry["text"]) or (
            "text" not in entry and "reference_text" in record
        ):
            raise ValueError("Native reference text mismatch")
        frames, original = record["valid_frames"], record["original_frames"]
        if (
            not 1 <= frames <= 400
            or original <= 0
            or frames != min(original, 400)
            or record["truncated"] is not (original > 400)
        ):
            raise ValueError("Invalid native frame/truncation contract")
        ids, count = record.get("token_ids"), record.get("token_count")
        if (
            type(count) is not int
            or not 0 <= count <= 100
            or not isinstance(ids, list)
            or len(ids) != count
            or any(type(i) is not int or not 0 <= i < 8404 for i in ids)
        ):
            raise ValueError("Invalid native token sequence")
        text = "".join(
            vocabulary[i].replace("@@", "")
            for i in ids
            if not (vocabulary[i].startswith("<") and vocabulary[i].endswith(">"))
        )
        if record.get("text") != text or record.get("decoder_executed") is not (
            count > 0
        ):
            raise ValueError("Native text/decoder state mismatch")
        timings = record.get("timings")
        if not isinstance(timings, dict):
            raise ValueError("Missing native timings")
        for key in ("encoder_ms", "predictor_ms", "cif_ms", "decoder_ms"):
            value = timings.get(key)
            if key == "decoder_ms" and count == 0:
                if value is not None:
                    raise ValueError("Bypassed decoder must have null timing")
            elif (
                type(value) not in (int, float) or not math.isfinite(value) or value < 0
            ):
                raise ValueError("Invalid native stage timing")
