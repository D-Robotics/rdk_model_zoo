"""Evaluate actual legacy HBM on the shared WikiText2 token stream."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
from hbm_runtime import HB_HBMRuntime


def main():
    """Run the pinned board evaluation; fail on incompatible tensors or input."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=int, default=140, choices=range(1, 141))
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--input-ids", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("legacy-ppl.json"))
    a = p.parse_args()
    input_hash = hashlib.sha256(a.input_ids.read_bytes()).hexdigest()
    if input_hash != "a82d4dedc5f60009e026d2cc8f96513054831d9b0bea407cf73b43d3411e2653":
        raise ValueError("Input IDs differ from the pinned WikiText2 TEST stream")
    ids = np.load(a.input_ids, allow_pickle=False).reshape(-1)
    digest = hashlib.sha256()
    with a.model.open("rb") as model_file:
        for block in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(block)
    model_hash = digest.hexdigest()
    m = HB_HBMRuntime(str(a.model))
    n = m.input_names["prefill"]
    o = m.output_names["prefill"]
    assert len(n) == 87 and len(o) == 85
    assert list(m.input_shapes["prefill"][n[0]]) == [1, 256]
    assert list(m.output_shapes["prefill"][o[0]]) == [1, 256, 130560]
    dtypes = {"S32": np.int32, "S16": np.int16, "S8": np.int8}
    x = {
        k: np.zeros(
            m.input_shapes["prefill"][k],
            dtype=dtypes[str(m.input_dtypes["prefill"][k]).split(".")[-1]],
        )
        for k in n
    }
    for ik, ok in zip(n[3:], o[1:]):
        iq = m.input_quants["prefill"][ik]
        oq = m.output_quants["prefill"][ok]
        assert list(m.input_shapes["prefill"][ik]) == [4096, 2, 128]
        assert list(m.output_shapes["prefill"][ok]) == [256, 2, 128]
        assert np.all(iq.zero_point == 0)
        assert np.array_equal(iq.scale, oq.scale) and np.array_equal(
            iq.zero_point, oq.zero_point
        ), (ik, ok)
        assert x[ik].dtype == dtypes[str(m.output_dtypes["prefill"][ok]).split(".")[-1]]
    q = m.output_quants["prefill"][o[0]]
    assert q.scale.size == 1
    scale = float(q.scale[0])
    assert math.isfinite(scale) and scale > 0
    zero = float(q.zero_point[0])
    maskq = m.input_quants["prefill"][n[2]]
    assert maskq.scale.size == 1
    assert np.all(maskq.zero_point == 0)
    masked = int(
        np.clip(
            np.rint(-32767 / float(maskq.scale[0]) + float(maskq.zero_point[0])),
            -32768,
            32767,
        )
    )
    start_time = time.time()
    total = 0.0
    count = 0
    rows = []
    for sample in range(a.samples):
        for k in n[3:]:
            x[k].fill(0)
        tokens = ids[sample * 2048 : (sample + 1) * 2048]
        assert len(tokens) == 2048
        loss = 0.0
        for offset in range(0, 2048, 256):
            x[n[0]][:] = tokens[offset : offset + 256]
            x[n[1]][:] = np.arange(offset, offset + 256)
            mask = x[n[2]]
            mask.fill(masked)
            # Old valid KV precedes the current causal chunk at the right edge.
            mask[:, 4096 - 256 - offset : 4096 - 256] = 0
            mask[:, -256:] = np.where(
                np.arange(256)[None, :] <= np.arange(256)[:, None], 0, masked
            )
            out = m.run({"prefill": x})["prefill"]
            # Score the next chunk's first token, but never the next segment.
            valid = min(256, 2047 - offset)
            logits = out[o[0]][0, :valid].astype(np.float32)
            logits -= zero
            logits *= scale
            labels = tokens[offset + 1 : offset + 1 + valid]
            selected = logits[np.arange(valid), labels].copy()
            maximum = logits.max(axis=1)
            logits -= maximum[:, None]
            np.exp(logits, out=logits)
            loss += float(
                (np.log(logits.sum(axis=1)) + maximum - selected).sum(dtype=np.float64)
            )
            del logits
            # K and V can have different integer widths. Preserve their raw
            # tensors; matching scales above make another quantization needless.
            for ik, ok in zip(n[3:], o[1:]):
                cache = x[ik]
                cache[:-256] = cache[256:]
                cache[-256:] = out[ok]
            del out
        total += loss
        count += 2047
        rows.append({"index": sample, "nll": loss, "predicted_tokens": 2047})
        result = {
            "num_samples": len(rows),
            "seq_len": 2048,
            "chunk_size": 256,
            "predicted_tokens": count,
            "total_nll": total,
            "perplexity": math.exp(total / count),
            "elapsed_seconds": time.time() - start_time,
            "samples": rows,
            "input_ids_sha256": input_hash,
            "model_sha256": model_hash,
        }
        temporary = a.output.with_suffix(a.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temporary.replace(a.output)
        print(
            json.dumps({k: v for k, v in result.items() if k != "samples"}), flush=True
        )
    print(
        "FULL_EVALUATION_COMPLETE" if a.samples == 140 else "PARTIAL_EVALUATION_ONLY",
        flush=True,
    )


if __name__ == "__main__":
    main()
