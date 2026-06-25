#!/usr/bin/env python3
"""轻量 Text BC 验证：只对比最后一个 chunk 的 logits cosine。

Usage:
    python quick_text_verify.py \
        --model-dir ../../gemma4-e2b \
        --bc-path ../../output/gemma4_e2b_text/gemma4-e2b_lm_chunk_256_cache_4096_ptq.prefill_convert.bc \
        --prompts ../../calibration_data/text_verify/calibration.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from hbdk4.compiler.apis import load

from leap_llm.apis.model.gemma4 import Gemma4TextCalibrationDataPreparer
from leap_llm.apis.verifier.utils import calculate_cosine
from leap_llm.models.gemma4.model import Gemma4Text

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHUNK_SIZE = 256
CACHE_LEN = 4096
DEVICE = "cpu"


def prepare_chunk_inputs(text_model, preparer, prompt, bc_meta):
    chunks, pos, full_masks, sliding_masks = preparer.prepare_inputs(prompt)
    if not chunks:
        raise ValueError("empty prompt")
    idx = len(chunks) - 1
    input_ids = chunks[idx]
    position_ids = pos[idx]
    full_mask = full_masks[idx]
    sliding_mask = sliding_masks[idx]
    caches = text_model.build_empty_caches(device=DEVICE, transpose_cache=True)

    for i in range(idx):
        ids = chunks[i].to(DEVICE)
        pids = pos[i].to(DEVICE)
        fm = full_masks[i].to(DEVICE)
        sm = sliding_masks[i].to(DEVICE)
        emb = text_model.get_input_embeddings(ids)
        out = text_model.forward(emb, ids, pids, fm, sm, caches)
        caches = text_model.update_caches(caches, list(out[1:]), ids.shape[-1])

    inputs_embeds = text_model.get_input_embeddings(input_ids)
    primary = [inputs_embeds.squeeze(0), input_ids, position_ids, full_mask, sliding_mask]
    feed = {}
    for i, data in enumerate(primary):
        name, _shape, dtype = bc_meta[i]
        feed[name] = data.detach().cpu().numpy().astype(dtype)
    for j in range(5, len(bc_meta)):
        name, _shape, dtype = bc_meta[j]
        feed[name] = caches[j - 5].detach().cpu().numpy().astype(dtype)
    return input_ids, inputs_embeds, position_ids, full_mask, sliding_mask, caches, feed


def main():
    parser = argparse.ArgumentParser(description="Text BC lightweight verification")
    parser.add_argument("--model-dir", default=str(REPO_ROOT / "gemma4-e2b"))
    parser.add_argument("--bc-path", default=str(REPO_ROOT / "output/gemma4_e2b_text/gemma4-e2b_lm_chunk_256_cache_4096_ptq.prefill_convert.bc"))
    parser.add_argument("--prompts", default=str(REPO_ROOT / "calibration_data/text_verify/calibration.json"))
    parser.add_argument("--output", default=str(REPO_ROOT / "output/e2b_text_verify_quick.json"))
    args = parser.parse_args()

    with open(args.prompts, encoding="utf-8") as f:
        prompts = [x["text"] for x in json.load(f)]

    print(f"[1/4] Load Gemma4Text from {args.model_dir}", flush=True)
    text = Gemma4Text.load_model(args.model_dir, chunk_size=CHUNK_SIZE, cache_len=CACHE_LEN)
    text.set_compile_mode(False)
    text.set_model_device(DEVICE, torch.float32)

    preparer = Gemma4TextCalibrationDataPreparer(
        args.model_dir,
        chunk_size=CHUNK_SIZE,
        cache_len=CACHE_LEN,
        sliding_window=text.config.sliding_window,
        device=DEVICE,
        mask_value=-32768,
    )

    print(f"[2/4] Load BC {args.bc_path}", flush=True)
    bc = load(args.bc_path)
    bc_meta = [(inp.name, inp.type.shape, inp.type.np_dtype) for inp in bc.functions[0].inputs]

    results = []
    for n, prompt in enumerate(prompts):
        print(f"[3/4] prompt_{n}: {prompt[:40]}...", flush=True)
        ids, emb, pids, fm, sm, caches, feed = prepare_chunk_inputs(text, preparer, prompt, bc_meta)
        with torch.no_grad():
            torch_out = text.forward(
                emb.to(DEVICE), ids.to(DEVICE), pids.to(DEVICE),
                fm.to(DEVICE), sm.to(DEVICE), caches,
            )
        torch_logits = torch_out[0].detach().cpu().numpy()

        bc_out = bc.functions[0].feed(inputs=feed)
        bc_logits = bc_out["_output_0"]
        if hasattr(bc_logits, "numpy"):
            bc_logits = bc_logits.numpy()

        cos = calculate_cosine(torch_logits, bc_logits)
        results.append({"prompt_id": f"prompt_{n}", "cosine": cos, "prompt": prompt})
        print(f"      cosine = {cos:.6f}", flush=True)

    print("[4/4] Summary", flush=True)
    vals = [r["cosine"] for r in results if r["cosine"] is not None]
    mean = float(np.mean(vals)) if vals else None
    for r in results:
        print(f"  {r['prompt_id']}: {r['cosine']}")
    print(f"  mean: {mean}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"results": results, "mean_cosine": mean}, f, indent=2, ensure_ascii=False)
    print(f"Saved: {args.output}")
    return 0 if mean and mean >= 0.99 else 1


if __name__ == "__main__":
    sys.exit(main())
