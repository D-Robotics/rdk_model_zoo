[English](README.md) | [简体中文](README_cn.md)

# S100 / S100P full evaluation

Use the board-specific SDK 1.0.0 HBM and runtime libraries. This evaluator uses
the board's `hbm_runtime.HB_HBMRuntime`, not the S600 RPC bundle. The tested boards
provide Python 3.10, NumPy and `hbm_runtime`; generation checks also require g++
and `nlohmann/json.hpp` (Ubuntu package `nlohmann-json3-dev`). Set the SDK library
path before Python imports the runtime. Do not run PPL and generation concurrently
on the same board.

## Prepare the shared input on the host

In an environment with NumPy, datasets and transformers, use the original pinned
MiniCPM5-2B tokenizer and WikiText2 TEST parquet from the conversion instructions:

```bash
python prepare.py --model-path /data/MiniCPM5-2B \
  --test-data /data/wikitext2-test/test-00000-of-00001.parquet \
  --output-dir /data/legacy-ppl-input
scp -r /data/legacy-ppl-input user@BOARD:/data/
```

This checks the TEST parquet digest and exact token-file digest used by S600.
It does not require the S600 SDK. An existing verified S600 `input_ids.npy` can
also be used directly. No chat template is applied during PPL evaluation.

## Run on each board

```bash
export BOARD=s100  # s100p on S100P
export OELLM_SDK_ROOT=/data/D-Robotics_LLM_S100_1.0.0_SDK
export EVAL_BUNDLE=/data/legacy-ppl-input
bash run.sh
python3 ../validate_result.py legacy-ppl.json
# Optional one-segment wiring check; it does not meet the full evaluation target.
OUTPUT=partial.json bash run.sh --samples 1

# Run after PPL has exited, with the same SDK and board selection.
bash run_acceptance.sh | tee acceptance.log
```

The model downloader verifies the archive and extracted members before running.
`evaluate.py` also records the executed HBM digest. Results are written atomically
after every segment. Only 140 segments, 286580 scored predictions and
`FULL_EVALUATION_COMPLETE` establish a complete run. The shared validator separately
checks the relative PPL increase against the float reference 14.0184 (target ≤3%).
A complete run can fail that accuracy target.

## Numerical and generation scope

PPL uses 140 independent 2048-token segments, eight 256-token chunks per segment,
and a fresh KV cache per segment. It scores all 2047 next-token labels, including
chunk boundaries. Quantized logits are converted with HBM scales and zero points;
raw K/V caches retain their individual integer types and validated matching scales.
The legacy cache rolls on axis 0. Loss uses float32 log-sum-exp and float64 sums.

The acceptance harness uses the same greedy parameters and non-thinking template
as the legacy demo, while keeping one SDK handle for conversation and repeat tests.
It compares six HF reference texts, a bilingual two-turn conversation, retrieval
from approximately 2000/3750 raw-token prompts, and 50 fresh requests. It records
every comparison, SDK return status, normal-end/error events, and wall time.
The SDK does not expose generated token IDs, so these are text comparisons, not
S600-style token-ID equality checks. Zero callback performance fields are not
reported as measured TTFT or decode throughput. This is not a concurrency or
long-duration soak test. Tools, thinking and multimodal requests are not covered.
