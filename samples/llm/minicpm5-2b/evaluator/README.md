[English](README.md) | [简体中文](README_cn.md)

# Full HBM evaluation

Use the [conversion environment](../conversion/README.md) on an x86 host. SAMPLE points to this sample and WORK to your working directory. Preparation needs the matching SDK and pinned checkpoint, but does not need CUDA. It verifies checkpoint, TEST parquet and FP16 embedding hashes, then exports token IDs and masks using the original SDK.

```bash
python "$SAMPLE/evaluator/prepare.py" --model-path "$WORK/MiniCPM5-2B" \
  --test-data "$WORK/datasets/wikitext2-test/test-00000-of-00001.parquet" \
  --output-dir "$WORK/ppl-bundle"
scp -r "$WORK/ppl-bundle" user@BOARD:/data/
```

On the board, from this evaluator directory:

```bash
python3 -m pip install --target .deps grpcio==1.74.0 protobuf==4.25.8 numpy
EVAL_BUNDLE=/data/ppl-bundle bash run.sh
python3 validate_result.py board-local-ppl.json
# Optional wiring check; this is NOT a complete evaluation.
EVAL_BUNDLE=/data/ppl-bundle OUTPUT=quick-check.json bash run.sh --samples=1
```

The generated bundle contains SDK-owned protocol/RPC files copied from the installed hbm-infer 3.15.3. They are not committed here and remain subject to SDK terms. The helper starts the service, reads its allocated port and cleans its own temporary directory on exit. MODEL_DIR defaults to `../model/s600`; OUTPUT defaults to `board-local-ppl.json`. Do not load another model concurrently: the tested board configuration cannot hold two instances of this model.

## Statistics and evidence

[Recorded full result](results/s600-wikitext2-full.json): 140 independent segments of 2048 tokens, eight 256-token chunks per segment, KV reset between segments. Every one of the 2047 next-token labels per segment is scored, including chunk boundaries: **286580 predictions**. PPL is the exponential of total negative log likelihood divided by that count. No chat prefix is added. Joining 4358 dataset rows with two newlines produces 288009 token IDs; the last incomplete segment is excluded consistently in all paths.

Final PPL **14.242767676160279**, compared with float 14.0184 and fake-quant 14.2687; relative increase **1.60052%**, elapsed **913.192 seconds**. Partial JSON is updated per segment and is not complete before 140 segments and the completion marker. The validator checks mathematical consistency, completeness and the 3% relative-PPL target; it does not independently prove which model was executed.

The local evaluator uses NumPy float32 log-softmax and float64 loss sums to avoid transferring full logits/KV over the network. First-five-segment PPL: SDK/PyTorch RPC 15.047808, local NumPy 15.048632, a 0.0055% difference. The delivery wrapper reproduces first-segment NLL 4907.887529. That wiring check does not replace the full run.

HBM SHA256: `7c54a0934b95c26ec378f93716618f17eb58d3efd5d5b3de7b016040513ed0ee`.

- Accuracy: hbm-infer 3.15.3, DNN 3.15.3_(4.11.2 HBRT).
- Generation: OELLM 2.0.4, UCP 3.15.2, DNN 3.15.2_(4.10.6 HBRT), RDK OS V5.1.0.
- RPC core IDs are 0,1,2,3; OELLM backends use 1,2,3,4.

## Generation checks

[Six prompt/token comparisons](../test_data/generation-reference.json) match official HF greedy output and terminate with EOS. Other recorded tests include a 53/45-token two-turn conversation, code retrieval at padded 2048/3840 input tokens, and 50 repeated requests. The repeat test averaged 53.2503 decode token/s and 147.8602 ms TTFT. Measurements exclude cold model loading and describe a short single-request workload, not soak/concurrency coverage. Prefill counts include chunk padding.

## S100 / S100P full validation (2026-09-09)

Use the separate [legacy evaluator](legacy/README.md), with SDK 1.0.0, UCP/DNN 3.7.3 and HBRT 4.2.11. Both board-specific HBMs completed the same 140 × 2048-token TEST stream and 286580 predictions. **Both PPL results are 17.91995474675122, a 27.83167% relative increase over float; the ≤3% accuracy target FAILS.** Completeness and numerical-consistency checks pass; the shared validator intentionally rejects the accuracy result.

| Board | Full PPL | Evaluation seconds | Reference text matches | Repeated requests |
|---|---:|---:|---:|---:|
| [S100 PPL](results/s100-wikitext2-full.json) / [generation](results/s100-generation-full.json) | 17.91995 | 1129.83 | 2/6 | 50/50 |
| [S100P PPL](results/s100p-wikitext2-full.json) / [generation](results/s100p-generation-full.json) | 17.91995 | 882.44 | 2/6 | 50/50 |

Both pass the English/Chinese two-turn conversation and retrieval from approximately 2000/3750 raw-token prompts. Each board completes all 60 requests with normal end events and successful SDK return/destruction codes. Four reference texts differ: translation is incorrect, the code response is incomplete, and JSON/list formatting differs. These are not six passing generation comparisons. The SDK exposes text but no output token IDs.

The 50-request mean end-to-end times are 671.37 ms (S100) and 543.22 ms (S100P), excluding model loading. These are wall times, not TTFT. Earlier short-request runtime logs reported approximately 12.1/13.0 decode tokens/s; zero callback performance fields are not measurements. Tools, thinking, multimodal, concurrency and long-duration soak coverage are not claimed.

[First-segment diagnostic](results/legacy-first-segment-diagnostic.json): HF float32 PPL 10.75668, legacy adapter float32 10.75612, actual HBM 14.01536. All eight masks exactly match SDK calibration helpers. The prepared input digest is shared with S600; both executed HBM digests were checked. This narrows further investigation to the quantized execution path but does not identify a specific quantization operation. The packaged evaluator independently reproduces first-segment NLL 5404.3955137729645 on both boards. S600's results above are historical and were not rerun for this addition.
