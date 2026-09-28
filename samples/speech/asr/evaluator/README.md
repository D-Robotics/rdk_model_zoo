# ASR transcript evaluation

English | [简体中文](README_cn.md)

<a id="dataset"></a>
## Dataset
This evaluator scores previously saved reference/hypothesis pairs. No labeled corpus is supplied and no dataset accuracy was measured. Use unique utterance IDs and preserve your dataset version, split, license and model/decoder provenance. Runtime chunk reports are not labeled prediction datasets: align their full transcripts with independent references before constructing the input schema below.

<a id="environment"></a>
## Environment
Python 3.10+ standard library only; no board, SDK, NumPy or model download. The evaluator performs no inference. Supplied provenance is retained but is not independently authenticated.

<a id="command"></a>
## Command
```bash
# cwd: repository root; synthetic example, not model predictions
asr_eval_dir=$(mktemp -d)
cat > "$asr_eval_dir/transcripts.json" <<'JSON'
{
  "schema": "rdk-model-zoo/asr-transcripts/v1",
  "provenance": {"dataset": "synthetic-example", "model": "none", "decode_mode": "ctc"},
  "records": [
    {"id": "one", "reference": "你好", "hypothesis": "你号"},
    {"id": "two", "reference": "世界", "hypothesis": "世界"}
  ]
}
JSON
python3 samples/speech/asr/evaluator/evaluate.py --predictions "$asr_eval_dir/transcripts.json" --output "$asr_eval_dir/metrics.json"
# Expected: count=2, reference_characters=4, substitutions=1, cer=0.25
```
Both `--predictions` and `--output` are required paths, with no default. Existing output files are rejected to preserve previous reports. Input requires the schema above, nonempty `dataset`, `model`, `decode_mode` provenance strings and a nonempty records list. Each record requires a unique nonempty `id` and string `reference`/`hypothesis`; either text may be empty.

<a id="metrics"></a>
## Metrics
Character error rate is `(substitutions + deletions + insertions) / total reference characters`, aggregated across utterances, not averaged per utterance. Characters are Python Unicode code points, including spaces, punctuation and case. There is no normalization, token stripping or word-boundary conversion. Combining marks count separately. If all references are empty CER is `null`; insertions are still counted. CER may exceed 1. Equal-cost edit alignments prefer diagonal, then deletion, then insertion, making the error breakdown deterministic.

<a id="outputs"></a>
## Outputs
The exclusive output JSON uses `rdk-model-zoo/asr-metrics/v1` and includes `count`, `reference_characters`, `exact_matches`, `cer`, aggregate `errors`, per-utterance `utterances`, normalization policy, input SHA-256 and unchanged provenance. `inference_executed` and `provenance_independently_verified` are false. Exit 0 means scoring succeeded, not recognition passed; malformed input or an existing output returns 2.

<a id="reference-results"></a>
## Reference results
The S source records S100 `hrt_model_exec perf --model_file asr.hbm --frame_count 100`: 100 frames, 34.426 ms average latency and 29.008 FPS. These are historical model-execution figures, not migrated end-to-end audio latency, not S600 results and not newly reproduced measurements. SDK/toolchain/model digest details are incomplete in the source.

![Historical performance](../test_data/readme_img/perf.jpg)
![Historical transcription](../test_data/readme_img/print.jpg)
![Historical quantization comparison](../test_data/readme_img/acc.jpg)

The accuracy illustration concerns a historical quantization comparison, not corpus CER. The source's “first three seconds” description is corrected: the bundled 4.59-second file is processed as three independent windows in the canonical runtime. Host tests and numerical frontend comparisons are tracked in [migration evidence](../../../../docs/releases/unified-migration/evidence/2026-09-28-b10-asr-core/); no real model transcript is inferred from fake SDK outputs.

<a id="boundaries"></a>
## Boundaries
CTC and legacy must be reported separately because they can decode the same logits differently. Chunking, resampling and final padding also affect transcription. Offline scoring cannot establish runtime correctness, quantization equivalence, model accuracy without authentic predictions, or latency. Board/model execution, OE conversion and corpus evaluation remain not-run.
