# ASR transcript evaluation

English | [简体中文](README_cn.md)

<a id="dataset"></a>
## Dataset
This evaluator computes character error rate from saved reference/hypothesis pairs. Use unique utterance IDs and preserve the dataset version, split, license, model and decoder metadata. For chunked runtime output, assemble each full transcript and pair it with its independent reference before adding the record.

<a id="environment"></a>
## Environment
Python 3.10+ standard library only. The evaluator reads transcript JSON and writes metrics without model inference; the supplied provenance object is copied to the report.

<a id="command"></a>
## Command
```bash
# Repository root; example input with two transcript records.
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
The exclusive output JSON uses `rdk-model-zoo/asr-metrics/v1` and includes `count`, `reference_characters`, `exact_matches`, `cer`, aggregate `errors`, per-utterance `utterances`, normalization policy, input SHA-256 and unchanged provenance. The schema records `inference_executed=false` and `provenance_independently_verified=false`. Exit 0 means the input was scored and the report was written; malformed input or an existing output returns 2.

<a id="reference-results"></a>
## Reference results
The S source records this S100 command: `hrt_model_exec perf --model_file asr.hbm --frame_count 100`. It reports 100 frames, 34.426 ms average latency and 29.008 FPS for model execution; these figures are not end-to-end audio latency.

![Reference performance](../test_data/readme_img/perf.jpg)
![Reference transcription](../test_data/readme_img/print.jpg)
![Reference quantization comparison](../test_data/readme_img/acc.jpg)

The figures show model-execution performance, a transcription example and a quantization comparison. The bundled 4.59-second WAV is processed as three independent windows. For corpus CER, align full-file transcripts with independent utterance references using the schema above.

<a id="boundaries"></a>
## Boundaries
CTC and legacy decode the same logits differently, so report their results separately. Chunking, resampling and final padding affect transcription. For system comparisons, use complete transcripts with matched references and record the target, model, decoder, dataset split and runtime configuration.
