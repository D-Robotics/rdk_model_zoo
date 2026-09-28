# Paraformer input audio

[简体中文](README_cn.md) · [Frontend guide](../runtime/python/README.md)

The two mono 16 kHz PCM16 WAVs and `manifest.json` are unchanged from S commit
`380e1a2bf42041af54be6f34935e50197cfadff9`. They provide small source-comparison
inputs, not a dataset-scale accuracy benchmark or new board-validation claim.

| Utterance ID | Samples | Reference text | Valid frontend frames |
| --- | --- | --- | --- |
| BAC009S0724W0121 | 68496 | 广州市房地产中介协会分析 | 71 |
| BAC009S0724W0168 | 75137 | 新地王的诞生迅速搅热南沙土地市场 | 78 |

The references are source annotations, not this migration's model predictions.
The frame counts are measured with the pinned FunASR frontend described in the
runtime guide. Both inputs produce float32 `[1,400,560]` features with zero padding
and no truncation, byte-identical to the source under the recorded host environment.
No HBM model was run for these results.

## Layout and custom inputs

`manifest.json` is a JSON list of objects containing `utt_id` and reference `text`.
Each corresponding audio file is `audio/<utt_id>.wav`. Keep IDs unique and treat
IDs as names, not paths. Preserve reference text separately from predictions and
never rewrite this source manifest when generating features. The Python manifest CLI now writes separate prepared-manifest/feature outputs;
see the runtime guide. The native C++ consumer is still being migrated. Archived
`run.sh` positional arguments are not the unified interface.

The current frontend API accepts finite float32 samples loaded from mono or
multichannel 16 kHz audio; multichannel audio is averaged. It does not resample.
More than 400 LFR frames preserve only the first 400 and return `truncated=True`;
that is not long-audio transcription. See the runtime guide for a complete
board-free, runnable feature example and dependencies.

## Reproducible verification

With the documented frontend environment active, run from repository root:

```bash
python docs/releases/unified-migration/evidence/2026-09-28-b10-paraformer-frontend/verify_real.py
```

This checks the two copied WAVs against the pinned Git source, compares actual
FunASR outputs on seven input cases and verifies CPU RNG restoration. It creates
evidence JSON/log output, not model transcripts, and does not edit this manifest.
[Recorded results](../../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-frontend-review.md)
include source audio hashes, feature hashes, original/valid lengths and truncation.
The original source's board smoke history remains historical; new SDK/board
inference, dataset CER and timing are not-run.
