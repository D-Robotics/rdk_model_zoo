English | [简体中文](README_cn.md)

# Paraformer input audio

[Frontend guide](../runtime/python/README.md)

The two mono 16 kHz PCM16 WAVs and `manifest.json` are unchanged from S commit
`380e1a2bf42041af54be6f34935e50197cfadff9`. They provide small source-comparison
inputs for feature generation and inference.

| Utterance ID | Samples | Reference text | Valid frontend frames |
| --- | --- | --- | --- |
| BAC009S0724W0121 | 68496 | 广州市房地产中介协会分析 | 71 |
| BAC009S0724W0168 | 75137 | 新地王的诞生迅速搅热南沙土地市场 | 78 |

The references are source annotations, not model predictions.
The frame counts are measured with the pinned FunASR frontend described in the
runtime guide. Both inputs produce float32 `[1,400,560]` features with zero padding
and no truncation.


## Directory structure

```text
test_data/
├── audio/  # Files for audio
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
└── manifest.json  # Structured data
```

## Layout and custom inputs

`manifest.json` is a JSON list of objects containing `utt_id` and reference `text`.
Each corresponding audio file is `audio/<utt_id>.wav`. Keep IDs unique and treat
IDs as names, not paths. Preserve reference text separately from predictions and
never rewrite this source manifest when generating features. The Python manifest
CLI writes separate prepared-manifest/feature outputs; see the runtime guide. The
native launcher reads the Python-prepared `prepared-manifest.json` and feature NPY
files instead of audio, as described in the
[native guide](../runtime/cpp/README.md#quickstart). Use the documented named
arguments for S100 SDK inference.

The current frontend API accepts finite float32 samples loaded from mono or
multichannel 16 kHz audio; multichannel audio is averaged. It does not resample.
More than 400 LFR frames preserve only the first 400 and return `truncated=True`;
that is not long-audio transcription. See the runtime guide for a complete
board-free, runnable feature example and dependencies.

## Reproducible verification

With the documented frontend environment active, regenerate the features from
repository root:

```bash
python samples/speech/paraformer/runtime/python/main.py --preprocess-only --output-dir outputs/paraformer-prepared
```

The prepared manifest records each feature file's SHA-256, original/valid frame
lengths and truncation state; the `feat_length` values are 71 and 78. The run
does not edit this manifest. SDK/board inference, dataset CER and timing follow
the [runtime](../runtime/python/README.md) and [evaluator](../evaluator/README.md)
guides.
