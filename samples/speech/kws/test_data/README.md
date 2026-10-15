English | [简体中文](README_cn.md)

# KWS test audio


## Directory structure

```text
test_data/
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```


`sample.wav` is the source example's “hey snips” demonstration recording: mono PCM16, 16000 Hz, 40000 frames, 2.5 seconds.

The fixed frontend adds 20000 zero samples to reach 60000, then produces float32 `[1,373,80]`. The reference score for this clip on S100 is approximately 0.985.

Use `--audio-file /path/to/mono-16k.wav` in the [runtime command](../runtime/python/README.md) for another clip. Non-mono or non-16-kHz input is explicitly rejected; preserve any external conversion as a separate input. Audio longer than 3.75 seconds is truncated, with the count recorded. Use labeled positive and negative clips with the [evaluator](../evaluator/README.md) to calculate application-level metrics.
