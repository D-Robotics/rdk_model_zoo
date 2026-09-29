# KWS test audio

English | [简体中文](README_cn.md)

`sample.wav` is copied byte-for-byte from the pinned S source (`380e1a2bf42041af54be6f34935e50197cfadff9`). It is the source's “hey snips” demonstration recording: mono PCM16, 16000 Hz, 40000 frames, 2.5 seconds.

SHA-256: `eb39ea9bff0e37e262ee3735eba4111a52bb53a84bb776d28a43d7cea6b88cad`.

The fixed frontend adds 20000 zero samples to reach 60000, then produces float32 `[1,373,80]`. Real host verification compares PCM decoding and all feature values with the original PaddleAudio path. The historical ~0.985 score belongs to source board execution and is not a new measured result.

Use `--audio-file /path/to/mono-16k.wav` in the [runtime command](../runtime/python/README.md) for another clip. Non-mono or non-16-kHz input is explicitly rejected; preserve any external conversion as a separate input. Audio longer than 3.75 seconds is truncated, with the count recorded. This directory contains no negative set, calibration corpus or transcript annotations. Do not report accuracy from this one positive recording; see [evaluation](../evaluator/README.md).
