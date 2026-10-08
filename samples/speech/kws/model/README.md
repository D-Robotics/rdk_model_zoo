# KWS model artifacts

English | [简体中文](README_cn.md)

<a id="artifacts"></a>
## Published artifact

| Identity | Target | Local path | Publisher SHA-256 |
| --- | --- | --- | --- |
| `s:kws:s100/kws.hbm` | S100 | `model/s100/kws.hbm` within this sample | Not recorded |

The authoritative [active manifest](../../../../docs/release/s/models.yaml) supplies the download URL. S100P/S600/X5 have no KWS publication; editing a filename or forcing a board alias does not create one.

<a id="directory"></a>
## Directory structure

```text
model/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── download.py  # Prepare model files
└── download.sh  # Model preparation command
```

<a id="preparation"></a>
## Explicit preparation

From the repository root, download separately from inference:

```sh
bash samples/speech/kws/model/download.sh --target s100
```

`--asset-id` defaults to the exact identity above and rejects any other value. `--output-dir /path/to/models` stores `s100/kws.hbm` under that directory. `PYTHON` selects the interpreter. The shared downloader checks file existence/size and configured checksums; this record has no publisher checksum, so the printed observed SHA binds bytes but does not authenticate their origin. No board is required to download. Errors return 2; incomplete downloads must not be treated as ready models.

<a id="accompanying-files"></a>
## Accompanying data

The fixed 80-bin PaddleAudio feature protocol and “hey snips” semantics belong to this artifact. There is no editable text vocabulary and no second model. The original [sample.wav](../test_data/sample.wav) is a demonstration input, not calibration data or a validation dataset. See [input identity](../test_data/README.md).

<a id="local-paths"></a>
## Existing file

The default is resolved relative to the sample, independent of the caller's working directory. To use an existing file, pass both `--model-path` and `--asset-id s:kws:s100/kws.hbm` to the [runtime](../runtime/python/README.md). Use the published model URL and checksum when available. Runtime checks actual board identity before SDK loading and checks exactly one model/input/output after loading.

<a id="formats-checksums"></a>
## Format and runtime contract

HBM is an S100 compiled runtime file, not ONNX or a Paddle checkpoint. Input is finite float32 `[1,373,80]`, derived from 60000 mono samples at 16 kHz. Output must have batch one, positive static dimensions and finite data. Output shape is read from SDK metadata rather than guessed from a historical console score. Float32 probabilities are used directly; integer outputs require valid SCALE metadata and shared dequantization before max reduction. Values outside [0,1] are rejected; no extra sigmoid or implicit clipping is used.

Keep the downloaded file's SHA-256 with each run report to identify the exact model bytes used.
