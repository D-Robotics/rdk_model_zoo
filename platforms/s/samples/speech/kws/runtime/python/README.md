English | [简体中文](./README_cn.md)

# KWS Keyword Spotting Sample (Python)

This sample demonstrates how to use the KWS model on RDK platforms for keyword spotting inference. It takes .wav audio as input and outputs keyword detection confidence scores.

## Dependencies

Ensure the following dependencies are installed:

```bash
pip install numpy==1.26.4 paddlepaddle paddleaudio
```

## Directory Structure

```text
.
├── README.md               # Sample documentation
├── main.py                 # Sample entry point for keyword spotting inference
├── kws.py                  # KWS model wrapper
└── run.sh                  # One-click run script
```

## Parameters

| Parameter        | Description                                         | Default                                          |
|------------------|-----------------------------------------------------|--------------------------------------------------|
| `--model-path`   | Model file path (.hbm format)                       | Auto-detected                                    |
| `--audio-file`   | Test audio file path (.wav format)                  | `../../test_data/sample.wav`                     |
| `--audio-maxlen` | Maximum audio sample count for truncation           | `60000`                                          |
| `--frame-shift`  | fbank frame shift (ms)                               | `10`                                             |
| `--frame-length` | fbank frame length (ms)                              | `25`                                             |
| `--n-mels`       | Number of fbank Mel filter banks                     | `80`                                             |
| `--priority`     | Model priority (0~255, higher = higher priority)     | `0`                                              |
| `--bpu-cores`    | List of BPU core IDs for inference                   | `[0]`                                            |

## Quick Start

- Run with script:

    ```bash
    ./run.sh
    ```

- With default parameters:

    ```bash
    python main.py
    ```

- With custom parameters:

    ```bash
    python main.py \
    --model-path /opt/hobot/model/s100/basic/kws.hbm \
    --audio-file ../../test_data/sample.wav
    ```

## Input Format

The model input is an fbank feature tensor extracted by paddleaudio. The .wav audio file is automatically truncated/padded and converted to fbank features after loading.

## Interface Documentation

The sample code includes detailed comments. For the most accurate and up-to-date interface definitions, refer directly to the docstrings in the source code:

- **KWSConfig** and **KWS**: See `kws.py`
