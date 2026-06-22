English | [简体中文](./README_cn.md)

# KWS Model Description

This directory describes the complete workflow for KWS (Keyword Spotting) in this Model Zoo, including: algorithm introduction, model conversion, runtime inference (Python), reusable pre/post-processing interface descriptions, and model evaluation steps.

> This model supports **RDK S100** platform.

---

## Algorithm Overview

KWS (Keyword Spotting) is a deep learning-based wake word detection model that uses the MDTC (Multi-Scale Dynamic Temporal Convolution) algorithm with the following features:

- **Multi-scale convolution**: Captures speech features at different temporal scales, improving detection robustness
- **Dynamic convolution**: Adaptively adjusts convolution weights to accommodate different speakers and environments
- **Edge-friendly**: Lightweight design suitable for embedded platform deployment
- **High accuracy**: Runs efficiently on BPU with high detection accuracy

### Algorithm Capabilities

KWS can accomplish the following tasks:

- Keyword spotting (input .wav audio, output keyword confidence score)

### References

- Framework: PaddlePaddle + PaddleAudio
- Algorithm: MDTC (Multi-Scale Dynamic Temporal Convolution)

---

## Platform Compatibility

| Platform   | Support     | Notes                                         |
|-----------|-------------|-----------------------------------------------|
| RDK S100  | ✅ Supported | Model compiled for S100 BPU, recommended      |
| RDK S600  | ❌ Not supported | Not yet adapted                           |

---

## Directory Structure

This directory contains:

```bash
.
|-- conversion                          # Model conversion workflow
|   `-- README.md                       # Model conversion instructions
|-- evaluator                           # Model evaluation content
|   `-- README.md                       # Model evaluation instructions
|-- model                               # Model files and download script
|   |-- download_model.sh               # HBM model download script
|   `-- README.md                       # HBM model download instructions
|-- runtime                             # Model inference samples
|   `-- python                          # Python inference sample
|       |-- README.md                   # Python sample instructions
|       |-- main.py                     # Python inference entry script
|       |-- kws.py                      # KWS model wrapper
|       `-- run.sh                      # Python example run script
|-- test_data                           # Test data
|   `-- sample.wav                      # Sample wake word audio ("hey snips")
`-- README.md                           # KWS overview and quick start
```

---

## Quick Start

Each model provides a `run.sh` script for quick setup. The script performs the following:
- Checks system environment and installs dependencies if needed
- Checks if the required HBM model file exists, downloads it if not
- Runs the Python script

### Python

- Navigate to the `runtime/python/` directory and run the `run.sh` script
    ```bash
    cd runtime/python/
    ./run.sh
    ```
- For detailed Python usage or step-by-step execution, refer to `runtime/python/README.md`

---

## Model Conversion

- The Model Zoo provides pre-adapted HBM model files. Users can directly run the `download_model.sh` script in the `model` directory to download and use them. If you are not concerned with the model conversion process, **you can skip this section**.
- For custom model conversion parameters or the complete conversion workflow, refer to `conversion/README.md`.

---

## Runtime

KWS inference sample is available in Python implementation only.

### Python Version

- Provided as scripts, suitable for quickly verifying model effects and algorithm flow;
- Demonstrates the complete process of model loading, audio preprocessing (fbank feature extraction), inference execution, and confidence score output;
- For usage, parameters, and interface details, refer to `runtime/python/README.md`;

---

## Inference Result

After successful execution, the confidence score will be printed to the terminal. Example output:

```text
Keyword confidence score: 0.9850
```

---

## Model Evaluation

`evaluator/` is used for model accuracy, performance, and numerical consistency evaluation. Refer to that directory for details.

---

## License

Follow the top-level Model Zoo License.
