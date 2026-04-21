# LPRNet Recognition Python Sample

This sample demonstrates how to run LPRNet recognition on RDK X5 with `hbm_runtime`.

## Environment Dependencies

- `RDK OS >= 3.5.0`
- `hbm_runtime` is preinstalled on the board image

## Directory Structure

```text
.
├── main.py
├── lprnet.py
├── run.sh
├── README.md
└── README_cn.md
```

## Argument Description

| Argument | Description | Default |
| --- | --- | --- |
| `--model-path` | Path to the LPRNet `.bin` model | `../../model/lpr.bin` |
| `--priority` | Runtime scheduling priority | `5` |
| `--bpu-cores` | BPU core indexes | `0` |
| `--test-bin` | Path to the packed float32 input tensor | `../../test_data/test.bin` |

## Quick Run

```bash
chmod +x run.sh
./run.sh
```

## Manual Run

- Run with default arguments:
  ```bash
  python3 main.py
  ```

- Run with explicit arguments:
  ```bash
  python3 main.py \
    --model-path ../../model/lpr.bin \
    --test-bin ../../test_data/test.bin
  ```

## Interface Description

- `LPRNetConfig`: Encapsulates model path and binary input path.
- `LPRNet`: Implements `set_scheduling_params`, `pre_process`, `forward`, `post_process`, and `predict`.
