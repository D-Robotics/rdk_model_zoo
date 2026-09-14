[中文](./README_cn.md)

# C++ Runtime

The C++ implementation preserves the hbUCP/hbDNN pipeline, including the aligned tensor copies required by BPU memory layouts, the three HBM stages, and CPU CIF. Users still start from WAV: `run.sh` invokes `../python/main.py --preprocess-only`, which uses the FunASR frontend in `paraformer.py` to generate temporary `[1, 400, 560]` features and valid frame lengths. The C++ executable consumes only this temporary bridge data; HBM inference logic does not change.

## Run

```bash
cd runtime/cpp
bash run.sh [test_data_directory]
```

The script downloads missing models, creates an isolated Python frontend environment, generates `build/eval/feats/` from WAV, builds with CMake, and runs the bundled manifest. Set `N_UTT=10` to limit items. Generated `.npy` features are an internal bridge, not user input.

## Input data

```text
<AUDIO_DATA_DIR>/
├── manifest.json
└── audio/
    └── <utt_id>.wav
```

WAV input must be 16 kHz. Each manifest object requires `utt_id` and may include reference `text`. The board needs the S100 `dnn` and `hbucp` libraries plus the `nlohmann/json.hpp` development header.
