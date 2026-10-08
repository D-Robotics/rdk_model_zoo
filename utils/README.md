English | [简体中文](README_cn.md)

# Shared runtime utilities


These helpers provide SDK sessions, model metadata, image and tensor processing, labels, and result rendering for Model Zoo samples. Conversion scripts also reuse numerical and image-processing functions.

## Directory structure

```text
utils/
├── py_utils/   # Python runtime and numerical helpers
├── c_utils/    # C++ runtime, image, tensor, and rendering helpers
└── tools/      # Compilation, evaluation and repository maintenance tools
```

## Usage

Use [Python helpers](py_utils/README.md) from the repository root. Include and link [C++ helpers](c_utils/README.md) through the sample's CMake project. Keep model-specific preprocessing and decoding in each sample's task file.

Repository maintenance and dataset preparation commands are in [tools](tools/).
