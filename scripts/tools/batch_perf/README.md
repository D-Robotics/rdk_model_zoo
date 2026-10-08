English | [简体中文](README_cn.md)

# Batch Perf

This tool runs `perf` tests in batches for every model file ending in `.bin` in the specified directory.

## Usage

Configure an alias to invoke the script from any model directory:

```bash
alias perf='python3 <path to your rdk_model_zoo>/scripts/tools/batch_perf/batch_perf.py'
```

Then run it in the model directory:

```bash
perf .
```

## Notes

- Models are tested in ascending order of file size.
- Thread counts usually range from `1` to `MAX_NUM`; `MAX_NUM` is generally set to `2`.
