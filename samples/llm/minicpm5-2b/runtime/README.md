# MiniCPM native launch orchestration

[简体中文](README_cn.md) | **English**

The Python 3 standard-library launcher selects the native SDK backend. Tokenization,
model execution and generation remain in C++. S100/S100P use [legacy OELLM 1.0.0](legacy/README.md);
S600 uses the [OELLM 2.0 backend](cpp/README.md). Shared filenames do not make SDKs or
artifacts interchangeable. These version names describe the pinned source release.

## Prepare, build, run

From the repository root, on the appropriate board:

```bash
cd samples/llm/minicpm5-2b
# Obtain the matching SDK separately and set its extracted runtime directory.
export OELLM_RUNTIME_ROOT=/path/to/matching-sdk/oellm_runtime
BOARD=s600 bash model/download_model.sh
python3 runtime/launcher.py --target s600 --build
python3 runtime/launcher.py --target s600 -- --prompt 'What is the capital of France?'
```

For S100/S100P, change both explicit target selections and use their OELLM 1.0.0
SDK. No launcher action installs packages, downloads models or runs quantization.
`--build` does not run inference. Normal execution never builds a missing binary.
Use separate model/build directories for each target. Native arguments follow `--`.

## Host preview

```bash
python3 samples/llm/minicpm5-2b/runtime/launcher.py --target s100p --dry-run
python3 samples/llm/minicpm5-2b/runtime/launcher.py --target s600 --build --dry-run
```

Preview emits JSON describing the SDK family, paths, command and timeout, without
executing subprocesses, reading models, creating files or requiring SDK installation.
A missing runtime root appears as `null` / `<set-runtime-root>`. Real build/run checks
board identity first and rejects a mismatched or unrecognized host; preview is not
proof of runtime compatibility. This migration has not run board tests.

## Options and environment

| Launcher option | Default / behavior |
| --- | --- |
| `--target` | `auto`; concrete `s100`, `s100p`, `s600` supported; preview on a host needs explicit target |
| `--runtime-root` | `OELLM_RUNTIME_ROOT`, then `$OELLM_SDK_ROOT/oellm_runtime`; SDK is never downloaded |
| `--model-dir` | `MODEL_DIR`, then sample `model/<target>` |
| `--build-dir` | `runtime/cpp/build-s600` or `runtime/legacy/build-<target>` |
| `--build` | Run CMake configure/build only; rejects native inference arguments |
| `--dry-run` | Print the chosen action, no execution |
| `--timeout` | S100/S100P timeout, positive finite seconds; `INFERENCE_TIMEOUT` or 120; no added S600 timeout |
| `-- NATIVE_ARGS` | Forward exact argument boundaries to the chosen C++ CLI |

S600 uses native `--model_path`, `--prompt`, `--follow_up`, `--max_new_tokens`.
Legacy uses `--model-path`, `--tokenizer-path`, `--template-path`, `--prompt`, and
has no equivalent output-token limit. Prefer launcher `--model-dir` for model
selection. Native direct invocation and its flags remain available in each guide.
Launcher binaries live in the per-target build directories; the manual CMake
examples use their explicitly selected `build` directory instead.

The launcher prefixes SDK `lib` to `LD_LIBRARY_PATH`; S600 sets L2M to `6:6:6:6`.
Native exit codes propagate; orchestration errors return 2 and legacy timeout 124.
The source model preparation command verifies pinned archive/manifest hashes;
launch itself does not repeat that verification. See [model guide](../model/README.md).

## Migration boundary

The native inference implementation and full README contract are still being
refactored. Source quantization/evaluator files and historical precision failures
remain available unchanged; a green launcher test does not close those migration
tasks or turn historical board results into new evidence.
