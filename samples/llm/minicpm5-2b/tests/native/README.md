[English](README.md) | [简体中文](README_cn.md)

# Native core host tests

## Directory structure

```text
native/
├── fixtures/  # Files for fixtures
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── legacy_lifecycle.cpp  # Source or data file
├── legacy_prepared_ownership.cpp  # Source or data file
├── legacy_request.cpp  # Source or data file
├── legacy_stream_sink.cpp  # Source or data file
├── s600_config_cleanup.cpp  # Source or data file
├── s600_metrics.cpp  # Source or data file
├── s600_stages.cpp  # Source or data file
└── scratch_dir.hpp  # Source or data file
```

C++ drivers under this directory compile the production sources against the SDK test doubles in `fixtures/` and run them on the host. The doubles are marked at the top of each header and are **not** the vendor SDK: they perform no tokenization, no BPU work and no generation; board behavior is defined by the board run. `tests/test_native_core.py` compiles and runs them; the legacy CLI scenarios additionally execute the real `src/main.cc`.

Running the suite requires a Python 3 interpreter (standard library only) and a C++ compiler; `nlohmann/json.hpp` is additionally required by the S600 drivers. Covered behavior: the legacy single-use lifecycle, request construction and greedy sampling parameters, template size limits, S600 temporary-configuration cleanup on success/error/exception paths, metric finite/nonnegative validation with zero preserved, stage boundaries and two-turn conversation fields, and test-harness isolation: every S600 driver creates its own atomically unique `mkdtemp` scratch directory (`scratch_dir.hpp`), removes only that owned directory, restores `TMPDIR` including an initially unset state, and the suite verifies a pre-existing `model/` sentinel under a hostile `TMPDIR` survives all drivers plus concurrent same-`TMPDIR` runs of the compiled binaries.

From the repository root:

```bash
python3 -m unittest discover -s samples/llm/minicpm5-2b/tests -p test_native_core.py -v
```

Overrides: `MINICPM_CXX` selects the compiler; `MINICPM_JSON_INCLUDE` points at a `nlohmann` header directory. Without the override, headers are discovered via `pkg-config nlohmann_json` or standard system include roots (`/usr/include`, `/usr/local/include`, `/opt/homebrew/include`) by `tools/host_validation/native_dependencies.py`; no personal path outside the repository is read. A machine with no headers skips only the JSON-dependent S600 drivers (with the reason recorded and the two full-driver-set helpers skipped rather than partially run), while an invalid override fails the suite.
