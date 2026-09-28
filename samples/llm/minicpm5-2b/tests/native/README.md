[English](README.md) | [简体中文](README_cn.md)

# Native core host tests

C++ drivers under this directory compile the production sources against the SDK test doubles in `fixtures/` and run them on the host. The doubles are marked at the top of each header and are **not** the vendor SDK: they perform no tokenization, no BPU work and no generation, and they never represent board evidence. `tests/test_native_core.py` compiles and runs them; the legacy CLI scenarios additionally execute the real `src/main.cc`.

Covered behavior: legacy single-use lifecycle (R1 regression), request construction and greedy sampling parameters, template size limits, S600 temporary-configuration cleanup on success/error/exception paths (R3 regression), metric finite/nonnegative validation with zero preserved (R2 regression), stage boundaries and two-turn conversation fields, and test-harness isolation (CORE-R4 regression): every S600 driver creates its own atomically unique `mkdtemp` scratch directory (`scratch_dir.hpp`), removes only that owned directory, restores `TMPDIR` including an initially unset state, and the suite verifies a pre-existing `model/` sentinel under a hostile `TMPDIR` survives all drivers plus concurrent same-`TMPDIR` runs of the compiled binaries.

```bash
../rdk_model_zoo/.venv/bin/python -m unittest discover -s tests -v
```

Overrides: `MINICPM_CXX` selects the compiler; `MINICPM_JSON_INCLUDE` points at a `nlohmann` header directory.
