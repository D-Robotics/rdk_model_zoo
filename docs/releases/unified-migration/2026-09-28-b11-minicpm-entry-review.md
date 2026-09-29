# B11 MiniCPM source migration and explicit entry — author record

Base `665cc61b`. MiniCPM is in progress; this is not full refactor acceptance or
independent review. Board/native SDK/model execution is not-run. Quantization
recipes are trusted source material and were not rerun.

## Source and preserved capabilities

All 62 source files were checked byte-for-byte against S pin
`380e1a2bf42041af54be6f34935e50197cfadff9` before migration. Source executable bits
are preserved. [Evidence](evidence/2026-09-28-b11-minicpm-entry/source.json) records
all source hashes; checks.json separates unchanged files from edited entry/docs.
Conversion, evaluator, test data and native inference core were preserved in this
round, including S100/S100P PPL 17.91995 (+27.83%, fails <=3%) and S600 PPL 14.2428
(+1.60%). Historical SDK forecasts and board results are explicitly attributed to
the source, not represented as current release facts or fresh tests.

## Entry changes

One standard-library Python launcher selects `legacy` for S100/S100P and `cpp`
for S600. The SDK families remain separate: 1.0.0 vs the source 2.0 beta package.
Model preparation is explicit, `--build` invokes only CMake, and ordinary execution
requires an existing binary. No implicit model download, installation or build.
Actual board identity is checked before build/run; host preview is read-only JSON.
Native arguments after `--` retain their boundaries; source SDK library/L2M settings,
legacy timeout and process exit statuses are preserved. S600 still supports the
source optional second prompt, and legacy remains a single-request native CLI.

Ruling: per-target build directories and explicit CMake target replace S600 CMake's
unconditional sysfs read. The launcher owns actual execution identity checking;
direct CMake configuration declares its intended target and does not certify a
board. Manual native invocation remains documented. Existing run.sh native flags
must now follow `--`; model preparation and build become explicit preceding steps.

Eight bilingual root/runtime guides now describe this split and source boundaries.
The source details, dependency requirements, parameter tables, memory configuration,
performance limitations and precision failure statements remain present.

## Host evidence and outstanding work

Eight launcher tests pass: target/backend selection, exact native argv/environment
and return code via a small local executable, build-only plan, real and injected
host rejection, missing binary, legacy artifact/timeout and invalid option scopes.
No native SDK execution occurs. Command:
`../rdk_model_zoo/.venv/bin/python -m unittest discover -s samples/llm/minicpm5-2b/tests -v`.

Migration checker now includes 51 samples and reports **70 violations**, all missing
MiniCPM README section anchors, with 51 policy skips and zero exemptions. This is an
open documentation-contract task, not a passed gate; no checker rule was weakened.
The complete source README content remains available while the systematic structure
is being finished. Core stage separation/resource review, model integration, all
README layers, complete host checks and independent review still remain. H0–H9 is open.
