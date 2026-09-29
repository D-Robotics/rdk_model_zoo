# B11 Gemma explicit model preparation — author record

Base `30db12bc`; independent acceptance remains pending. No models were downloaded,
no board was contacted and no quantization recipe was executed or revalidated.

## Change

The source downloader inferred the SoC and fell back to S100P. It ignored all
arguments, so even `--dry-run` performed transfers. Successful empty transfers
were renamed to final filenames. Six host tests with a fake `wget` reproduce five
failures against the original script and pass after the change.

Preparation now requires explicit `GEMMA4_SOC=s100|s100p|s600`. `--dry-run` prints
reuse/download decisions without network or filesystem mutations. `--help` prints
usage, unknown arguments fail, and empty transfers do not publish final files.
S100 still supports locally supplied HBMs or a custom URL; source archive URLs,
file names, resume behavior and reuse of existing nonempty files are preserved.

Ruling: remove implicit target discovery/default from the preparation command.
The download computer may not be the inference board. Existing callers that omitted
GEMMA4_SOC must now supply it; runtime board identity checking remains separate.

Six bilingual root/model/C++ README files now show target-specific data directories.
The model guides document every environment override, dry-run, dependencies,
resume/error handling, S100 prerequisites and the limits of nonempty-file reuse.
Source HBM hashes remain attributed reference values, not claimed automatic checks.

## Validation and remaining scope

[Evidence](evidence/2026-09-28-b11-gemma-preparation/) contains baseline failures,
code hashes, 20 passing Sample tests and migration contract output (50 samples,
zero violations, 51 policy skips, zero exemptions). `bash -n` and `git diff --check`
also passed. Host tests replace wget with a script writing tiny fixture files;
these are orchestration tests, not archive availability or artifact verification.

Commands: `python -m unittest discover -s samples/llm/gemma4-e2b/tests -v` and
`python tools/sample_contract/check.py --scope migration --format text`, using
`../rdk_model_zoo/.venv/bin/python` from the worktree root.

Text's complete tensor/stage contract, third-party preparation, MiniCPM, the other
H0–H9 items and whole-branch independent review remain open. Board tests are not-run.
