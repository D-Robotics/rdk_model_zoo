# B6 independent board review — in progress

Review: changes-required. Closed: no. The earlier host review remains historical evidence; it did not establish Python 3.10 board compatibility.

## B6-B1 — evaluator hashing requires an unavailable Python API

On X5 8GB and S100, GitHub checkpoint `73a6de135ddbcc343922a2a31e31c22729f09296` successfully downloaded each sample's exact encoder/decoder pair. Both EfficientSAM and MobileSAM then failed before inference in `samples/_shared/sam_evaluator.py:32`: `hashlib.file_digest` is unavailable in the board Python 3.10 environment, contrary to the documented Python 3.10+ requirement. Four independent run records reproduce the same failure. The earlier metadata projection repair has not yet been exercised by these runs.

The failure evidence also exposes a return-code inconsistency: persisted comparison.json says return_code=2 while the uncaught exception exits the actual CLI with 1. Preserve the failed report and make the public command's documented execution-error code agree with its record.

[Original execution and comparison records](evidence/2026-09-24-b6-initial-board/) include full stdout/stderr and the four failed reports. Model preparation succeeded; inference and numerical comparison did not run. No metadata/payload checks may be declared passed from these failures.

## Remediation and closure requirements

A local Claude Code + GLM task is addressing the exact compatibility and error-propagation issues in an isolated worktree. Require a regression under absence of file_digest, known SHA tests including multiple chunks, failure-report preservation and CLI rc coverage, followed by independent host review and GitHub-delivered reruns on the same two boards. Continue the outstanding X5 4GB/S100P/S600 target matrix after the tools run successfully; do not infer their status from X5 8GB/S100. Conversion, dataset quality and historical latency measurements are separate scopes.
