# Develop Delivery Readiness Implementation Plan

> Execution: local Claude Code + GLM implements bounded packages; Codex dispatches, independently reviews/tests, and integrates Git. Preserve the already authorized execution method.

**Goal:** Deliver the completely refactored develop with repeatable host/CI gates, accurate customer documentation and a main promotion procedure requiring no runtime rewrite.
**Architecture:** Keep the approved Sample structure. Add maintainer-only test orchestration and portable native test dependency discovery. Preserve separate source, artifact, Skills and board evidence lifecycles.
**Tech Stack:** Python unittest (3.10/3.12 CI), C++17/CMake/CTest and host test doubles, Node 22.12–22.x/Vitest, GitHub Actions.
**Spec:** [design](../specs/2026-10-05-develop-delivery-readiness-design.md).

## Global Constraints

- No SSH/server work, model weights, real export/quantization/toolchain/board execution or VLA initialization.
- 51 native Samples remain in scope; no new framework/global inference CLI or active platforms copies.
- Preserve all asset URLs/checksums/Benchmark values, trusted recipes, source pins, historical tags and legacy compatibility.
- Executors edit only owned paths; no staging/commit/push/reset. Codex commits accepted packages by path and integrates develop.
- Full Git history is a declared maintainer prerequisite for pinned source comparisons; do not silently skip when a pin is missing.

## Review Focus

- New checkout without personal .coordination/Homebrew paths: standard declared dependencies suffice and missing prerequisites fail explicitly.
- Linux iconv/gflags vs macOS: compile flags reflect platform and dependency discovery; ABI/model results stay unclaimed.
- Nested test directories, identical test module names, zero tests, timeout and changing source: report actual isolated results and reject false success.
- main/develop branch changes: equivalent CI gates and immutable generated links; preserve historical data versions.
- Customer/Agent reading current root/index: no stale scope claims, hidden runner requirements or unsupported target defaults.

## Task 1: Portable native host dependencies

**Owned files:** tools/host_validation/native_dependencies.py and test_native_dependencies.py; samples/llm/{gemma4-e2b,minicpm5-2b}/tests Python helper consumers and relevant runtime/tests README pairs.
**Interfaces:** expose json_include_dir(override=None) -> Path, iconv_link_flags(system=None) -> list[str], gflags_compile_flags() -> tuple[list[str], list[str]]. Header overrides GEMMA_JSON_INCLUDE/MINICPM_JSON_INCLUDE remain compatible; gflags may use documented explicit env overrides or pkg-config/standard system paths.

- [ ] Reproduce hard-coded JSON/gflags and unconditional Linux iconv command behavior with meaningful failing tests; save logs.
- [ ] Implement standard dependency discovery with validated overrides, no personal .coordination fallback. Use -liconv only where the platform requires it; retain real gflags in entry tests.
- [ ] Exercise the resolver's explicit/standard/missing/invalid dependency branches and Linux/macOS flags; run both full LLM suites, no new silently skipped native coverage.
- [ ] Codex review production compile commands, meaningful tests and dependency evidence; commit accepted owned paths.

## Task 2: Complete host verification and CI

**Owned files:** tools/host_validation/run.py, its orchestration tests and host requirements; .github/workflows/{host-validation,sample-contract,model-catalog-data}.yml; tools/host_validation README pairs. Do not edit Task 1 resolver/consumer files.
**Interfaces:** maintainer command python tools/host_validation/run.py --repo PATH --report PATH [--python PYTHON] executes native Sample Python suites plus nested conversion/evaluator suites, shared excluding VLA, affected tool/Skills tests, static contract, applicable native CTest and Catalog. Reports include actual commands/counts/skip reasons/exit codes/source identity and logs. CTest wrappers and their underlying cases are separate counters, not falsely additive unique coverage.

- [ ] Test discovery against fixture layouts, isolation of identical module names, no tests, failures, skipped native checks, timeout, missing historical objects and source changes; reproduce failures before implementation.
- [ ] Implement deterministic isolated execution and structured reports outside source; enumerate all actual Python test directories and existing host-safe CTest projects, no vendor SDK build or download commands.
- [ ] Declare repeatable core host dependency versions supporting Python 3.10/3.12. Install ONNX/ORT for synthetic graph tests; missing Torch/FunASR model export tests remain explicit optional scope, not fake successes.
- [ ] Add Ubuntu 3.10/3.12 CI and applicable macOS host coverage with compiler/CMake/OpenCV/nlohmann-json/gflags dependencies and full-history checkout; enforce no skipped native coverage. Trigger equivalent gates for develop/main and PR. Update stale platforms workflow comments/paths.
- [ ] Run runner unit tests and real full command, fix only identified issues; Codex independently verifies clean checkout/fresh Python deps and live CI after integration.

## Task 3: Customer documentation, source version and Catalog provenance

**Owned files:** README.md/README_cn.md, samples/README.md/README_cn.md, AGENTS.md/CLAUDE.md, VERSION, CHANGELOG.md (new candidate section only), docs/adr/0006, new docs/releases/unified-source-release.md and support matrix, tools/catalog-publisher/sources.json plus minimum source loader/tests changes. Do not edit workflows or test orchestrator.

- [ ] Verify all current status claims against actual source/accepted reports; update active customer/Agent entry wording, preserve historical report bodies and real known gaps.
- [ ] Document X5/S target and artifact selection, readable main/model chain and native/multistage exceptions, local dependencies and actual CI command; do not turn maintainer tools into inference prerequisites.
- [ ] Add source VERSION 2.0.0 candidate, zoo-v tag naming and clear main promotion/rollback/check procedure; platform artifact and Skills versions remain distinct, no tag/Release/default-branch action.
- [ ] Build support/verification matrix from actual declared Sample data, preserve not-run board dimensions; no blanket all-target approval or reclassification of known missing assets.
- [ ] Write failing Catalog tests for worktree link_ref HEAD resolution to actual complete source commit and unavailable Git/invalid refs; preserve immutable historical links. Implement minimum source loader change and set worktree sources to HEAD.
- [ ] Run complete Catalog check and documentation links/bilingual/contract checks; Codex assess current source/version and user readability, commit owned paths.

## Task 4: Independent closeout and develop delivery

- [ ] Review all packages, fix material findings through bounded Claude tasks; confirm approved runtime architecture preserved and all native Sample directories covered.
- [ ] Use an independent full-history clone and fresh Python dependencies; run the maintained verification command, all model-free entries from outside clone and source/evidence stability checks. Preserve first failures and exact scope.
- [ ] Recheck remote/local develop and all checkout states; integrate without rewriting history, record source commit and local/remote hashes.
- [ ] Execute CI on the exact develop snapshot, inspect every job including counts/skips/failures and correct regressions. Unobserved CI is not passed.
- [ ] Write independent final evidence, complete requirements matrix and delivery report. Audit the full goal against actual develop and external state before declaring complete; no tag, formal release or default-branch switch.
