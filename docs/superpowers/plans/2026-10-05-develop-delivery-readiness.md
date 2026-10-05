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

- [x] Reproduce hard-coded JSON/gflags and unconditional Linux iconv command behavior with meaningful failing tests; save logs.
- [x] Implement standard dependency discovery with validated overrides, no personal .coordination fallback. Use -liconv only where the platform requires it; retain real gflags in entry tests.
- [x] Exercise the resolver's explicit/standard/missing/invalid dependency branches and Linux/macOS flags; run both full LLM suites, no new silently skipped native coverage.
- [x] Codex review production compile commands, meaningful tests and dependency evidence; commit accepted owned paths.

## Task 2: Complete host verification and CI

**Owned files:** tools/host_validation/run.py, its orchestration tests and host requirements; .github/workflows/{host-validation,sample-contract,model-catalog-data}.yml; tools/host_validation README pairs. Do not edit Task 1 resolver/consumer files.
**Interfaces:** maintainer command python tools/host_validation/run.py --repo PATH --report PATH [--python PYTHON] executes native Sample Python suites plus nested conversion/evaluator suites, shared suites including the safe parent VLA integrity guard, affected tool/Skills tests, static contract, applicable native CTest and Catalog. ACT/Pi0 upstream gitlinks remain uninitialized and excluded. Reports include actual commands/counts/skip reasons/exit codes/source identity and logs. CTest wrappers and their underlying cases are separate counters, not falsely additive unique coverage.

- [x] Test discovery against fixture layouts, isolation of identical module names, no tests, failures, skipped native checks, timeout, missing historical objects and source changes; reproduce failures before implementation.
- [x] Implement deterministic isolated execution and structured reports outside source; enumerate all actual Python test directories and existing host-safe CTest projects, no vendor SDK build or download commands.
- [x] Declare repeatable core host dependency versions supporting Python 3.10/3.12. Install ONNX/ORT for synthetic graph tests; missing Torch/FunASR model export tests remain explicit optional scope, not fake successes.
  Independent fresh-env evidence: `local-execution/20261005-develop-delivery-readiness/independent-fresh-findings.md` (workspace outer directory). SciPy1.15.3 failed standalone on local macOS; SciPy1.17.1 imports successfully in Python3.12. Ensure version markers cover Python3.10 Linux separately. Include ftfy, regex and Pillow; do not skip CLIP/UNet/source-reference checks to hide missing dependencies.
- [x] Add Ubuntu 3.10/3.12 CI and applicable macOS host coverage with compiler/CMake/OpenCV/nlohmann-json/gflags dependencies and full-history checkout; enforce no skipped native coverage. Trigger equivalent gates for develop/main and PR. Update stale platforms workflow comments/paths.
- [x] Run runner unit tests and real full command, fix only identified issues; Codex independently verifies clean checkout/fresh Python deps and live CI after integration.

## Task 3: Customer documentation, source version and Catalog provenance

**Owned files:** README.md/README_cn.md, samples/README.md/README_cn.md, AGENTS.md/CLAUDE.md, VERSION, CHANGELOG.md (new candidate section only), docs/adr/0006, new docs/releases/unified-source-release.md and support matrix, tools/catalog-publisher/sources.json plus minimum source loader/tests changes. Do not edit workflows or test orchestrator.

- [x] Verify all current status claims against actual source/accepted reports; update active customer/Agent entry wording, preserve historical report bodies and real known gaps.
- [x] Document X5/S target and artifact selection, readable main/model chain and native/multistage exceptions, local dependencies and actual CI command; do not turn maintainer tools into inference prerequisites.
- [x] Add source VERSION 2.0.0 candidate, zoo-v tag naming and clear main promotion/rollback/check procedure; platform artifact and Skills versions remain distinct, no tag/Release/default-branch action.
- [x] Build support/verification matrix from actual declared Sample data, preserve not-run board dimensions; no blanket all-target approval or reclassification of known missing assets.
- [x] Write failing Catalog tests for worktree link_ref HEAD resolution to actual complete source commit and unavailable Git/invalid refs; preserve immutable historical links. Implement minimum source loader change and set worktree sources to HEAD.
- [x] Run complete Catalog check and documentation links/bilingual/contract checks; Codex assess current source/version and user readability, commit owned paths.

## Task 4: Independent closeout and develop delivery

- [x] Review all packages, fix material findings through bounded Claude tasks; confirm approved runtime architecture preserved and all native Sample directories covered.
- [x] Use an independent full-history clone and fresh Python dependencies; run the maintained verification command, all model-free entries from outside clone and source/evidence stability checks. Preserve first failures and exact scope.
- [x] Recheck remote/local develop and all checkout states; integrate without rewriting history, record source commit and local/remote hashes.
- [x] Execute CI on the exact develop snapshot, inspect every job including counts/skips/failures and correct regressions. Unobserved CI is not passed.
- [x] Write independent final evidence, complete requirements matrix and delivery report. Audit the full goal against actual develop and external state before declaring complete; no tag, formal release or default-branch switch.

## Independent review checkpoint (2026-10-05)

- Task 1 implementation plus bounded re-review accepted on macOS/Python3.12: resolver35, Gemma41, MiniCPM20, HIMLoco37, shared180; actual standard/pkg-config gflags and Darwin iconv link checks green. Linux remains for live CI. Accepted files are not committed yet.
- Task 3 matrix51 rows and source/asset/version invariants independently reviewed; Node22 full Catalog gate136 green after narrow workflow-list regression correction. Commit pending.
- All51 Sample architecture audited:49 direct-predict Python mains, native LLM exceptions preserved, production Runtime matches previously accepted source.
- Task 2 independent material findings assigned to `host_ci_rereview`: actual content drift, full applicable native extras, failure reports, strict Node/skip scope, missing Sample test inventory, parent integrity guard, SciPy markers. Final clean-clone/CI evidence is still pending.
- Exact external receipts are in `local-execution/20261005-develop-delivery-readiness/independent-package-review.md` and `native-independent-review/review-report.md` under the outer workspace; final publishable review will summarize them in this repository.
- Full native audit found six safe CTest projects, including Paraformer and HIMLoco with vendor SDK/production CLI explicitly OFF. The first four projects now pass 49/49; the two additional projects independently pass 6/6 and 1/1 and are assigned for registry inclusion. YOLOE Apple/OpenCV uses an explicit TBB link while retaining ASan+UBSan; an independent new build passes 11/11.
- First full fresh-Python runner receipt is deliberately retained as failed: 2286 tests, 5 failures and 50 errors across ByteTrack, GoogLeNet, MobileOne, YOLOE evaluator and Skills. Read-only triage identified mandatory lap/jsonschema dependency omissions, two whole-module-table restoration defects in source-comparison tests, and optional COCO metadata handling. Bounded Claude fixes are assigned; final source and CI are not accepted yet. The strict inventory must also fail if its accepted coverage record is missing or malformed.

## Final source review checkpoint (2026-10-06)

- Portable dependency consumers, Apple/OpenCV sanitized YOLOE tests, fresh-process OpenCV fixture imports and optional COCO metadata compatibility are independently accepted. Lower-version OpenCV4.12 and pycocotools2.0.10 probes pass; asset manifests, historical pins and VLA gitlinks remain unchanged.
- The maintained host gate includes all58 Python suites and all six host-safe CTest projects (56 cases), strict51 accepted-inventory checks, actual content hashing, supported Node engines and explicit optional-export scope. The frozen predecessor run passed2305 Python tests with zero failures/errors, two declared optional skips and one missing optional Torch export module, plus56 CTest cases and136 Catalog tests. This is a local dirty-checkout receipt, not final clean-clone/live-CI evidence.
- Independent review found typed CMake keys and padded false Boolean values could bypass protected vendor/production flags. Sequential Claude repairs reject both before configure/build;75 runner fixtures and35 dependency resolver tests pass. Final independent guard review and package commits are pending at this checkpoint.
- The CI preflight review found no provable first-run blocker. Linux3.10/3.12 and hosted macOS3.12 remain for actual CI. Root will record the committed clean-clone verification and real develop refs/run IDs in the final delivery evidence.

## Accepted packages entering committed verification (2026-10-06)

All five owned packages are accepted for commit. The final bounded repair classifies actual CMake false values, including case-sensitive NOTFOUND suffixes and trailing-cache whitespace semantics, while the protected vendor/production switches reject any padded value. Root independently ran78 runner fixtures and35 resolver tests, with stable owned-file hashes;14 previously recorded real-CMake observations matched the resulting scope classification. No material guard finding remains in the targeted independent review.

The complete committed clean-clone pass and actual develop CI are the remaining delivery gates. Their absence here is not a passed result; the final evidence record will bind them to the actual implementation commit. Board, real export/calibration/compiler work remain outside this accepted scope by user decision.

## Independent implementation closeout (2026-10-06)

The five accepted packages were committed, followed by bounded real-failure fixes:
`565ab9d8` executes suites from the requested repo, `24668498` builds Catalog before
its dependent Python tests, and `545a3b58` repairs three Linux test fixtures without
changing Runtime, support, pins or skips. Original failed clean-clone and CI
receipts remain intact; earlier checkpoints above retain their as-of status.

Implementation C=`545a3b5874ae723663d2c817ae9ef964bc495746` passed an independent
full-history clean clone with isolated declared dependencies:58 suites/2315 Python
checks (0F/0E,2 explicit optional skips,1 optional missing Torch module),6 CTest
projects/56 cases,Catalog136,contract51/0/87/0 and149 successful SDK-free final
entry checks with17 expected unsupported-target probes. Source digests and clean
Git states held; pins present,upstream gitlinks uninitialized.

C was fast-forwarded and pushed to actual develop; local/tracking/remote matched.
Its observed host run37360844437 passed all Linux3.10/Linux3.12/macOS3.12 jobs;
contract37360844575 and Catalog37360844432 passed. Downloaded reports and payload
hashes were independently verified. Scope, requirements and evidence live in
[delivery review](../../releases/2026-10-06-develop-delivery-review.md) and its JSON.

This record binds implementation C. The subsequent documentation-only closeout
must pass the same workflows on its actual Develop HEAD before final delivery;
earlier C CI is not substituted for that snapshot. Board/export/compiler scopes
stay deferred by user decision. No main/tag/Release/default-branch/site action.
