# B11 MiniCPM native core refactor, R1–R5 closure and author audit — author record

Implementer: Claude Code + GLM. Working tree on
`codex/b7-board-integration-20260924` (package diff kept uncommitted for
review; reviewer instructions are
[the baseline](2026-09-28-minicpm-reviewer-baseline.md), whose findings and
evidence are preserved unchanged). This is an author record: **author
complete, pending Codex independent review.** No board, no SDK, no model
download, no quantization/export/calibration run; the prior launch and README
packages' fixes were not overwritten.

## Reviewer findings closed

**R1 — legacy request state survived reinitialization (P2).** The legacy
runtime now enforces the documented single-use lifecycle: `predict()` sets a
`finalized_` flag, and both `init()` and `predict()` on a spent instance throw
`Request already completed; create a new MiniCPM5 instance`. A second request
can never silently miss END because it can never start on the same instance;
the source's alternative of resetting state was not taken, so no
conversation/multi-request capability was added. Regression
(`tests/native/legacy_lifecycle.cpp`) replays the reviewer's order: first
request streams text, ends with END and exits 0; reinitialization is rejected;
no second `xlm_infer` call occurs; missing-END, ERROR-state, destroy-failure
and nonzero-infer-status cases all still report exit 1 with their status
fields.

**R2 — S600 accepted non-finite/negative metrics (P2).** New
`minicpm5::validate_metrics(ttft, decode_tps, e2e)` runs in post-processing
before any value reaches `Result` or CLI JSON: non-finite or negative values
throw a `std::runtime_error` naming the offending metric, so JSON can no
longer serialize an invalid measurement as null success. Zero is preserved —
the documented one-token length-limited request still succeeds with
`decode_tps == 0` and status 6. Values are never coerced to zero. Regression
(`tests/native/s600_metrics.cpp`) covers invalid TTFT, decode and e2e
separately (NaN, negative, infinite) plus zero and fully valid passthrough.

**R3 — runtime configuration was not exception-safe (P2).** Model-file
validation, OELLM JSON settings and the `mkstemp` configuration file moved out
of the inference file into `runtime/cpp/src/runtime_config.cc`. The temporary
descriptor and path are owned by an RAII `TemporaryFile` whose destructor
closes the descriptor and removes the file on every exit path; `write()` now
loops over short writes. Regression (`tests/native/s600_config_cleanup.cpp`)
asserts zero leftover `minicpm5-*` temp files after success, after an SDK
error return and after an injected C++ exception from `Init`, plus the
missing-model-file guard. The vendor API is not assumed to throw in normal
operation; the exception is injected by the test double only.

## Stage separation

S600 (`runtime/cpp`): `pre_process` (prompt validation and OELLM request
construction; no SDK calls), `infer` (one synchronous runtime call plus
response-shape validation) and `post_process` (text/token/status extraction,
status whitelist 3/4/6, metric fetch and `validate_metrics`) are public free
functions matching the repository C++ free-function contract; `Generate`
chains them (the Predict analogue). Configuration/file IO lives in
`runtime_config.cc`; gflags parsing and JSON RESULT rendering stay in
`main.cc`. Tokenization and template rendering remain inside the runtime —
the SDK exposes no tokenization API, so no public tokenization stage is
invented.

Legacy (`runtime/legacy`): `load_chat_template` (template file IO and the
empty/65535-byte size bounds) lives in `chat_template.cc`, outside the
inference file; `prepare_request` builds the fixed request (request_id=0,
`XLM_INPUT_PROMPT`, `new_chat=true`, `XLM_INFER_BACKEND_BPU_ANY`, owned
strings) as pure pre-process; `predict()` chains template load, request
build, one `xlm_infer` call and the single-use teardown, returning a
`RequestOutcome` (raw `sdk_status`, `ended`, `failed`, `destroy_status`) whose
`exit_code()` maps to the exact source statuses; `main.cc` renders the
unchanged `\nRESULT status= ended= failed= destroy=` line. Streaming token
printing stays inside the SDK callback because the vendor API invokes it
asynchronously; end-state text suppression is preserved verbatim.

## Capability preservation

Both SDK families and their assets remain strictly separate (1.0.0 `xlm` vs
2.0 beta `oellm_runtime_basic`). Preserved exactly: greedy sampling parameters
(temp 0, top_k 1, top_p 1, min_p 0, neutral penalties; captured by
`legacy_request.cpp`), context 4096 and `XLM_MODEL_TYPE_DEEPSEEK`, the
`--help`/option-parsing and error messages, template size limits, S600
two-turn conversation (`request_id` advancing 1→2, `new_chat` per flag,
`conversation_id` 1; captured by `s600_stages.cpp`), `max_new_tokens` bounds
1–4096, response/status/metric error mapping, CLI RESULT formats and exit
statuses, the launcher and its eight tests, and all historical evaluation
records: S100/S100P PPL 17.91995 (+27.83%, fails ≤3%) with 2/6 text matches,
S600 PPL 14.2428 (+1.60%), still attributed to pinned source `380e1a2` and
not represented as new results. No parameter, session capability, output,
evaluation capability or evidence was removed.

## Navigation and manifest sync (R4/R5)

Root `README.md`/`README_cn.md` and `samples/README.md`/`README_cn.md` now
state 51 unified samples (45 vision, three speech, one robotics, two LLM),
link MiniCPM bilingually in the task tables and the LLM index with its
in-progress status, and distinguish Gemma's completed model preparation from
its still-open text-generation core work and independent acceptance. Native
core acceptance is not implied anywhere. `docs/release/s/models.yaml`'s
MiniCPM note now describes the actual unified-path migration state and
pending board tests; all archive URLs, filenames and SHA-256 digests are
byte-identical (verified structurally; see evidence `checks.txt`). No other
sample's manifest or index rows were touched.

Final interface documentation was synced: both runtime README pairs describe
the implemented stage APIs, the RAII temporary-file behavior, the metric
rules and the single-use legacy lifecycle; `tests/native/README.md`/`README_cn.md`
document the host doubles and their explicit non-SDK/non-board status.

## Host evidence

Commands (run on the host; no SDK, weights or board involved):

- `../rdk_model_zoo/.venv/bin/python -m unittest discover -s samples/llm/minicpm5-2b/tests -v`
  — 16 tests, all pass: 8 launcher tests (unchanged) + 5 native drivers + 3
  legacy CLI tests compiling the real `main.cc` against the doubles
  ([output](evidence/2026-09-28-b11-minicpm-core-refactor/native-core-tests.txt)).
- `../rdk_model_zoo/.venv/bin/python tools/sample_contract/check.py --scope migration --format text`
  — 51 samples, 0 violations, 51 policy skips, 0 exemptions
  ([output](evidence/2026-09-28-b11-minicpm-core-refactor/contract-check.txt)).
- `git diff --check` clean; black reformatted the new Python test file;
  clang-format (Google style) applied to all new/rewritten C++ files;
  legacy CMakeLists re-configures cleanly with the added source list.
- File hashes for this package:
  [changed-files.sha256](evidence/2026-09-28-b11-minicpm-core-refactor/changed-files.sha256).

## Unverified boundaries and scope notes

Native code is exercised only through host test doubles that share API shapes
but perform no inference; real SDK behavior, model loading and board
execution remain not-run. The write-failure branch of `TemporaryFile::write`
is covered by RAII structure, not by an injected short-write test. The S600
`main.cc` (gflags) could not be host-compiled because the host has no gflags;
it is unchanged apart from relying on the same header API the drivers compile.
During this package, Codex landed reviewer-doc commits on the branch and a
concurrent readme-pair package left its own uncommitted changes in the shared
worktree; both were left untouched and are outside this report's hashes.

## Remediation round — CORE-R1/R2/R3 and docs follow-through (2026-09-28)

After [the independent review](2026-09-28-minicpm-core-independent-review.md)
returned changes-required, this round fixed all findings; reviewer reports and
evidence were not modified. Evidence files under
`evidence/2026-09-28-b11-minicpm-core-refactor/` now reflect the
post-remediation state; the sections above describe the first round.

**CORE-R2 (fixed first, unblocking the concurrent PointNet suite).** The
manifest note's `path: the B11` colon sequence made
`docs/release/s/models.yaml` a mapping-values ScannerError repository-wide.
The sentence was rewritten to plain prose with no colon; `yaml.safe_load`
passes and a real manifest-backed resolution confirms the MiniCPM
`sample_path`/`download_scripts` and all three archives with byte-identical
URLs and SHA-256 digests
([checks.txt](evidence/2026-09-28-b11-minicpm-core-refactor/checks.txt),
exit 0).

**CORE-R1.** `PreparedRequest` is now a self-rebinding carrier: implemented
copy/move constructors and assignments call `bind()`, which points
`request.prompt`/`request.chat_template`/`input.requests` at the instance's
own storage; a moved-from carrier is also rebound so it never keeps dangling
pointers into transferred buffers. `prepare_request` keeps its exact public
signature. Regression `legacy_prepared_ownership.cpp` compiles with
`-fno-elide-constructors` and covers non-elided return, copy/move
construction and assignment, function-boundary copy/move, short (SSO) and
long (heap) strings, independence of copies and self-consistency of
moved-from instances. The reviewer's `prepared_ownership.cpp` driver run
verbatim now reports `return pointers owned=1 copy pointers owned=1 move
pointers owned=1`, exit 0. The embedded-view lifetime rule ("pass
`&request.input` only while the instance is alive and unmodified") is
documented in the header and both legacy READMEs.

**CORE-R3.** The legacy inference file no longer includes console IO:
`MiniCPM5Config::text_sink` (`std::function<void(const char*)>`, default
empty = silent, console-free library) receives each streamed chunk; `main.cc`
injects the stdout sink. END- and ERROR-state text suppression is preserved
verbatim, the exit-code mapping is unchanged, and the single-use lifecycle
holds. A throwing consumer is contained inside the C callback boundary — the
exception never escapes the vendor callback; streaming stops, the request
completes with its normal status mapping, and `RequestOutcome::stream_error`
records the failure (surfaced by `main.cc` as a stderr warning). Regression
`legacy_stream_sink.cpp` covers the custom sink, silent no-sink use,
END/ERROR-carried text suppression, incremental multi-chunk delivery within
one infer call and the contained consumer failure.

**Public API / documentation follow-through.** The S600 `pre_process` now
enforces its own public argument contract inside the function — an empty
prompt or `max_new_tokens` outside 1–4096 throws for every caller,
independent of the constructor check — with `s600_stages.cpp` covering the
bounds and boundary values. The six MiniCPM README banners no longer claim
"host launch orchestration only"; they now state the round covers launch
orchestration plus the native core refactor with host-side SDK-double tests,
keeping historical board claims unchanged. Both runtime README pairs gained a
complete native library usage example (define variables, initialize, call,
consume results/status) matching the final interfaces; the legacy pair
documents the sink, containment and carrier lifetime. S600 two-turn
behavior, the S100/S100P PPL +27.83% / 2-of-6 failures and the S600 +1.60%
history are unchanged and still attributed to pinned source `380e1a2`.

Verification after remediation (full commands and exit codes in
[native-core-tests.txt](evidence/2026-09-28-b11-minicpm-core-refactor/native-core-tests.txt)
and [checks.txt](evidence/2026-09-28-b11-minicpm-core-refactor/checks.txt)):
18/18 unittest tests pass, exit 0 (8 launcher + 7 native drivers incl. the
two new ones + 3 legacy CLI); contract checker 51 samples / 0 violations /
51 skips / 0 exemptions, exit 0; `git diff --check` exit 0; manifest parse
and asset resolution exit 0; reviewer ownership driver exit 0. black and
clang-format (Google) applied to touched files.

## Remediation round — CORE-R4 test-harness isolation (2026-09-28)

Codex's follow-up review found that the native drivers themselves reused and
deleted shared temporary paths: `s600_metrics.cpp` and `s600_stages.cpp`
wrote fixtures at `temp_directory_path()/model` and removed that directory
afterwards, and `s600_config_cleanup.cpp` used fixed scenario directory
names — so a direct driver run could delete unrelated files (the reviewer's
`test-temp-isolation.json` shows a sentinel under a reviewer-owned
`model/` removed by the actual metrics binary, exit 0), and concurrent runs
could corrupt one another's fixtures.

New `tests/native/scratch_dir.hpp` fixes this at the driver level, not only
in the Python wrapper: `ScratchDir` allocates one atomically unique
directory with `mkdtemp`, points `TMPDIR` at it for its scope, and on
destruction removes only that owned directory and restores the previous
`TMPDIR` — including restoring an initially unset state via `unsetenv`.
Model fixtures are written inside the owned scratch directory only. All
three S600 drivers were rewritten onto it; the legacy drivers already used
`mkstemp` files they own and remove individually. `s600_config_cleanup.cpp`
additionally asserts in-driver that the `TMPDIR` state at exit equals the
state at start (set or unset).

Regressions added to `test_native_core.py` and exercised by directly
invoking the compiled binaries (no wrapper reliance):
`test_drivers_preserve_parent_temp_sentinel` runs every driver with `TMPDIR`
pointing at a freshly created directory holding `model/sentinel.txt` and
requires success plus an intact sentinel, and
`test_s600_drivers_isolate_concurrent_runs` starts five S600 driver
instances sharing one `TMPDIR` and requires all to succeed. A RED
sensitivity check (throwaway program mimicking the pre-fix
`temp_directory_path()/model` + `remove_all` pattern, run only inside a
fresh isolation directory) confirms the sentinel check catches the old
behavior; the reviewer's original reproduction stands as the RED evidence
and was not modified. All demonstrations used newly created directories; no
pre-existing user temp paths were touched.

Verification ([native-core-tests.txt](evidence/2026-09-28-b11-minicpm-core-refactor/native-core-tests.txt)
and [checks.txt](evidence/2026-09-28-b11-minicpm-core-refactor/checks.txt),
exit codes inline): 20/20 unittest tests pass (18 prior + the two new
isolation regressions), exit 0; the three S600 binaries compiled and run
directly against a hostile `TMPDIR` keep the sentinel (rc=0 each); five
concurrent instances rc=0; contract checker 51/0/51/0 exit 0;
`git diff --check` exit 0; manifest `yaml.safe_load` exit 0; reviewer
`prepared_ownership.cpp` rerun exit 0. CORE-R1–R3 fixes, public API and
documentation changes from the previous rounds are unchanged.

## Remediation round — CORE-N1 README example self-containment (2026-09-28)

Codex's recheck passed CORE-R1–R4 (20 tests, the original pointer
counterexample at owned=1/1/1, and a TMPDIR-unset direct driver run) and
raised one documentation finding: the legacy runtime README C++ examples
used `std::cout`/`std::flush` without `<iostream>`, and their statements sat
outside any function, so extracting the block verbatim and compiling failed.

All four runtime README examples (legacy and S600, English and Chinese) are
now complete, self-contained programs — legacy adds `<iostream>` and wraps
the flow in `main()` ending with `return outcome.exit_code();`; the S600
pair wraps its flow in `main()` returning from the observed statuses. No
runtime code changed. The actual first `cpp` block of each README was
extracted verbatim into
[readme-examples/](evidence/2026-09-28-b11-minicpm-core-refactor/readme-examples/)
and syntax-compiled against the production headers plus the existing SDK
doubles with `-Wall -Wextra -Werror`; all four return 0 with the exact
commands recorded in
[readme-example-compile.txt](evidence/2026-09-28-b11-minicpm-core-refactor/readme-example-compile.txt).
`git diff --check` exit 0. The 20-test suite was not rerun per the bounded
scope; reviewer verdicts stay as they are.

## Status

Author complete for the CORE-N1 README-example remediation. **Independent
review remains with Codex**; H7/B11 are not closed from the author's green
tests. Gemma remaining Text work, H8 whole-repo integration, H9 full
regression and whole-branch independent review stay open; no further package
was started.
