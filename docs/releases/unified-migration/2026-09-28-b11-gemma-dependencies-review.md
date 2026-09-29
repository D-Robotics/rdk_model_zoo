# B11 Gemma dependency preparation — author record

Base `4b9ae7e0`; whole-branch independent review and H0–H9 remain open.
No remote dependency, model or toolchain was downloaded; no board or quantization
process was run. Installer tests use temporary local Git repositories.

## Change and rationale

The original installer deleted an incomplete destination, reused existing CMake
files without checking their source revision, and could install Rust automatically.
Its lockfile patch also depended on GNU `sed -i`. The baseline test log reproduces
loss of a fixture's local file, unexpected attempted Rust download, and portability
failures rather than claiming every red result is a distinct product defect.

Preparation now checks explicitly installed stable Rust >=1.80 and Git, supports
an offline `--dry-run`, and never installs a toolchain. A new checkout is prepared
in its own temporary directory, pinned and checked before moving into place.
Existing destinations are preserved and must match the root commit, expected
CMake files, recursive submodule pins and clean working content. Only the exact
documented Cargo.lock format-4-to-3 patch is permitted. Failed clones clean up
only their own staging directory. Existing symlink destinations are rejected.

Ruling: dependency preparation no longer bootstraps Rust or resets unknown source
trees. Users prepare a stable Rust toolchain explicitly and preserve/relocate their
own checkout before retrying. This costs an explicit prerequisite step but matches
separate preparation/build/run responsibilities. Source commit and dependency
versions remain unchanged; the lockfile patch uses portable awk/temp-file handling.

## Evidence and README

[Evidence](evidence/2026-09-28-b11-gemma-dependencies/) contains baseline and current
logs plus code hashes. Eight installer cases use real local Git operations,
including a pinned gitlink with a deliberately modified submodule. They cover
install/reuse, modified/wrong/incomplete existing source, old Rust, dry-run and
failed clone cleanup. Rust version reporting is a double; no actual Rust build
was performed. Curl in the fixture refuses network access.

The complete Sample suite passes 28 tests; migration contract remains 50 samples,
zero violations, 51 policy skips and zero exemptions. Shell syntax and diff checks
pass. Four third_party/C++ bilingual guides now explain explicit toolchain setup,
pins/local patch, preview, failure preservation, proxy settings and build boundary.

Commands use `../rdk_model_zoo/.venv/bin/python -m unittest discover -s
samples/llm/gemma4-e2b/tests -v` and the same Python with
`tools/sample_contract/check.py --scope migration --format text` from the worktree.

Real remote checkout availability and vendor/native compilation are not established
by these local fixture results. Text's full tensor/stage contract, MiniCPM and the
remaining completion-plan work continue; board evidence stays not-run.
