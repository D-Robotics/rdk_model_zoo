# Current-candidate skill behavior protocol — Codex

Status: queued, not executed or accepted. This extends deterministic checks with
actual fresh local Claude Code + GLM sessions; the reviewer will inspect every
case's tool trace and final response independently. The supervisor waits for
H8-SKILL-R1 author remediation to finish before copying the seven complete skill
source directories into an isolated snapshot. Each case gets a fresh session and
only its own skill directory; no skill is installed or changed in user directories.

Fourteen scenarios cover AG01 and AG03–AG15 from Spec section 8.7. AG02 is the
separate no-Agent README path already checked with native host commands and is
not replaced with an LLM conversation. Prompts are preserved under the evidence
directory. Case fixtures are explicitly synthetic; the target checkout is real.
The exact current source snapshot, model/CLI version, argv, timestamps, output and
source hashes will be retained before any grading. Historical 75-case runs are
not relabeled as current-candidate results.

Available tools are Read/Glob/Grep, plus Edit/Write for AG06's isolated one-line
fixture edit. No Bash, network, SDK, MCP or subagent tools are exposed. Explicit
skill-source loading tests routing/reading/reasoning and single-directory resource
use; it is not automatic Agent skill discovery, vendor tool execution, production
Hub installation, model validation or a baseline-efficiency comparison. A task
that cannot be proven under these tools remains not-run or insufficient rather
than receiving a fabricated pass. No whole-H8 closure follows from a process exit.

Runtime records live under `.coordination/20260928-skills-behavior/` outside the
checkout until reviewed and persisted. The supervisor's process state identifies
the exact active case/PID; observation timeouts must not restart a live case.

## 2026-09-29 execution follow-up

All fourteen initial sessions completed. [Independent grading and durable evidence](2026-09-29-skills-behavior-independent-review.md) retain three failed initial accuracy/reporting dimensions and three explicitly prompted corrections. Explicit-load minimum behaviors are bounded acceptance, not 84 eval definitions executed, automatic discovery, production installation or all-accuracy pass. The queued status above is the original protocol snapshot.
