# B10 Paraformer publication and SDK binding — host verification

Status: SDK adapter implemented and host-tested; actual SDK/board inference and
complete Paraformer sample acceptance remain **not-run/open**. No OE, CER or
latency result is claimed.

## Selection and tensor contracts

`model_binding.py` selects only the exact three S100 HBM records in the active
`docs/release/s/models.yaml`. Auxiliary JSON/MVN/YAML assets cannot be mistaken for
models. External paths require a complete map of matching stage-specific asset
IDs. X5/S100P/S600, incomplete maps, mixed/relabelled stages and undeclared default
path changes are rejected. Full selection validation happens before any SDK object
is created; the first implementation did this too late, and the retained
`preload-red.log` documents the counterexample and subsequent fix.

Each file must expose exactly one model. Binding uses explicit names, shapes and
dtypes from pinned source extraction and native lookup, with only the documented
Python acoustic alias `shape_8609`. It is insensitive to metadata order. The
optional decoder `token_num` output is validated as int32 `[1]` when present;
otherwise only logits is allowed. Unknown/ambiguous/extra tensors and mismatched
geometry/dtypes fail. The source INT16 model designation is not treated as proof
of integer physical I/O: all feature/logit tensors require float32, count int32.
Real published HBM metadata has not been observed on a board during this work.

## Shared execution and scheduling

`runtime.py` creates three shared `NamedArrayRunner` instances and maps their bound
roles into the existing pipeline. The production path enforces local target and
asset-file validation before SDK construction. An explicit runtime factory is a
host-test injection seam only. Source archives and the shared runner are unchanged.

The source method `set_scheduling_params` discarded both arguments. The unified
bundle delegates them to all three models through the shared runner's validated,
model-keyed dictionaries. It prechecks that all setters exist, preventing partial
mutation for a known missing setter. Values follow the shared priority/core-list
contract. A later SDK execution error propagates; rollback across already changed
SDK objects is not promised. No-argument scheduling is a no-op.

## Verification and documentation

Eight binding/adapter tests plus the prior thirteen CPU tests pass: **21** sample
tests before package checks; two preparation tests bring the total to **23**. They cover exact manifest identity, alternate paths, tensor reorder,
geometry/dtype/name rejection, optional count/alias, pre-SDK rejection, and the
real shared transport composing all three synthetic SDK model calls. Scheduling
arguments are observed at each SDK boundary and a missing setter rejects before
any model changes. This is not an HBM inference test.

The first shared run failed `test_download_scripts_of_unified_samples_exist_on_disk`
because the newly present sample lacked its manifest-listed downloader. The failure
is retained in `shared-red.log`; it was fixed by implementing the complete six-file
preparation flow, not by excluding the sample. The shared suite then passes **156** tests. Both README languages describe exact
publication IDs, physical tensor names, external-path prerequisites, scheduling
semantics and failure behavior. Eight bash blocks are executed: CIF, tests,
synthetic pipeline and host-safe selection in each language. The board integration
Python sketch is intentionally not executed; it explicitly requires board SDK,
local models and prepared frontend features. All local README links resolve.

Reproduce the host document/sample/shared checks with Python, NumPy and PyYAML:

```bash
python docs/releases/unified-migration/evidence/2026-09-28-b10-paraformer-binding/verify.py
```

[Evidence summary](evidence/2026-09-28-b10-paraformer-binding/summary.json),
[sample log](evidence/2026-09-28-b10-paraformer-binding/green.log),
[shared log](evidence/2026-09-28-b10-paraformer-binding/shared.log).

## Remaining work

Real FunASR frontend with RNG isolation and source equivalence; model/auxiliary
file preparation and report identities; complete audio/manifest CLI without input
manifest mutation; all conversion/calibration and evaluator flows; source native
C++ capability; full root/model/conversion/evaluator/test-data bilingual READMEs.
No B10 or H0–H9 closure. Board testing remains deferred by the user.

## Explicit package preparation

The active manifest's existing `download_model.sh` entry now delegates to a Python
preparer: four remote files through the shared atomic download helper, two pinned
source frontend files through checked, non-overwriting installation. Both source
auxiliaries are copied byte-for-byte. Vocabulary content is pinned to the observed
published package, separate from the unchanged publisher-null manifest hashes.
Preview lists six sources/destinations without writes/downloads. Tests substitute
HTTP response bytes only, verify complete package layout, rerun reuse and preserved
user-modified files on rejection. Full HBM downloads remain unexecuted in this step.
Model-level bilingual READMEs explain all flags, defaults, paths, hash provenance,
partial failures and validation boundaries.
