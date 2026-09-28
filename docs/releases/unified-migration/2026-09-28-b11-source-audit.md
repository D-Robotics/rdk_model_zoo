# B11 fixed-source inventory and MiniCPM snapshot recovery

Integration base: `c62fe995`.
Author source audit only: no unified LLM/VLA migration acceptance or new board result.
Source: `rdk_s` commit `380e1a2bf42041af54be6f34935e50197cfadff9`.

## Inventory and restoration

| Source | Fixed source | Initial archived snapshot | Action |
| --- | --- | --- | --- |
| gemma4-e2b | 67 files | All 67 byte-equal | Preserve unchanged |
| minicpm5-2b | 62 files | 40 equal, 13 absent, 9 README files older | Restore absent files and update only the nine stale guides to exact pinned bytes |
| VLA ACT | gitlink `326ea043be204de25223d95c7d918efe8672dc66` | Same commit under platforms/s/samples/vla/act, uninitialized | Record URL/path/pin; relocation and upstream-content integration pending |
| VLA Pi0 | gitlink `a32de276bc1681a2b1531012de111eaa1c16acb6` | Same commit under platforms/s/samples/vla/pi0, uninitialized | Record URL/path/pin; relocation and upstream-content integration pending |

The MiniCPM README snapshot had no subsequent integration edits (last touched by
initial platform consolidation). Recovery preserves the fixed source rather than
rewriting it. Its 13 absent files comprised seven legacy evaluator files, five
historical result JSON files and the long-prompt fixture. Nine stale guides are
now identical to the fixed commit. All 129 LLM files are verified against Git blob
IDs and recorded with SHA-256 and modes in the
[source inventory](evidence/2026-09-28-b11-source-audit/source-inventory.json).
Earlier snapshot bytes remain in Git history, with pre-restoration digests recorded.

## Material source facts to preserve during migration

- MiniCPM S100/S100P uses the separate OELLM 1.0.0 legacy runtime/artifacts. The
  previously omitted full WikiText2 record reports PPL 17.91995, 27.83% above the
  float reference, **failing the source ≤3% target**. Passing short generation
  does not erase that failure. These are inherited source results, not this
  audit's measurements.
- MiniCPM S600 uses the separate OELLM 2.0 beta workflow and its own publication.
  Source-reported PPL 14.2428 and other S600 measurements cannot be extended to
  S100/S100P. The source's SDK release forecast is historical text; the unified
  guide must distinguish it from verified present availability.
- Gemma includes native chat, server, demo, text benchmark and golden verification,
  plus tokenizer integration, image examples and detailed quantization tutorials.
  Migration must preserve those capabilities and references instead of reducing
  the sample to a single decoder entry.
- ACT and Pi0 use the same upstream repository URL at **different commits**.
  They must remain exact Git submodules; a placeholder directory or tracking
  upstream HEAD is not an equivalent migration. Current .gitmodules paths are
  still archival and need explicit unified-path treatment.

## Next implementation scope

1. Migrate the native LLM runtimes with source-target distinctions, shared
   mechanisms where appropriate, separated numerical/SDK/application concerns,
   explicit preparation and host-inspectable interfaces.
2. Preserve all README levels, reference tables/images, tokenizer/dependency
   provenance, evaluation failures and existing conversion tutorials. Reorganize
   trusted quantization guidance without running exports/calibration/compilation
   or model evaluations, as explicitly instructed by the user.
3. Relocate VLA gitlinks with matching .gitmodules, read the exact pinned upstream
   content and provide practical bilingual integration/navigation. Do not run
   robot-control commands or connect to hardware.
4. Add appropriate host tests and enter samples into the contract scope as their
   real unified implementations land. Independent whole-branch review remains
   a separate final step.

No model download, package/toolchain installation, conversion, numerical model
evaluation, board connection or submodule initialization occurred in this audit.
B11 Refactor/Docs remain pending; this record closes only the source-inventory gap.
