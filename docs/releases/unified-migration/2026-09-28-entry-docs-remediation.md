# H1 (bounded): entry-document follow-up remediation — 2026-09-28

Status: **implemented by Claude Code + GLM; pending independent Codex review.
H1/H7 are not closed by this document.**

Scope: exactly six product files — root `README.md`/`README_cn.md`,
`samples/README.md`/`samples/README_cn.md`,
`platforms/README.md`/`platforms/README_cn.md` — plus this author report and
`evidence/2026-09-28-entry-docs-remediation/`. No code, manifest, mapping,
config, JSON or other documentation file was touched; the Gemma/bytetrack/
yolov5/catalog-publisher changes visible in the shared worktree belong to
other workers. Documentation-only: no install, download, build, board/SSH use
or network access; every inherited link is copied verbatim from archived
in-tree sources with no claim of remote availability or live link status.
Counts, commands, target/artifact facts, the develop/customer-delivery warning
and the board-evidence limitations are preserved.

## Findings addressed

| Finding (review input) | Resolution |
| --- | --- |
| ENTRY-DOC-R1: both platform guides said the publisher locates manifests "through the registry" | `platforms/README.md` / `README_cn.md` now state that `tools/catalog-publisher` resolves the manifest paths from its own `sources.json` and that `platforms/registry.json` is cross-checked against that configuration by tests, not read by the build — matching the publisher guide's "Data sources" section. Neither JSON was edited and source provenance is unchanged |
| ENTRY-DOC-R2 (MiniCPM): indexes implied no native-core acceptance | `samples/README.md` / `README_cn.md` now distinguish the [final independent core disposition](2026-09-28-minicpm-core-independent-review.md) — core refactor accepted within host scope (20 passing host tests, four compiled README examples) — from what stays open: batch B11, vendor ABI/model accuracy, board acceptance; board tests not-run; no live board, vendor ABI or model accuracy acceptance claimed. The exact review is linked in both languages |
| ENTRY-DOC-R2 (B8–B11): generic "B8–B11 migration and final review remain active" | Replaced with: B9 and B11 migration and their independent review remain active; B8/B10 acceptance is non-board only and repository-wide review (H1/H8/H9) is not closed by any batch. A new "Read validation status correctly" bullet names the [B8 aggregate review](2026-09-28-b8-batch-independent-review.md) (eight scopes, H4) and the [B10 integrated review](2026-09-28-b10-batch-independent-review.md) (ASR/KWS/Paraformer/HIMLoco, H6) as non-board batch acceptance with board scope not-run. Gemma Text is left as in-flight/open — no acceptance is claimed from the live worker |
| ENTRY-DOC-N1: canonical guides delegated source-branch resources to archived guides | Restored concise direct inherited links with source attribution, in the root "Data, source-branch resources and validation" bullet list and the platform guides' "Retained development references": the published [online model catalog](https://d-robotics.github.io/rdk_model_zoo/), [GitHub Issues](https://github.com/D-Robotics/rdk_model_zoo/issues), the [D-Robotics developer community](https://developer.d-robotics.cc/) and its [user manual](https://developer.d-robotics.cc/information). Each states the links are inherited from the archived X5 (`ac11571`) / S (`380e1a2`) root guides, describe the delivery branches' published material, do not certify this integration branch, and carry no live-link-status claim. Legacy entries stay archives, not adaptation targets: [`rdk_x5_legacy`](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_x5_legacy) and [`rdk_model_zoo_s`](https://github.com/D-Robotics/rdk_model_zoo_s), with `rdk_x5`/`rdk_s` remaining the registered delivery lines. The root Community section's "repository Issues" is now the direct Issues link; task navigation, badges/star-history marketing and run recipes were not copied, and platform toolchain manuals are left in the archived guides rather than re-asserted as current |

## Verification (host, static)

All four checks in `evidence/2026-09-28-entry-docs-remediation/verification.json`
pass; the scoped diff is `entry-docs.diff` (+12/−7 over six files):

- **Fenced command blocks unchanged**: all four blocks in each root README
  (and therefore every shell command) hash-identical before/after; the other
  four files contain no fenced blocks.
- **Bilingual inventory 51**: sample links parsed from both indexes equal
  exactly the reviewer's 51-root inventory (45 vision, three speech, one
  robotics, two LLM); no missing or extra entries.
- **Relative links**: 496 relative links across the six files all resolve,
  including the `platforms/x5/README.md#community--contribution` fragment.
- **Exact resource links**: every inherited URL is byte-identical to its
  occurrence in the archived `platforms/x5/README*.md` (`ac11571`) and
  `platforms/s/README*.md` (`380e1a2`) guides.

## Remaining limits (not claimed)

- No remote check of any restored URL: availability, current online catalog
  content and toolchain compatibility are not asserted, per the review.
- Non-board scope only: B8/H4 and B10/H6 batch acceptance, the MiniCPM core
  host disposition and all board fields remain exactly as their reviews state;
  H1/H7/H8/H9, B9, B11 and whole-branch acceptance stay open, and this package
  closes none of them. Parent-README hierarchy beyond the six allowed files
  was not edited.
- The two archived guides spell the S legacy org differently (`d-Robotics` vs
  `D-Robotics`); the S-guide spelling is used and the variant is recorded in
  `evidence/evidence.md`, with no claim about which resolves.
- Independent Codex review of this package is pending.

## Files

- Modified: `README.md`, `README_cn.md`, `samples/README.md`,
  `samples/README_cn.md`, `platforms/README.md`, `platforms/README_cn.md`
- New: `docs/releases/unified-migration/2026-09-28-entry-docs-remediation.md`
  (this record)
- Evidence: `evidence/2026-09-28-entry-docs-remediation/`
  (`evidence.md`, `verification.json`, `entry-docs.diff`)
