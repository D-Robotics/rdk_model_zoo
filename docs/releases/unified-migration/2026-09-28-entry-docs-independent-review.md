# Root, sample index and platform guide review — 2026-09-28

Reviewer: Codex. H1 entry-document follow-up required.

Current EN/CN sample indexes exactly cover 51 native sample roots: 45 vision,
three speech, one robotics and two LLM, with VLA separately linked. Static
inventory verifies 490 local file links across six root/index/platform guides;
none is missing. This does not prove remote availability or all fragment IDs.
See `evidence/2026-09-28-entry-docs-independent-review/inventory.json`.

## ENTRY-DOC-R1 — publisher source configuration described incorrectly

Both platform guides say the publisher locates manifests from registry.json.
Actual source resolution reads tools/catalog-publisher/sources.json; registry
agreement is checked by tests. Align this with the publisher guide without
editing either JSON or changing source provenance.

## ENTRY-DOC-R2 — current status and navigation lag behind accepted work

The sample indexes still imply no MiniCPM native-core acceptance. Its final
independent core disposition accepted 20 tests and four compiled README examples,
while full B11, vendor ABI/model accuracy/board acceptance remain open. Update
that distinction explicitly, linking the exact review. B8/H4 and B10/H6 have
non-board batch acceptance; the generic B8–B11 migration-is-active statement
should distinguish them from B9/B11 pending work and repository-wide review.
Gemma Text is still in flight; do not claim acceptance for it from a live worker.
Keep the develop/customer-delivery warning and exact board evidence limitations.

## ENTRY-DOC-N1 — restore useful source-level resource navigation

X5 ac11571 and S 380e1a2 root guides directly link the published online catalog,
Issues and D-Robotics developer resources, and explain X5 legacy branch/S legacy
repository. Canonical root/platform guides currently mostly delegate those
resources to archived guides. Restore concise direct inherited links in the
appropriate root/platform sections, preserving source attribution and clarifying
that published catalog content does not certify this integration branch. Keep
legacy entries identified as archives, not active adaptation targets. No new
claims about current online content, toolchain compatibility or live link status.
Do not copy marketing/badges for length, replace task navigation or rerun recipes.

Scope for Claude Code + GLM: README.md/README_cn.md, samples/README pair,
platforms/README pair, plus its author report/evidence only. Preserve commands,
counts, target/artifact facts and customer-release boundary. Codex will review.

## Independent acceptance of the six-file correction

Codex reread the complete root/platform/sample-index changes against sources.json,
registry use in tests, fixed source README resources and the final MiniCPM/B8/B10
review dispositions. ENTRY-DOC-R1/R2/N1 are closed. MiniCPM core host acceptance
is distinct from B11 and real-model acceptance; Gemma Text remains in progress.
The root still directs customers to published delivery lines rather than declaring
the integration branch release-ready. Resource links retain source attribution.

`final-recheck.json` records hashes of all six guides, unchanged fenced command
blocks and 496 resolving local file/directory links. Newly added URLs were checked
against the fixed source README text, not fetched or asserted live. Source sample
inventory remains the earlier exact 51-root EN/CN mapping; no table rows changed.
This closes only the entry package; H1 final sample/body reconciliation remains.
