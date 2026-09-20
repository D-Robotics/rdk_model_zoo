# Phase 0.5 Q1 review — README content contract and templates (2026-09-20)

Scope: plan Q1 — README content contract, bilingual templates for six README
levels, root guidelines revision, old-spec conflict reconciliation, reference
coverage check (ResNet single-model / OCR multi-stage).

## Delivered

| Artifact | Content |
| --- | --- |
| `docs/sample-standards/readme-contract.md` | §1 applicability per level (incl. not-applicable rules), §2 bilingual pairing, §3 fixed section-ID mechanism (explicit HTML anchors, language-independent), §4 per-level contract tables (ID → must-answer questions → evidence → applicability) for sample/model/runtime-python/runtime-cpp/conversion/evaluator, §5 unified content discipline (command five-elements, defined-variable examples, no hand-off, no masquerade, sha256 discipline, API summary, verified/not-run), §6 old-spec conflicts C1–C7 with dispositions, §7 ResNet/OCR coverage analysis, §8 maintenance rules |
| `docs/sample-standards/templates/{sample,model,runtime-python,runtime-cpp,conversion,evaluator}.{en,zh}.md` | 12 templates; every section carries its fixed anchor plus a must-answer guidance block; placeholders `⟪…⟫`; not title-only skeletons |
| `docs/Model_Zoo_Repository_Guidelines.md` | develop linking version: spec layering table (AGENTS/contracts/migration/ADR), develop directory layout, sample layout & language coverage rules, release pointers. The file did not exist on develop; rdk_x5's full coding/comment/task chapters merge in at Phase 1 A6 with directory chapters rewritten per conflict C7 |

## Verification performed (this session, host-side only)

1. **Anchor parity**: automated regex check — for each of the 6 levels the
   ordered anchor-ID list is identical between `.en.md` and `.zh.md`
   (9/5/7/7/8/7 anchors). Note: header comments contain a literal
   `<a id="…"></a>` example; Q3 must parse anchors outside HTML comments.
2. **Placeholder balance**: whole-file ⟪/⟫ count equality per template — all
   12 balanced. Self-review caught 6 typos (swapped/stray brackets) before
   verification passed; the check snippet is recorded below and its
   generalization feeds Q3 rule design.
3. **Coverage check (design-level)**: contract §7 — ResNet expresses
   single-model stage I/O and variant rows; OCR expresses per-stage artifacts
   and pipeline composition through `stage-io` subsections. No structural gap
   found. This validates the CONTRACT's expressiveness only; the two samples
   are re-authored and accepted in Q4, not by this review.
4. **Old-spec conflict analysis**: rdk_x5 guidelines (1089 lines) read; C1–C7
   recorded in contract §6 (auto-download QuickStart, under-specified model
   README, unregulated conversion/evaluator, no support matrix / unconditional
   dual-language, default-value checking upgraded to automated, absent
   source_reference on develop, rdk_x5 directory tree).
5. **Referenced paths exist**: AGENTS.md, docs/{adr,release,releases,
   superpowers,sample-standards}, platforms/, samples/_shared/,
   tools/catalog-publisher — all present on develop.

Verification snippet (balance/anchors):

```bash
# cwd: docs/sample-standards/templates
python3 - <<'EOF'  # counts ⟪/⟫ per file; regex-compares <a id="…"> lists en vs zh
EOF  # full text in session transcript; to be superseded by Q3 checker rules
```

## Explicit limits / not-run

- No sample README has been authored against the templates yet — that is Q4
  (resnet, paddle_ocr) and per-batch work. The templates are structurally
  verified, not yet battle-tested on real content.
- The anchor/balance checks are ad-hoc snippets recorded here, not yet rules
  in an automated checker — Q3 lands `tools/sample_contract/check.py`; its
  scope includes template conformance of authored READMEs.
- Root guidelines on develop are a linking version by design; A6 performs the
  rdk_x5 content merge and directory-chapter rewrite (C7).
- Bilingual semantic equivalence of the templates is by mirrored authoring +
  single self-review; independent dual-perspective review happens in Q4/batch
  delivery reviews, not here.
- Board/runtime behavior untouched — documentation-only change set.

## Self-review findings and fixes during Q1

- 6 placeholder-bracket typos across 4 templates (incl. one introduced by a
  fix and caught on re-run) — fixed; motivates making bracket/anchor sanity a
  permanent automated rule rather than a one-off.
- Contract §4.3 originally lacked the explicit `hbm_runtime` host-unavailability
  note; added to the template's `environment` section (matches develop
  architecture reality).
- Pilot-row board nuance corrections belong to Phase 0 (already recorded in
  the phase 0 review, not double-counted here).

## Verdict

Q1 deliverables complete for the baseline's structural scope: contract + 12
templates + linking root guidelines + conflict record + coverage analysis, with
verification evidence above. Q1 does NOT claim any existing sample README is
compliant — that is Q4's acceptance. Proceed to Q2 (inference contract).
