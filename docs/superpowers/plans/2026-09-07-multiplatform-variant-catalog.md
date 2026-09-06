# Multi-platform Variant Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a card per exact model variant while aggregating X5, S, and X3 support and measurements.

**Architecture:** The build script reads three immutable release-manifest pairs from Git worktrees created from release tags. A checked-in identity registry maps only reviewed equivalent variants. The generated catalog exposes an aggregated card with one platform payload per supported platform; UI components render platform sections and separate measurement types.

**Tech Stack:** Node.js 22, TypeScript, Vite, Vitest, YAML, Ajv, GitHub Actions, GitHub Pages.

**Spec:** `docs/superpowers/specs/2026-09-07-multiplatform-variant-catalog-design.md`

## Global Constraints

- Data comes only from `x5-v1.0.0`, `s-v1.0.0`, and `x3-v1.0.0` manifests and their existing source evidence.
- Do not run board tests or invent performance, accuracy, equivalence, checksums, or test conditions.
- Exclude YOLOE from all generated data and UI.
- A missing performance or accuracy measurement means “not yet measured”; missing metadata means “not recorded”.
- Do not merge variant names that differ in model version, task, or recorded input.

### Task 1: Define multi-platform catalog data and identity validation

**Files:**
- Create: `release/catalog-identities.yaml`
- Modify: `site/src/catalog/types.ts`
- Modify: `site/scripts/catalog-builder.ts`
- Modify: `site/scripts/build-catalog.ts`
- Modify: `site/tests/catalog-builder.test.ts`

**Interfaces:**
- `buildCatalog` accepts a named source list of platform, tag, and manifest paths.
- It returns `Catalog.cards`, where each card has `id`, `name`, `task`, `input`, and `platforms` keyed by `x5`, `s`, or `x3`.
- Each platform payload preserves source assets and benchmark records.

- [ ] **Step 1: Write failing aggregation tests**

```ts
expect(cardById(catalog, "efficientformer-l1-224").platforms.x5).toBeDefined();
expect(cardById(catalog, "yolov5su-detect-640").platforms.s).toBeDefined();
expect(cardById(catalog, "yolov5s-v7-detect-640").id)
  .not.toBe("yolov5su-detect-640");
```

- [ ] **Step 2: Run the focused test and confirm aggregation is absent**

Run: `npm test -- tests/catalog-builder.test.ts`

Expected: FAIL because the X5-only catalog has no aggregated cards.

- [ ] **Step 3: Add the reviewed identity registry and source-list types**

Create registry entries with a canonical card id and explicit platform benchmark selectors. Include only known exact equivalences and leave every other variant platform-specific. Add `PlatformCatalogSource`, `VariantCard`, and `PlatformVariant` types.

- [ ] **Step 4: Implement manifest loading, registry validation, and card aggregation**

Load all three source pairs, validate each source tree and tag, index benchmark records by `(sample_id, variant_id)`, validate registry selectors, and emit one card per registry identity or unregistered platform record. Reject a registry mapping that has different task or recorded input values.

- [ ] **Step 5: Run builder and full catalog tests**

Run: `npm test -- tests/catalog-builder.test.ts tests/benchmark-coverage.test.ts`

Expected: source evidence remains immutable; exact YOLO variants remain separate; no YOLOE occurs.

- [ ] **Step 6: Commit**

```bash
git add release/catalog-identities.yaml site/src/catalog/types.ts site/scripts site/tests/catalog-builder.test.ts
git commit -m "feat(catalog): aggregate exact variants across platforms"
```

### Task 2: Render platform support and independent measurements

**Files:**
- Modify: `site/src/catalog/query.ts`
- Modify: `site/src/ui/model-card.ts`
- Modify: `site/src/ui/model-details.ts`
- Modify: `site/src/ui/summary.ts`
- Modify: `site/src/i18n/translations.ts`
- Modify: `site/src/styles.css`
- Modify: `site/tests/app.test.ts`
- Modify: `site/tests/model-details.test.ts`
- Modify: `site/tests/query.test.ts`

**Interfaces:**
- Cards render X5, S, and X3 support separately.
- `renderMetricSection(platformVariant, kind)` renders either performance or accuracy and independently uses the “not yet measured” state.

- [ ] **Step 1: Add failing UI tests**

```ts
expect(card.textContent).toContain("S");
expect(card.textContent).toContain("X3");
expect(details.textContent).toContain("Performance not yet measured");
expect(details.textContent).toContain("Accuracy");
```

- [ ] **Step 2: Run UI tests and confirm the current sample-family UI fails**

Run: `npm test -- tests/app.test.ts tests/model-details.test.ts tests/query.test.ts`

Expected: FAIL because current cards carry only one platform and flatten benchmark records.

- [ ] **Step 3: Update query, card, detail, summary, and translation units**

Search and filters use card identity and every platform payload. Render model precision from metadata only. Render Performance and Accuracy tables independently under each platform heading. Use “not yet measured” for absent measurement arrays and “not recorded” only for missing conditions or checksums.

- [ ] **Step 4: Run focused UI tests**

Run: `npm test -- tests/app.test.ts tests/model-details.test.ts tests/query.test.ts tests/language.test.ts`

Expected: PASS in English and Chinese; YOLOv5s and YOLOv5su are distinct cards.

- [ ] **Step 5: Commit**

```bash
git add site/src site/tests
git commit -m "feat(catalog): show platform-specific variant evidence"
```

### Task 3: Deploy and verify the multi-platform catalog

**Files:**
- Modify: `.github/workflows/model-catalog-pages.yml`
- Modify: `docs/RELEASE.md`
- Modify: `docs/RELEASE_cn.md`
- Modify: `README.md`
- Modify: `README_cn.md`
- Test: `site/tests/pages-workflow.test.ts`

- [ ] **Step 1: Add workflow/documentation expectations**

Require the Pages job to use the three immutable release tags when generating catalog data. Document that an X5 release deploys the aggregate catalog rather than an X5-only view.

- [ ] **Step 2: Run the workflow test and full verification**

Run: `npm run check` followed by `git diff --check`.

Expected: all tests, type checking, catalog generation, and production build pass.

- [ ] **Step 3: Commit and push**

```bash
git add .github/workflows README.md README_cn.md docs/RELEASE.md docs/RELEASE_cn.md site/tests/pages-workflow.test.ts
git commit -m "docs(catalog): describe multi-platform variant release data"
git push origin HEAD:rdk_x5
```

- [ ] **Step 4: Deploy and inspect**

Run `gh workflow run model-catalog-pages.yml --repo D-Robotics/rdk_model_zoo --ref rdk_x5 -f catalog_ref=rdk_x5`, then wait with `gh run watch`.

Verify the public catalog contains separate cards for YOLOv5s and YOLOv5su, a multi-platform support row, independent performance/accuracy sections, and no YOLOE text.
