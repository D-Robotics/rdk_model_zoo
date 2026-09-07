# Hardware Model Detail Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development to implement and review these tasks. Existing user authorization covers upstream publishing and subagents.

**Goal:** Ship the approved family catalog with canonical hardware filters, standalone detail pages, row-associated quantized downloads and accuracy retention.

**Architecture:** Keep the static TypeScript/Vite site. Normalize supported hardware and runnable variants in a shared catalog module, preserving source manifests. UI consumes these variants; URL state selects model/hardware/task independently of directory filters.

**Tech Stack:** Existing TypeScript, Vite, Vitest/jsdom, GitHub Pages; no new runtime dependencies.

**Spec:** `docs/superpowers/specs/2026-09-07-hardware-model-detail-design.md`, approved in conversation after interactive preview.

## Global constraints

- Hardware order: X3 / X5 / S100 / S100P / S600; unsupported hardware hidden in details.
- No board testing, ONNX download entry or YOLOE. Do not move existing release tags.
- Preserve bilingual UI, themes, source provenance and test conditions.
- Missing measurements: 尚未实测; missing metadata: 未记录.
- Downloadable but unmeasured variants stay visible. No fabricated file/benchmark matches.

## Shared interfaces

Add to `site/src/catalog/types.ts`:

```ts
export type HardwareId = "x3" | "x5" | "s100" | "s100p" | "s600";
export interface ModelVariant {
  id: string;
  name: string;
  hardware: HardwareId;
  task: string; // existing manifest task identifiers
  input?: BenchmarkRecord["input"];
  assets: ModelRecord["assets"];
  benchmarks: BenchmarkRecord[];
  sample_path: string;
  release_tag: string;
}
// ModelRecord adds optional variants?: ModelVariant[].
```

`site/src/catalog/variants.ts` exports `HARDWARE_IDS: HardwareId[]`, `normalizeHardware(value: string): HardwareId | undefined`, `getModelVariants(model: ModelRecord): ModelVariant[]`, `getHardwareIds(model: ModelRecord): HardwareId[]`.

## Task 1: Hardware/variant/asset data integrity

Files: `site/src/catalog/types.ts`, new `site/src/catalog/variants.ts`, `site/scripts/multiplatform-catalog.ts`, relevant tests in `site/tests/variants.test.ts` and `multiplatform-catalog.test.ts`.

- [x] Write regression tests for X3 aliases; distinct S100/S100P/S600 evidence; unmeasured runnable files; exact YOLO task/size/input matching; cross-family asset exclusion.
- [x] Run targeted variant/catalog regressions; the final suite includes real X5 associations, unmeasured S hardware, source paths, ambiguous inputs, and NHWC shapes.
- [x] Populate ModelVariant from explicit benchmark asset references and verified source filename conventions. Resolve multiple configurations separately. Unknown mappings remain unmapped, with no guessed download. Filter non-runnable files.
- [x] Generator aggregates every family's tasks and records, supplies all hardware variants and coherent summary totals, retains source sample/ref per variant. Runtime fallback supports existing catalog fixtures and legacy records.
- [x] Run the targeted tests and inspect real YOLOv8/MobileNet rows.

Example hardware expectation:
```ts
expect(normalizeHardware("RDK X3 & RDK X3 Module (Bernoulli2)")).toBe("x3");
expect(HARDWARE_IDS).toEqual(["x3", "x5", "s100", "s100p", "s600"]);
```

## Task 2: Standalone detail contents and comparable metrics

Files: `site/src/ui/model-details.ts`, new `site/src/catalog/metric-display.ts`, new `site/src/ui/detail-labels.ts`, `site/tests/model-details.test.ts`, `site/tests/metric-display.test.ts`.

Consumes shared ModelVariant APIs. Keep `readModelId`, `writeModelId`, `renderModelDetails`. Extend DetailContext with `hardware?: HardwareId`, `task?: string`, `onSelectionChange?: (hardware: HardwareId, task: string) => void`; retain legacy `platform?` as alias. Details expose `data-hardware` and `data-task` selections on root.

- [x] Write tests for supported tabs, switch-local files, same-row thread pairs, explicit and derived retention, wrong dataset, zero denominator and X3 postprocessing separation.
- [x] Run the tests and implement a section (not dialog) with back button `[data-action="close-details"]`, H1, hardware tabs, supported task selector and semantic grouped tables.
- [x] Group performance by compatible scope/statistic/concurrency; unknown concurrency remains labeled. Keep every original metric accessible in row details. Pair accuracy by metric, dataset, unit, scope/statistic and artifact; raw retention wins.
- [x] For example `0.292 / 0.306 * 100` displays `95.42%`; error metrics do not automatically get retention. Downloads use variant.assets with runnable formats only. Show input sizes separately from variant labels.
- [x] Run targeted tests. No CSS/app/query edits in this task, allowing independent integration.

## Task 3: Directory, navigation and styling

Files: `site/src/app.ts`, `site/src/catalog/query.ts`, `site/src/ui/filters.ts`, `site/src/ui/model-card.ts`, `site/src/styles.css`, `site/src/ui/summary.ts`, tests.

- [x] Extend app tests: detail view hides directory; hardware/task URL survives refresh and popstate; back restores directory filter/focus; five fixed platform choices; card labels link to selected hardware.
- [x] Use `getHardwareIds` for support and `getModelVariants` for filtered records. Cards show family/tasks/specifications/hardware without misleading representative benchmark values.
- [x] Persist directory query in independent URL fields; model/hardware/task select details. Use pushState on navigation, replaceState on canonicalization and popstate to restore. Preserve unrelated query params.
- [x] Apply approved orange-accent layout, tab focus, grouped numeric cells, sticky specification column and local horizontal table scrolling. Make summary totals reflect all platforms.
- [x] Run app/query/filter/summary tests and typecheck.

Navigation mutation:
```ts
const next = writeModelId(new URL(window.location.href), model.id);
next.searchParams.set("hardware", hardware);
next.searchParams.set("task", task);
window.history.pushState({}, "", next);
```

## Task 4: Review, verify and publish

- [x] Review all changed modules against the spec; fix source/artifact and metric comparability findings first.
- [x] Run `npm run check` under supported Node 22 and complete repository CI checks applicable to changed files.
- [x] Inspect browser directory/detail at desktop/mobile widths, language/theme and hardware/task/history behavior.
- [ ] Commit scoped changes; push to already-authorized official rdk_x5 after checking remote changes; dispatch existing Pages workflow.
- [ ] Wait for successful deployment and verify public catalog plus direct model/hardware/task URL. Report live link and limitations that remain in source data.

## Execution record

- Initial state: design commit `44609a3`; production baseline `f1c5769`; existing isolated feature worktree, clean before implementation.
- Ruling: execute without another permission round because the user approved the complete design and previously authorized official upstream publication.
- Ruling: current X3 YOLOv8 postprocessing is not BPU latency; retain original scope and do not infer concurrency.


- Final local validation: 103 tests passed; TypeScript and Vite production build passed. Browser checked at desktop and 390px widths, with hardware/task navigation, locale/theme changes and source links. Runnable download counts exclude ONNX and auxiliary metadata files.

