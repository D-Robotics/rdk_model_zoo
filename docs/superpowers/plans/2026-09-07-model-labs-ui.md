# Model Labs UI Implementation Plan

**Goal:** Implement the approved catalog and detail presentation using existing data.

**Architecture:** Keep static TypeScript/Vite. Separate hardware navigation, task filters, card presentation and detail evidence components; retain existing metric and artifact adapters.

**Spec:** ../specs/2026-09-07-model-labs-ui-framework-design.md

## Constraints

- Do not change manifests, benchmark values, download URLs, source records or tags.
- Keep five hardware choices, independent detail links, bilingual controls and themes.
- No backend or extra root entry. Continue under docs/catalog.

## Implementation

- [x] Directory: expose a hardware bar, search toolbar and sidebar from filters.ts; group tasks through an explicit taxonomy; preserve multiselect URL state and support mobile draft/apply/cancel.
- [x] Cards: use a presentation adapter for scoped variants and specification labels; add a decorative task illustration with no invented results or descriptions.
- [x] Details: extract downloads/evidence responsibilities, compact header and full-width expandable evidence rows; retain numerical behavior.
- [x] Verify: add interaction regressions for filters and detail expansion; run npm run check, inspect desktop/mobile in browser, verify source data unchanged.
- [ ] Publish the verified UI to rdk_x5 and check Pages deployment.
