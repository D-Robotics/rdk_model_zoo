# RDK Model Zoo UI — editorial style

## Visual direction

The September 2026 redesign adapts the supplied Boomerang reference to a model catalog: white canvas, near-black text, serif display typography, generous introductory spacing, hairline separators, and restrained 200 ms interaction transitions. The model directory remains the primary product. No imagery, background video, canvas animation, marketing navigation, or additional UI library is required.

The display stack uses Georgia and local CJK serif fallbacks. UI text uses the available Inter/system sans-serif stack. No third-party font request is required for the page to work. Keep the existing TypeScript/Vite application and catalog schema.

## Page structure

- Header: RDK Model Zoo home link, language/theme preferences, one GitHub link.
- Directory: bilingual introduction, Explore models anchor, directory heading, visible X3/X5/S100/S100P/S600 platform controls, task filters and search, text-only family cards. Existing inventory/release metadata follows the collection.
- Family cards: family title, task labels, size/variant summary, supported hardware shortcuts, one full-width details action. Long names wrap naturally.
- Detail page: return/share controls, model title, hardware tabs, task selection, specification/performance/accuracy table and quantized file downloads. The directory introduction is hidden when a detail is open.
- Mobile: header controls wrap, hardware controls use two rows, the collection becomes one column, task filters use the existing modal drawer. Benchmark tables scroll inside their own container.
- Dark mode: the same hierarchy with warm near-white text and charcoal surfaces. Keyboard focus, reduced-motion preferences and existing accessible tab behavior remain supported.

## Integration boundaries

This change edits the HTML shell, app composition, presentation-only UI components, CSS and a navigation test. It does not edit `src/catalog/`, schemas, catalog generation, manifests, benchmark records, release pins or versions. The content-refresh agent can merge its data changes independently.

`src/ui/editorial-theme.css` is imported last by `src/main.ts`. It is the final visual layer over the existing structural component styles. New visual adjustments should be made here rather than adding competing overrides to the older component styles. `src/ui/editorial-shell.ts` owns the directory introduction only. Keep numeric results and hardware support derived from the existing catalog functions.

The preference controls mount into `#header-preferences` in the HTML shell; the app still supports mounting without that host (tests and embedded use). App teardown removes controls mounted outside the root. The mobile filter drawer marks the introduction and directory heading inert along with the existing background elements.

## Validation and handoff

- `npm run check`: 114 tests, TypeScript checking and production build passed.
- Headless Edge: desktop 1440 px and mobile 390 px; search, family navigation, hardware switching, filter apply, background isolation, preserved search on return, language remount, dark mode and viewport overflow checks passed.
- Visually inspected the directory, model detail, mobile layout and dark theme. No model/media assets or package dependencies were added.

Development preview: run `npm ci` then `npm run dev -- --host 127.0.0.1 --port 5188` in `docs/catalog`. The URL is `http://127.0.0.1:5188/rdk_model_zoo/`.

Merge the UI commit with the content refresh before deploying the combined site. Preserve the refreshed manifest and benchmark files during that merge; do not replace them with the baseline used for UI preview. Existing published release tags remain immutable.
