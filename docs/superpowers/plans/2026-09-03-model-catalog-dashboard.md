# RDK Model Zoo Online Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and publicly deploy a bilingual, Manifest-driven RDK Model Zoo catalog that presents model cards, downloads, performance, accuracy, test conditions, and immutable source evidence.

**Architecture:** A Vite + TypeScript static application reads a generated `catalog.json`. A build-time TypeScript program validates `release/models.yaml` and `release/benchmarks.yaml` against JSON Schema, enforces cross-file references, and emits the browser payload. GitHub Actions validates every relevant change and deploys release-linked static artifacts to GitHub Pages without a backend.

**Tech Stack:** Node.js 22, npm, Vite 7, TypeScript 5, Vitest 3 with jsdom, YAML, Ajv JSON Schema validation, vanilla DOM APIs, GitHub Actions, GitHub Pages

**Spec:** `docs/superpowers/specs/2026-09-03-model-catalog-dashboard-design.md`

## Global Constraints

- Production URL: `https://d-robotics.github.io/rdk_model_zoo/` with Vite base path `/rdk_model_zoo/`.
- Canonical release inventory remains `release/models.yaml`; canonical benchmark data is `release/benchmarks.yaml`.
- First release only transcribes values already published in repository documentation. Do not run new board or dataset tests.
- Never derive FPS from latency, derive latency from FPS, or invent an unspecified test condition.
- Every benchmark record must identify an immutable source ref, repository path, and section.
- Missing benchmark data is valid and must render an explicit empty state.
- Cross-model performance sorting is allowed only for records with matching hardware, metric, unit, timing scope, concurrency, and input conditions.
- Accuracy sorting additionally requires matching dataset, metric, unit, and model stage.
- The site is bilingual Chinese/English, follows browser language on first visit, and persists an explicit user choice.
- YOLOE must not occur in catalog data, navigation, cards, details, statistics, or generated output.
- The site has no backend, account system, comments, online inference, model upload, or global fastest/most-accurate leaderboard.
- Production deployment must display the Release Tag used as its data source.

## Planned File Structure

```text
release/
├── benchmarks.yaml                       # Published performance and accuracy evidence
└── schemas/
    ├── models.schema.json                # Structural contract for models.yaml
    └── benchmarks.schema.json            # Structural contract for benchmarks.yaml
site/
├── index.html                            # Accessible static document shell
├── package.json                          # Build, test, and validation commands
├── package-lock.json                     # Reproducible dependency lock
├── tsconfig.json                         # Strict browser and script TypeScript settings
├── vite.config.ts                        # GitHub project-site base path and output
├── vitest.config.ts                      # jsdom and test setup
├── scripts/
│   ├── build-catalog.ts                  # CLI that emits public/data/catalog.json
│   └── catalog-builder.ts                # YAML, Schema, reference, and source validation
├── src/
│   ├── app.ts                            # Application state and render orchestration
│   ├── main.ts                           # Browser entry point
│   ├── styles.css                        # Theme, responsive layout, and accessibility styles
│   ├── catalog/
│   │   ├── query.ts                      # Search, filters, comparable sorting
│   │   └── types.ts                      # Shared release/catalog domain types
│   ├── i18n/
│   │   ├── language.ts                   # Browser detection and persisted language state
│   │   └── translations.ts               # Chinese and English UI strings
│   └── ui/
│       ├── filters.ts                    # Search and filter controls
│       ├── model-card.ts                 # Model summary card
│       ├── model-details.ts              # Detail, evidence, asset, and benchmark tables
│       └── summary.ts                    # Release and inventory statistics
└── tests/
    ├── app.test.ts                       # Catalog interaction and empty-state integration tests
    ├── catalog-builder.test.ts           # Schema and cross-reference tests
    ├── fixtures/                         # Minimal valid and invalid YAML inputs
    ├── language.test.ts                  # Browser language and preference tests
    ├── model-details.test.ts             # Detail URL, tables, evidence links
    ├── query.test.ts                     # Search, filtering, and comparison rules
    └── setup.ts                          # jsdom cleanup and localStorage reset
.github/workflows/
├── model-catalog-ci.yml                  # Data validation, unit tests, production build
└── model-catalog-pages.yml               # Release/manual Pages deployment
README.md                                 # Public catalog link
README_cn.md                              # Chinese public catalog link
docs/RELEASE.md                           # Catalog release/deployment step
docs/RELEASE_cn.md                        # Chinese catalog release/deployment step
release/README.md                         # Benchmark file contract and maintenance notes
```

### Task 1: Establish the release data contracts and catalog builder

**Files:**
- Create: `site/package.json`
- Create: `site/package-lock.json`
- Create: `site/tsconfig.json`
- Create: `site/vite.config.ts`
- Create: `site/vitest.config.ts`
- Create: `site/src/catalog/types.ts`
- Create: `site/scripts/catalog-builder.ts`
- Create: `site/scripts/build-catalog.ts`
- Create: `site/tests/setup.ts`
- Create: `site/tests/catalog-builder.test.ts`
- Create: `site/tests/fixtures/models.valid.yaml`
- Create: `site/tests/fixtures/benchmarks.valid.yaml`
- Create: `release/schemas/models.schema.json`
- Create: `release/schemas/benchmarks.schema.json`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: repository-root paths to `release/models.yaml`, `release/benchmarks.yaml`, and `release/schemas/*.schema.json`.
- Produces: `buildCatalog(options: BuildCatalogOptions): Promise<Catalog>` and `writeCatalog(options: BuildCatalogOptions & { outputPath: string }): Promise<Catalog>`.
- Produces: shared `Catalog`, `ModelRecord`, `BenchmarkRecord`, `MetricRecord`, `CatalogSummary`, and `CatalogValidationError` types.

- [ ] **Step 1: Add the site test/build toolchain and failing builder tests**

Create `site/package.json` with scripts that always generate data before a production build:

```json
{
  "name": "rdk-model-zoo-catalog",
  "private": true,
  "type": "module",
  "engines": { "node": ">=22.12 <23" },
  "scripts": {
    "catalog:build": "tsx scripts/build-catalog.ts",
    "dev": "npm run catalog:build && vite",
    "test": "vitest run",
    "typecheck": "tsc --noEmit",
    "build": "npm run catalog:build && npm run typecheck && vite build",
    "check": "npm test && npm run build"
  },
  "dependencies": {
    "ajv": "8.17.1",
    "ajv-formats": "3.0.1",
    "yaml": "2.8.1"
  },
  "devDependencies": {
    "jsdom": "26.1.0",
    "tsx": "4.20.5",
    "typescript": "5.9.2",
    "vite": "7.1.3",
    "vitest": "3.2.4"
  }
}
```

Use `npm install` inside `site/` to create `package-lock.json`. Configure strict TypeScript with `target: ES2022`, `moduleResolution: Bundler`, DOM libraries, and `noUncheckedIndexedAccess: true`. Configure Vite with `base: "/rdk_model_zoo/"`, and configure Vitest for `environment: "jsdom"` and `setupFiles: ["./tests/setup.ts"]`.

Add `site/node_modules/`, `site/dist/`, and `site/public/data/catalog.json` to `.gitignore`.

Write this first test in `site/tests/catalog-builder.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { fileURLToPath } from "node:url";
import { buildCatalog } from "../scripts/catalog-builder";

describe("buildCatalog", () => {
  it("joins models and benchmarks and preserves the release tag", async () => {
    const catalog = await buildCatalog({
      repositoryRoot: fileURLToPath(new URL("fixtures/", import.meta.url)),
      modelsPath: "models.valid.yaml",
      benchmarksPath: "benchmarks.valid.yaml",
      modelsSchemaPath: "../../../release/schemas/models.schema.json",
      benchmarksSchemaPath: "../../../release/schemas/benchmarks.schema.json"
    });
    expect(catalog.release.tag).toBe("x5-v1.0.0");
    expect(catalog.models[0]?.benchmarks[0]?.sample_id).toBe("convnext");
  });

  it("rejects a benchmark whose sample_id is absent from models.yaml", async () => {
    await expect(buildCatalog({
      repositoryRoot: fileURLToPath(new URL("fixtures/invalid-reference/", import.meta.url)),
      modelsPath: "models.yaml",
      benchmarksPath: "benchmarks.yaml",
      modelsSchemaPath: "../../../../release/schemas/models.schema.json",
      benchmarksSchemaPath: "../../../../release/schemas/benchmarks.schema.json"
    })).rejects.toEqual(expect.objectContaining({
      code: "UNKNOWN_SAMPLE"
    }));
  });
});
```

- [ ] **Step 2: Run the builder tests and verify the missing-module failure**

Run:

```bash
cd site
npm test -- tests/catalog-builder.test.ts
```

Expected: FAIL because `scripts/catalog-builder.ts` and the schemas do not exist.

- [ ] **Step 3: Define exact domain types and JSON Schemas**

Define `site/src/catalog/types.ts` with these public shapes:

```ts
export type Locale = "zh" | "en";
export type MetricUnit = "ms" | "fps" | "percent" | "ratio" | "mae" | "rmse";

export interface MetricRecord {
  metric: string;
  value: number;
  unit: MetricUnit;
  statistic?: "min" | "mean" | "p50" | "p95" | "max";
  scope?: string;
  concurrency?: number;
  dataset?: string;
  model_stage?: "float" | "quantized" | "compiled" | "runtime";
}

export interface BenchmarkRecord {
  id: string;
  sample_id: string;
  variant_id: string;
  display_name: string;
  asset_filename?: string;
  model_format?: string;
  precision?: string;
  input?: { shape?: number[]; layout?: string; format?: string };
  environment: {
    hardware: string;
    rdk_os?: string;
    runtime?: string;
    cpu_mode?: string;
    bpu_cores?: number;
  };
  performance?: MetricRecord[];
  accuracy?: MetricRecord[];
  source: {
    ref: string;
    path: string;
    section: string;
    provenance: "existing-repository-documentation";
  };
}

export interface ModelRecord {
  id: string;
  name: string;
  tasks: string[];
  sample_path: string;
  availability: "download" | "manual";
  download_scripts: string[];
  assets: Array<{ filename: string; format: string; url?: string; sha256?: string | null }>;
  benchmarks: BenchmarkRecord[];
}

export interface Catalog {
  schema_version: number;
  release: Record<string, unknown> & { tag: string; platform: string; version: string };
  summary: Record<string, number>;
  models: ModelRecord[];
}
```

Create Draft 2020-12 JSON Schemas. `models.schema.json` must encode all existing release, compatibility, validation, summary, model, and asset fields while allowing the approved optional presentation fields. `benchmarks.schema.json` must set `additionalProperties: false` on benchmark, environment, metric, input, and source objects; require the IDs, environment hardware, and complete source tuple; constrain numeric values to finite JSON numbers and concurrency/BPU counts to positive integers.

- [ ] **Step 4: Implement YAML loading, Schema validation, and cross-file validation**

Implement the public API in `site/scripts/catalog-builder.ts`:

```ts
export interface BuildCatalogOptions {
  repositoryRoot: string;
  modelsPath: string;
  benchmarksPath: string;
  modelsSchemaPath: string;
  benchmarksSchemaPath: string;
}

export class CatalogValidationError extends Error {
  constructor(public readonly code: string, message: string) {
    super(message);
  }
}

export async function buildCatalog(options: BuildCatalogOptions): Promise<Catalog> {
  const modelsDocument = await readYaml(resolve(options.repositoryRoot, options.modelsPath));
  const benchmarksDocument = await readYaml(resolve(options.repositoryRoot, options.benchmarksPath));
  await validateWithSchema(modelsDocument, resolve(options.repositoryRoot, options.modelsSchemaPath), "MODELS_SCHEMA");
  await validateWithSchema(benchmarksDocument, resolve(options.repositoryRoot, options.benchmarksSchemaPath), "BENCHMARKS_SCHEMA");
  validateReleaseTags(modelsDocument.release.tag, benchmarksDocument.release.tag);
  validateUniqueIds(modelsDocument.models, benchmarksDocument.benchmarks);
  validateModelAndAssetReferences(modelsDocument.models, benchmarksDocument.benchmarks);
  await validateRepositorySources(options.repositoryRoot, benchmarksDocument.benchmarks);
  validateNoYoloe(modelsDocument, benchmarksDocument);
  return joinCatalog(modelsDocument, benchmarksDocument);
}

export async function writeCatalog(options: BuildCatalogOptions & { outputPath: string }): Promise<Catalog> {
  const catalog = await buildCatalog(options);
  await mkdir(dirname(options.outputPath), { recursive: true });
  await writeFile(options.outputPath, `${JSON.stringify(catalog, null, 2)}\n`, "utf8");
  return catalog;
}
```

Source validation must reject absolute paths and `..`, confirm the referenced file exists, and confirm the exact section text appears in that file. `validateNoYoloe` must inspect IDs, names, paths, tags, benchmark IDs, asset filenames, and generated serialized content case-insensitively.

Implement `site/scripts/build-catalog.ts` with paths derived from `import.meta.url`, not the caller's current directory, so `npm run build` works consistently on Windows and Linux.

- [ ] **Step 5: Run builder tests and commit the validated builder**

Run:

```bash
cd site
npm test -- tests/catalog-builder.test.ts
npm run typecheck
git check-ignore public/data/catalog.json
```

Expected: all fixture tests and type checking pass; `git check-ignore` prints `public/data/catalog.json`. The repository catalog build begins in Task 2 after `release/benchmarks.yaml` exists.

Commit:

```bash
git add .gitignore release/schemas site/package.json site/package-lock.json site/tsconfig.json site/vite.config.ts site/vitest.config.ts site/src/catalog/types.ts site/scripts site/tests
git commit -m "feat(catalog): add release data validation"
```

### Task 2: Transcribe existing published performance and accuracy evidence

**Files:**
- Create: `release/benchmarks.yaml`
- Modify: `site/tests/catalog-builder.test.ts`
- Create: `site/tests/fixtures/invalid-reference/models.yaml`
- Create: `site/tests/fixtures/invalid-reference/benchmarks.yaml`
- Modify: `release/README.md`

**Interfaces:**
- Consumes: `BenchmarkRecord` and the Schema/cross-reference rules from Task 1.
- Produces: release-linked benchmark records joined into `Catalog.models[].benchmarks`.

- [ ] **Step 1: Add failing assertions for known published evidence and exclusions**

Extend `site/tests/catalog-builder.test.ts` to build from the real repository and assert exact, already-published evidence:

```ts
const repositoryRoot = fileURLToPath(new URL("../../", import.meta.url));

function buildRepositoryCatalog() {
  return buildCatalog({
    repositoryRoot,
    modelsPath: "release/models.yaml",
    benchmarksPath: "release/benchmarks.yaml",
    modelsSchemaPath: "release/schemas/models.schema.json",
    benchmarksSchemaPath: "release/schemas/benchmarks.schema.json"
  });
}

it("includes exact published evidence and excludes YOLOE", async () => {
  const catalog = await buildRepositoryCatalog();
  const himloco = catalog.models.find((model) => model.id === "himloco");
  const cpp = himloco?.benchmarks.find((record) => record.id === "himloco-cpp-runtime-x5");
  expect(cpp?.performance).toContainEqual(expect.objectContaining({
    metric: "throughput",
    value: 2853.09,
    unit: "fps"
  }));
  const text = JSON.stringify(catalog).toLowerCase();
  expect(text).not.toContain("yoloe");
});

it("keeps plus-suffixed throughput as reported rather than inventing an exact value", async () => {
  const catalog = await buildRepositoryCatalog();
  const atto = catalog.models.find((model) => model.id === "convnext")
    ?.benchmarks.find((record) => record.variant_id === "convnext-atto-224");
  expect(atto?.performance).toContainEqual(expect.objectContaining({
    metric: "throughput",
    value: 732,
    unit: "fps",
    qualifier: "lower-bound"
  }));
});
```

Add `qualifier?: "exact" | "lower-bound" | "upper-bound" | "approximate"` to `MetricRecord` and its Schema because values such as `732+` must retain their published meaning.

- [ ] **Step 2: Run the evidence tests and verify they fail**

Run:

```bash
cd site
npm test -- tests/catalog-builder.test.ts
```

Expected: FAIL because `release/benchmarks.yaml` and the required records do not exist.

- [ ] **Step 3: Inventory every existing source before entering records**

Generate the candidate source list without including YOLOE:

```powershell
rg -l -i -g 'README.md' 'performance|benchmark|latency|fps|top-1|map|miou|mae|rmse|cosine' samples |
  Where-Object { $_ -notmatch '[\\/]yoloe[\\/]' } |
  Sort-Object |
  Set-Content release/benchmark-source-inventory.txt
```

Review root sample README tables first. Use evaluator/runtime/conversion READMEs only for metrics or conditions absent from the root README. Remove duplicate tables that repeat the same values and delete `release/benchmark-source-inventory.txt` after transcription so it does not become a second data source.

- [ ] **Step 4: Create benchmark records without changing the published claims**

Create `release/benchmarks.yaml` with `schema_version: 1`, `release.tag: x5-v1.0.0`, and one record per distinct model variant/test environment. Apply these exact transcription rules:

```text
5.71            -> value: 5.71, qualifier: exact
200+            -> value: 200, qualifier: lower-bound
77.37%          -> value: 77.37, unit: percent, model_stage: float
71.75%          -> value: 71.75, unit: percent, model_stage: quantized
0.999606 cosine -> value: 0.999606, unit: ratio, metric: cosine_similarity
not published   -> omit the metric array; do not create a zero value
```

For classification README tables that provide `Float Top-1` and `Quant Top-1`, use `dataset: ImageNet-1K` only when the same source identifies the dataset. If it does not, omit `dataset` and allow the builder warning. Preserve ConvNeXt's single-frame/single-thread/single-BPU latency scope and four-thread throughput condition. Preserve HiMLoco Python and C++ as separate runtime records because their timing scopes differ.

Every `source.ref` is `x5-v1.0.0`; every path must use `/` separators and point to an English repository document; every section must exactly match the referenced heading.

- [ ] **Step 5: Document the benchmark contract and run full data checks**

Add to `release/README.md`:

```markdown
## Benchmark manifest

`benchmarks.yaml` records performance and accuracy values already published in repository documentation. Each record references a model in `models.yaml` and an immutable release source. Missing fields mean that the source did not publish that condition; they must not be inferred. A metric qualifier preserves claims such as `200+ FPS` as a lower bound.
```

Run:

```bash
cd site
npm test -- tests/catalog-builder.test.ts
npm run catalog:build
rg -ni "yoloe" public/data/catalog.json
```

Expected: tests and generation pass; `rg` exits 1 with no matches.

- [ ] **Step 6: Commit the release benchmark dataset**

```bash
git add release/benchmarks.yaml release/README.md site/src/catalog/types.ts release/schemas/benchmarks.schema.json site/tests
git commit -m "feat(release): add published model benchmarks"
```

### Task 3: Implement deterministic search, filtering, and comparable sorting

**Files:**
- Create: `site/src/catalog/query.ts`
- Create: `site/tests/query.test.ts`
- Create: `site/tests/fixtures/catalog.ts`

**Interfaces:**
- Consumes: `Catalog.models`, localized search text, and `BenchmarkRecord` from Tasks 1–2.
- Produces: `queryModels(models: ModelRecord[], query: CatalogQuery): QueryResult`.
- Produces: `comparisonKey(record: BenchmarkRecord, metric: MetricRecord): string | null`.

- [ ] **Step 1: Write failing behavior tests for search, filters, and comparison safety**

```ts
import { describe, expect, it } from "vitest";
import { comparisonKey, queryModels } from "../src/catalog/query";
import { benchmarkFixture, incomparableModels, modelFixture } from "./fixtures/catalog";

it("matches names, ids, tasks, variants, and asset filenames case-insensitively", () => {
  const result = queryModels([modelFixture], { text: "ATTO", tasks: [], formats: [], precisions: [], benchmark: "all", sort: "name" });
  expect(result.models.map((model) => model.id)).toEqual(["convnext"]);
});

it("returns no comparison key when timing scope or concurrency is missing", () => {
  const record = benchmarkFixture({ environment: { hardware: "RDK X5" } });
  expect(comparisonKey(record, { metric: "throughput", value: 100, unit: "fps" })).toBeNull();
});

it("does not sort incomparable performance records by their numeric values", () => {
  const result = queryModels(incomparableModels, { text: "", tasks: [], formats: [], precisions: [], benchmark: "performance", sort: "fps" });
  expect(result.sortApplied).toBe(false);
  expect(result.reason).toBe("incomparable-benchmarks");
});
```

- [ ] **Step 2: Run the query tests and verify the missing-module failure**

Run `cd site && npm test -- tests/query.test.ts`.

Expected: FAIL because `src/catalog/query.ts` does not exist.

- [ ] **Step 3: Implement normalized search and explicit comparison keys**

Define:

```ts
export interface CatalogQuery {
  text: string;
  tasks: string[];
  formats: string[];
  precisions: string[];
  benchmark: "all" | "performance" | "accuracy" | "none";
  sort: "name" | "latency" | "fps" | "accuracy";
}

export interface QueryResult {
  models: ModelRecord[];
  sortApplied: boolean;
  reason?: "incomparable-benchmarks" | "missing-benchmarks";
}
```

Normalize search with `value.normalize("NFKC").toLocaleLowerCase()`. The comparison key must join hardware, metric, unit, statistic, scope, concurrency, input shape, input layout, input format, and BPU core count. Accuracy keys must additionally include dataset and `model_stage`. Return `null` when a required comparison condition is absent. Numeric sorting only runs when all candidate metrics share one non-null key; otherwise retain name order and return the reason.

- [ ] **Step 4: Run all query tests and commit**

```bash
cd site
npm test -- tests/query.test.ts
npm run typecheck
git add src/catalog/query.ts tests/query.test.ts tests/fixtures/catalog.ts
git commit -m "feat(catalog): add safe model discovery queries"
```

### Task 4: Add bilingual language and theme preferences

**Files:**
- Create: `site/src/i18n/translations.ts`
- Create: `site/src/i18n/language.ts`
- Create: `site/tests/language.test.ts`

**Interfaces:**
- Consumes: browser `navigator.language`, `localStorage`, and `matchMedia`.
- Produces: `createLanguageController(storage: Storage, browserLanguage: string): LanguageController`.
- Produces: `t(locale: Locale, key: TranslationKey, params?: Record<string, string | number>): string`.

- [ ] **Step 1: Write failing language preference tests**

```ts
it("uses Chinese for a zh browser when no preference is stored", () => {
  expect(createLanguageController(localStorage, "zh-CN").current()).toBe("zh");
});

it("persists an explicit English selection over browser language", () => {
  const controller = createLanguageController(localStorage, "zh-CN");
  controller.set("en");
  expect(createLanguageController(localStorage, "zh-CN").current()).toBe("en");
});

it("falls back to English for an unsupported browser language", () => {
  expect(createLanguageController(localStorage, "fr-FR").current()).toBe("en");
});
```

- [ ] **Step 2: Run the tests and verify failure**

Run `cd site && npm test -- tests/language.test.ts`.

Expected: FAIL because the i18n modules do not exist.

- [ ] **Step 3: Implement the controller and complete translation dictionary**

Use storage key `rdk-model-zoo-locale`. Export a closed `TranslationKey` union containing every static string used by summary, filters, cards, details, empty states, errors, qualifiers, units, missing conditions, source labels, and theme/language controls. `t()` must throw during tests for a missing key and substitute `{name}` placeholders from `params`.

Set `<html lang>` whenever the language changes. Keep model names, dataset names, hardware names, metric symbols, and file formats unchanged; translate their explanatory labels.

- [ ] **Step 4: Verify translation parity and commit**

Add a test that compares `Object.keys(translations.zh)` and `Object.keys(translations.en)` and expects exact equality.

Run:

```bash
cd site
npm test -- tests/language.test.ts
npm run typecheck
git add src/i18n tests/language.test.ts
git commit -m "feat(catalog): add bilingual preferences"
```

### Task 5: Render the catalog summary, filters, and model cards

**Files:**
- Create: `site/index.html`
- Create: `site/src/main.ts`
- Create: `site/src/app.ts`
- Create: `site/src/ui/summary.ts`
- Create: `site/src/ui/filters.ts`
- Create: `site/src/ui/model-card.ts`
- Create: `site/tests/app.test.ts`

**Interfaces:**
- Consumes: `/rdk_model_zoo/data/catalog.json`, `queryModels()`, `LanguageController`, and `t()`.
- Produces: `mountCatalog(root: HTMLElement, catalog: Catalog, options: AppOptions): CatalogApp`.
- Produces: semantic DOM for summary, controls, result count, cards, and missing-data states.

- [ ] **Step 1: Write a failing catalog interaction test**

```ts
it("renders release statistics and filters model cards", async () => {
  document.body.innerHTML = '<main id="app"></main>';
  const app = mountCatalog(document.querySelector("#app")!, catalogFixture, { locale: "en" });
  expect(document.querySelector('[data-testid="release-tag"]')?.textContent).toContain("x5-v1.0.0");
  expect(document.querySelectorAll("article[data-model-id]")).toHaveLength(2);
  const search = document.querySelector<HTMLInputElement>('input[type="search"]')!;
  search.value = "HiMLoco";
  search.dispatchEvent(new Event("input", { bubbles: true }));
  expect(document.querySelectorAll("article[data-model-id]")).toHaveLength(1);
  expect(document.querySelector("article")?.textContent).toContain("HiMLoco");
  app.destroy();
});

it("shows explicit performance and accuracy empty states", () => {
  mountCatalog(document.querySelector("#app")!, catalogWithoutBenchmarks, { locale: "en" });
  expect(document.body.textContent).toContain("No published performance data");
  expect(document.body.textContent).toContain("No published accuracy data");
});
```

- [ ] **Step 2: Run the integration test and verify failure**

Run `cd site && npm test -- tests/app.test.ts`.

Expected: FAIL because `mountCatalog` and UI modules do not exist.

- [ ] **Step 3: Implement the semantic document shell and application state**

The document must contain a skip link, `<header>`, `<main id="app">`, and `<footer>`. `main.ts` fetches `${import.meta.env.BASE_URL}data/catalog.json`, validates the HTTP status, parses `Catalog`, and displays a localized recoverable error with links to `release/models.yaml` and the repository when loading fails.

`mountCatalog()` owns one state object:

```ts
interface AppState {
  locale: Locale;
  query: CatalogQuery;
  selectedModelId: string | null;
  theme: "system" | "light" | "dark";
}
```

All event listeners are registered through the app instance and removed by `destroy()` so rerender tests do not leak handlers.

- [ ] **Step 4: Implement summary, controls, and cards**

`summary.ts` derives displayed counts from the built catalog and compares asset/sample counts with the Manifest summary during build. `filters.ts` creates a labeled search input, checkbox/select filters, sort control, result count, and reset button. Filter values come from the actual catalog and are sorted with `Intl.Collator`.

`model-card.ts` renders one `<article>` with model name, task badges, platform, formats, variants, availability, checksum coverage, and representative published latency/FPS/accuracy. It renders the metric qualifier and a concise scope; it does not use “fastest”, “best”, or numeric rank copy. A detail button uses the model name in its accessible label.

- [ ] **Step 5: Run focused and full tests, then commit**

```bash
cd site
npm test -- tests/app.test.ts tests/query.test.ts tests/language.test.ts
npm run typecheck
git add index.html src/main.ts src/app.ts src/ui tests/app.test.ts
git commit -m "feat(catalog): render searchable model cards"
```

### Task 6: Implement shareable model details and evidence tables

**Files:**
- Create: `site/src/ui/model-details.ts`
- Create: `site/tests/model-details.test.ts`
- Modify: `site/src/app.ts`
- Modify: `site/src/ui/model-card.ts`

**Interfaces:**
- Consumes: `ModelRecord`, locale, `release.tag`, and query parameter `model`.
- Produces: `renderModelDetails(model: ModelRecord, context: DetailContext): HTMLElement`.
- Produces: `readModelId(url: URL): string | null` and `writeModelId(url: URL, modelId: string | null): URL`.

- [ ] **Step 1: Write failing detail, URL, and evidence tests**

```ts
it("reads and writes a shareable model query parameter", () => {
  expect(readModelId(new URL("https://example.test/?model=convnext"))).toBe("convnext");
  expect(writeModelId(new URL("https://example.test/"), "himloco").search).toBe("?model=himloco");
});

it("renders metric conditions and immutable source links", () => {
  const element = renderModelDetails(convnextFixture, {
    locale: "en",
    repositoryUrl: "https://github.com/D-Robotics/rdk_model_zoo",
    releaseTag: "x5-v1.0.0"
  });
  expect(element.textContent).toContain("single-frame, single-thread, single-BPU-core");
  const source = element.querySelector<HTMLAnchorElement>('a[data-testid="benchmark-source"]')!;
  expect(source.href).toContain("/blob/x5-v1.0.0/samples/vision/convnext/README.md");
});

it("renders assets without a URL as manual downloads", () => {
  const element = renderModelDetails(manualAssetFixture, detailContext);
  expect(element.textContent).toContain("Manual model required");
  expect(element.querySelector('a[download]')).toBeNull();
});
```

- [ ] **Step 2: Run detail tests and verify failure**

Run `cd site && npm test -- tests/model-details.test.ts`.

Expected: FAIL because `model-details.ts` does not exist.

- [ ] **Step 3: Implement details, history state, and invalid-ID recovery**

Render details in a `<dialog>` when supported and an accessible in-page panel fallback otherwise. Use tables with captions for variants, performance, accuracy, and assets. Every metric row displays value, qualifier, unit, environment, scope, concurrency, input, dataset/model stage when applicable, plus a source link built as:

```ts
const sourceUrl = `${repositoryUrl}/blob/${encodeURIComponent(source.ref)}/${source.path}`;
```

Opening a card calls `history.pushState` with `?model=<id>`; closing removes only the `model` parameter; `popstate` restores the correct view. An unknown ID displays a localized “model not found” message and a button that clears the parameter. Do not construct HTML through untrusted `innerHTML`; create text nodes and attributes with DOM APIs.

- [ ] **Step 4: Run details and application tests, then commit**

```bash
cd site
npm test -- tests/model-details.test.ts tests/app.test.ts
npm run typecheck
git add src/ui/model-details.ts src/app.ts src/ui/model-card.ts tests/model-details.test.ts
git commit -m "feat(catalog): add model benchmark details"
```

### Task 7: Apply the responsive visual system and accessibility behavior

**Files:**
- Create: `site/src/styles.css`
- Modify: `site/index.html`
- Modify: `site/src/app.ts`
- Modify: `site/tests/app.test.ts`

**Interfaces:**
- Consumes: semantic DOM and state from Tasks 5–6.
- Produces: responsive card grid, light/dark variables, visible focus, motion preference handling, and usable 320 px–wide layout.

- [ ] **Step 1: Add failing semantic and control-state assertions**

```ts
it("exposes labeled controls and current filter state", () => {
  mountCatalog(document.querySelector("#app")!, catalogFixture, { locale: "en" });
  expect(document.querySelector('label[for="catalog-search"]')).not.toBeNull();
  expect(document.querySelector('[aria-live="polite"][data-testid="result-count"]')).not.toBeNull();
  expect(document.querySelector('button[aria-label="Switch language"]')).not.toBeNull();
  expect(document.querySelector('button[aria-label="Change color theme"]')).not.toBeNull();
});
```

- [ ] **Step 2: Run the test and verify the missing controls fail**

Run `cd site && npm test -- tests/app.test.ts`.

Expected: FAIL until the language/theme controls and live result status have the required names.

- [ ] **Step 3: Implement the visual tokens and responsive layouts**

Define CSS custom properties for canvas, surface, elevated surface, text, muted text, border, accent, success, warning, focus ring, radius, and shadow under `:root` and `[data-theme="dark"]`. Use a restrained D-Robotics orange accent, neutral card surfaces, and tabular numerals for benchmark values.

Use `grid-template-columns: repeat(auto-fit, minmax(min(100%, 18rem), 1fr))` for cards. At widths below 640 px, stack filters and make detail tables horizontally scroll within a labeled region. Add `:focus-visible` outlines, a functional skip link, minimum 44 px pointer targets, and `@media (prefers-reduced-motion: reduce)` that disables nonessential transitions.

The theme control cycles `system → light → dark`; system mode follows `prefers-color-scheme`. Persist the explicit value under `rdk-model-zoo-theme`.

- [ ] **Step 4: Build and inspect both languages at desktop and mobile widths**

Run:

```bash
cd site
npm test
npm run build
npm run dev -- --host 127.0.0.1
```

Inspect `/rdk_model_zoo/` at 1440×900 and 390×844 in Chinese and English. Verify card text does not overlap, tables remain usable, keyboard focus traverses every control, the dialog returns focus to its opening card, and light/dark contrast remains readable. Record defects directly in `site/tests/app.test.ts` when they can regress through DOM behavior, then fix them before continuing.

- [ ] **Step 5: Commit the complete user interface**

```bash
git add site/index.html site/src/styles.css site/src/app.ts site/tests/app.test.ts
git commit -m "feat(catalog): add responsive accessible styling"
```

### Task 8: Add continuous validation and GitHub Pages deployment

**Files:**
- Create: `.github/workflows/model-catalog-ci.yml`
- Create: `.github/workflows/model-catalog-pages.yml`

**Interfaces:**
- Consumes: `site/package-lock.json`, `npm run check`, GitHub Release events, and `workflow_dispatch.catalog_ref`.
- Produces: required catalog validation status and a GitHub Pages deployment artifact.

- [ ] **Step 1: Add and locally validate the CI workflow**

Create `.github/workflows/model-catalog-ci.yml` with `pull_request` and `push` path filters for `site/**`, `release/models.yaml`, `release/benchmarks.yaml`, `release/schemas/**`, and the workflow itself. Use:

```yaml
permissions:
  contents: read
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 22
          cache: npm
          cache-dependency-path: site/package-lock.json
      - run: npm ci
        working-directory: site
      - run: npm run check
        working-directory: site
```

Parse both workflow files with Ruby's built-in YAML parser while preserving GitHub's `on` key as a string:

```bash
ruby -e "require 'yaml'; YAML.load_file('.github/workflows/model-catalog-ci.yml', aliases: true); puts 'workflow yaml ok'"
```

- [ ] **Step 2: Add release/manual Pages deployment with immutable refs**

Create `.github/workflows/model-catalog-pages.yml` with triggers:

```yaml
on:
  release:
    types: [published]
  workflow_dispatch:
    inputs:
      catalog_ref:
        description: Git ref containing the catalog site and release data
        required: true
        default: rdk_x5
```

Set `contents: read`, `pages: write`, and `id-token: write`; set environment `github-pages`; cancel superseded runs through a `pages` concurrency group. Checkout `${{ github.event.release.tag_name }}` for a release event and `${{ inputs.catalog_ref }}` for manual runs. Run `npm ci` and `npm run check`, then use `actions/configure-pages@v5`, `actions/upload-pages-artifact@v3` with `site/dist`, and `actions/deploy-pages@v4`.

Before upload, assert that `site/dist/data/catalog.json` contains a non-empty `release.tag`. For release events, assert it equals `${{ github.event.release.tag_name }}`. This prevents publishing branch data under the wrong release label.

- [ ] **Step 3: Run local equivalents and commit workflows**

```bash
cd site
npm ci
npm run check
cd ..
ruby -e "require 'yaml'; Dir['.github/workflows/model-catalog-*.yml'].each { |p| YAML.load_file(p, aliases: true) }; puts 'workflow yaml ok'"
git add .github/workflows
git commit -m "ci(catalog): validate and deploy GitHub Pages"
```

### Task 9: Link the catalog into repository and release documentation

**Files:**
- Modify: `README.md`
- Modify: `README_cn.md`
- Modify: `docs/RELEASE.md`
- Modify: `docs/RELEASE_cn.md`
- Modify: `release/README.md`

**Interfaces:**
- Consumes: public Pages URL and workflow names from Task 8.
- Produces: discoverable catalog links and repeatable release-operator instructions.

- [ ] **Step 1: Add the public catalog entry to both top-level READMEs**

Add `Online model catalog` / `在线模型目录` beside the current release links. Describe that cards show published models, assets, performance, accuracy, and test conditions, and state that missing metrics have not been inferred.

Use the exact URL `https://d-robotics.github.io/rdk_model_zoo/` in both files.

- [ ] **Step 2: Add the release and recovery procedure in both languages**

Append these operational checks to the release policy:

```text
1. Confirm models.yaml and benchmarks.yaml carry the new Release Tag.
2. Run npm ci && npm run check from site/.
3. Publish the GitHub Release and wait for model-catalog-pages.yml.
4. Open the Pages URL and verify the displayed Tag and model/asset counts.
5. If deployment failed, fix the source and publish a corrective commit/tag; use workflow_dispatch only to retry an unchanged approved ref.
```

State that the catalog is a presentation layer; the YAML Manifest, Release assets, and model READMEs remain the authoritative downloadable/reproducible artifacts.

- [ ] **Step 3: Check bilingual links and commit documentation**

```bash
rg -n "d-robotics.github.io/rdk_model_zoo|model-catalog-pages|benchmarks.yaml" README.md README_cn.md docs/RELEASE.md docs/RELEASE_cn.md release/README.md
git diff --check
git add README.md README_cn.md docs/RELEASE.md docs/RELEASE_cn.md release/README.md
git commit -m "docs(catalog): document the online model directory"
```

### Task 10: Complete release-grade verification and publish the first catalog

**Files:**
- Modify only files required to correct failures found by the checks above.

**Interfaces:**
- Consumes: all outputs from Tasks 1–9 and repository admin access.
- Produces: a clean reviewed commit series and the public GitHub Pages site.

- [ ] **Step 1: Run the complete local verification from a clean dependency install**

```bash
cd site
npm ci
npm run check
cd ..
git diff --check
git status --short
```

Expected: all tests, type checking, catalog validation, and production build pass; only deliberate tracked changes are present before commits, and the final status is clean after commits.

- [ ] **Step 2: Verify generated release facts**

```powershell
$catalog = Get-Content -Raw site/dist/data/catalog.json | ConvertFrom-Json
if ($catalog.release.tag -ne 'x5-v1.0.0') { throw 'Unexpected catalog release tag' }
if (($catalog.models | Where-Object { $_.id -match '(?i)yoloe' }).Count -ne 0) { throw 'YOLOE leaked into catalog' }
if ($catalog.models.Count -ne $catalog.summary.sample_count) { throw 'Model count mismatch' }
$assetCount = ($catalog.models | ForEach-Object { $_.assets.Count } | Measure-Object -Sum).Sum
if ($assetCount -ne $catalog.summary.asset_count) { throw 'Asset count mismatch' }
Write-Output "Catalog facts verified: $($catalog.models.Count) models, $assetCount assets, tag $($catalog.release.tag)"
```

- [ ] **Step 3: Review the branch diff against the approved specification**

Run:

```bash
git log --oneline 0564ead..HEAD
git diff --stat 0564ead..HEAD
git diff --check 0564ead..HEAD
```

Review every spec acceptance criterion against a concrete test, generated value, workflow guard, or browser inspection. Fix any uncovered requirement and commit the focused correction before publishing.

- [ ] **Step 4: Push the reviewed branch and enable GitHub Pages Actions**

```bash
git push origin rdk_x5
gh api --method POST repos/D-Robotics/rdk_model_zoo/pages -f build_type=workflow
```

If the Pages endpoint reports that the site already exists, inspect it with `gh api repos/D-Robotics/rdk_model_zoo/pages`; require `build_type: workflow` before continuing.

- [ ] **Step 5: Run and monitor the initial manual deployment**

```bash
gh workflow run model-catalog-pages.yml --repo D-Robotics/rdk_model_zoo -f catalog_ref=rdk_x5
gh run list --repo D-Robotics/rdk_model_zoo --workflow model-catalog-pages.yml --limit 1
```

Watch the returned run with `gh run watch <run-id> --repo D-Robotics/rdk_model_zoo --exit-status`. Expected: conclusion `success` and a Pages deployment URL under `https://d-robotics.github.io/rdk_model_zoo/`.

- [ ] **Step 6: Verify the public deployment**

Open the public URL and verify:

```text
Release tag: x5-v1.0.0
Model count: equals release/models.yaml summary.sample_count
Asset count: equals release/models.yaml summary.asset_count
YOLOE search: zero results and no source text occurrence
Languages: Chinese and English both render and persist
Direct URL: ?model=convnext survives refresh
Evidence: ConvNeXt and HiMLoco source links resolve under blob/x5-v1.0.0
Mobile: 390 px viewport has no page-wide horizontal overflow
```

Use `curl.exe -fsSL https://d-robotics.github.io/rdk_model_zoo/data/catalog.json` to confirm the deployed JSON tag and counts. Stop after these checks pass; do not add unrelated site features.
