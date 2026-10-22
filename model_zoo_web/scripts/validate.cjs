const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const { assertCoverResolves, assertLocalImageFile } = require('./cover-assets.cjs');

const HERO_WIDTH = 1672;
const HERO_HEIGHT = 941;

// Detail heroes used to ship as <figure class="mz-demo is-frame">, and the
// matching CSS padded the bitmap with a white mat. Pending and published
// pages share one figure class, so the generated files must not grow that
// class or rule back.
function assertDetailHeroMarkup(detailJs) {
  const heroFiles = [
    'index.html', 'reports.html', 'detail-view.js', 'detail.css', 'product-ui.css',
    'style.css', 'gallery.css', 'reference.css', 'refresh.css',
  ];
  for (const filename of heroFiles) {
    const source = fs.readFileSync(path.join(root, filename), 'utf8');
    assert.doesNotMatch(
      source,
      /is-frame/,
      `${filename} must not put an is-frame class or rule on detail heroes`,
    );
  }
  const figures = [...detailJs.matchAll(/<figure\b[^>]*>/g)].map(match => match[0]);
  assert.deepEqual(
    figures,
    ['<figure class="mz-demo">'],
    'pending and published detail pages must use the same <figure class="mz-demo"> hero',
  );
}

function rasterSize(bytes, label) {
  if (bytes[0] === 0x89 && bytes.toString('ascii', 1, 4) === 'PNG') {
    assert.equal(bytes.toString('ascii', 12, 16), 'IHDR', `${label}: PNG is missing IHDR`);
    return { width: bytes.readUInt32BE(16), height: bytes.readUInt32BE(20) };
  }
  assert.equal(bytes.toString('ascii', 0, 4), 'RIFF', `${label}: cover is not a PNG or WebP`);
  assert.equal(bytes.toString('ascii', 8, 12), 'WEBP', `${label}: cover is not a WebP`);
  const chunk = bytes.toString('ascii', 12, 16);
  if (chunk === 'VP8 ') {
    assert.equal(bytes[20] & 1, 0, `${label}: VP8 cover is not a keyframe`);
    assert.deepEqual([bytes[23], bytes[24], bytes[25]], [0x9d, 0x01, 0x2a], `${label}: VP8 start code mismatch`);
    return {
      width: bytes.readUInt16LE(26) & 0x3fff,
      height: bytes.readUInt16LE(28) & 0x3fff,
    };
  }
  if (chunk === 'VP8L') {
    assert.equal(bytes[20], 0x2f, `${label}: VP8L signature mismatch`);
    const bits = bytes.readUInt32LE(21);
    return { width: 1 + (bits & 0x3fff), height: 1 + ((bits >>> 14) & 0x3fff) };
  }
  if (chunk === 'VP8X') {
    return {
      width: 1 + (bytes[24] | (bytes[25] << 8) | (bytes[26] << 16)),
      height: 1 + (bytes[27] | (bytes[28] << 8) | (bytes[29] << 16)),
    };
  }
  assert.fail(`${label}: unsupported WebP chunk ${JSON.stringify(chunk)}`);
}

function assertGeneratedCoverSizes() {
  const directory = path.join(root, 'assets', 'models');
  const covers = [];
  function walk(dir) {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) walk(fullPath);
      else if (/\.(?:png|webp)$/i.test(entry.name)) covers.push(fullPath);
    }
  }
  walk(directory);
  assert(covers.length > 0, 'the preview must ship generated PNG or WebP covers');
  for (const filePath of covers) {
    const label = path.relative(root, filePath);
    const size = rasterSize(fs.readFileSync(filePath), label);
    assert.equal(size.width, HERO_WIDTH, `${label} must be ${HERO_WIDTH}px wide`);
    assert.equal(size.height, HERO_HEIGHT, `${label} must be ${HERO_HEIGHT}px tall`);
  }
  return covers.length;
}

const preview = process.argv.includes('--preview');
const webRoot = path.resolve(__dirname, '..');
const repoRoot = path.resolve(webRoot, '..');
const root = path.join(webRoot, 'dist');
const requiredFiles = [
  'index.html', 'app.js', 'data.js', 'style.css', 'gallery.css',
  'reference.css', 'detail.css', 'ask-ai.css', 'ask-ai.js', 'product-ui.css', 'refresh.css', 'i18n.js', 'facets.js',
  'model-properties.js', 'detail-view.js', 'reports.html', 'reports.js',
  'reports.css', 'reports/reports-data.js', 'reports/inventory.json',
];

for (const filename of requiredFiles) {
  assert(fs.statSync(path.join(root, filename)).size > 0, `${filename} is missing or empty`);
}

const index = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
assert.match(index, /<body class="compare">/);
assert.match(index, /detail-view\.js/);
assert.match(index, /model-properties\.js/);
assert.match(index, /ask-ai\.js/);

const reportPage = fs.readFileSync(path.join(root, 'reports.html'), 'utf8');
assert.match(reportPage, /reports\/reports-data\.js/);
assert.match(reportPage, /reports\.js/);

const pagesCompatibleFiles = [
  'index.html', 'reports.html', 'app.js', 'detail-view.js', 'reports.js',
  'style.css', 'gallery.css', 'reference.css', 'detail.css', 'ask-ai.css', 'ask-ai.js', 'product-ui.css', 'refresh.css', 'reports.css',
];
for (const filename of pagesCompatibleFiles) {
  const source = fs.readFileSync(path.join(root, filename), 'utf8');
  assert.doesNotMatch(
    source,
    /(?:href|src)\s*=\s*["']\/(?!\/)/i,
    `${filename} must use relative local URLs for GitHub Pages project paths`,
  );
  assert.doesNotMatch(
    source,
    /url\(\s*["']?\/(?!\/)/i,
    `${filename} must use relative CSS URLs for GitHub Pages project paths`,
  );
}

const context = { window: {} };
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(root, 'data.js'), 'utf8'), context);
vm.runInContext(fs.readFileSync(path.join(root, 'reports/reports-data.js'), 'utf8'), context);
vm.runInContext(fs.readFileSync(path.join(root, 'model-properties.js'), 'utf8'), context);

const data = context.window.MODEL_DATA;
const reportEntries = Array.from(context.window.OE_REPORTS);

if (preview) {
  assert(data.models.length >= 1, 'The preview site must contain at least one released model');
  assert.equal(data.catalog.status, 'published', 'The preview catalog must be published');
  const modelIds = new Set(data.models.map(model => model.id));
  assert.equal(modelIds.size, data.models.length, 'Model ids must be unique');

  const checkedCovers = new Set();
  for (const model of data.models) {
    // Catalog cards (app.js) and detail heroes (detail-view.js) both request
    // coverImage, falling back to the same relative placeholder.
    if (!checkedCovers.has(model.coverImage)) {
      assertCoverResolves(model.coverImage, root, `${model.id} catalog cover and detail hero`);
      checkedCovers.add(model.coverImage);
    }
    if (model.releaseStatus === 'data-pending') {
      assert.equal(model.dataPending, true, `${model.id}: pending models must set dataPending`);
      assert.deepEqual(Array.from(model.assets), [], `${model.id}: pending models must not publish artifacts`);
      assert.equal(model.benchmark?.documentation, true, `${model.id}: pending benchmarks are documentation excerpts`);
      for (const row of [...(model.benchmark.performance || []), ...(model.benchmark.accuracy || []), ...(model.benchmark.stages || []), ...(model.benchmark.extras || [])]) {
        assert.ok(row.metric && row.unit && Number.isFinite(row.value), `${model.id}: documented metric needs metric, unit and value`);
        assert.match(String(row.source || ''), /^https:\/\/github\.com\/D-Robotics\/rdk_model_zoo\/blob\/[0-9a-f]{40}\//, `${model.id}: documented metric must cite a blob`);
        if (row.model_stage !== undefined) assert.ok(['float', 'quantized', 'board'].includes(row.model_stage), `${model.id}: unknown model stage`);
        if (row.concurrency !== undefined) assert.ok(Number.isInteger(row.concurrency) && row.concurrency > 0, `${model.id}: concurrency must be a positive integer from the source`);
      }
      if (model.coverImage.startsWith('assets/models/pending-heroes/')) {
        const runtime = model.heroProvenance?.runtime;
        assert.ok(
          runtime === 'display-composition' || runtime === 'host-float-reference',
          `${model.id}: pending hero runtime must be display-composition or host-float-reference`,
        );
        assert.match(model.heroProvenance?.revision || '', /^[0-9a-f]{40}$/, `${model.id}: composed hero needs the pinned revision`);
        assert.match(model.heroProvenance?.input || '', /^samples\/.+\.(?:jpg|jpeg|png|webp)$/i, `${model.id}: composed hero needs the source image`);
        assert.ok(model.heroProvenance?.kind, `${model.id}: composed hero needs a kind`);
        if (runtime === 'host-float-reference') {
          assert.match(model.heroProvenance?.note || '', /not board output/, `${model.id}: host float hero must say it is not board output`);
          assert.match(model.heroProvenance?.modelUrl || '', /^https:\/\//, `${model.id}: host float hero needs the checkpoint URL`);
          assert.ok(model.heroProvenance?.framework, `${model.id}: host float hero needs a framework`);
          assert.equal(model.coverLabel, '浮点参考', `${model.id}: host float hero must not use a board result label`);
          if (model.heroProvenance.kind === 'detection-overlay' || model.heroProvenance.kind === 'segmentation-overlay') {
            const threshold = Number(model.heroProvenance.confidenceThreshold);
            assert.ok(
              threshold >= 0.5 && threshold <= 1,
              `${model.id}: detection or segmentation hero needs a confidence threshold of at least 0.5`,
            );
          }
        }
      } else if (model.coverImage !== 'assets/pending-cover.svg') {
        assert.fail(`${model.id}: pending cover must be a reviewed local hero or the shared placeholder`);
      }
      for (const download of model.documentedDownloads || []) {
        assert.match(download.url || '', /^https:\/\//, `${model.id}: documented download must be HTTPS`);
        assert.ok(download.filename, `${model.id}: documented download needs a filename`);
      }
      assert.ok(model.platforms?.length, `${model.id}: pending models need a supported board`);
      assert.match(model.source || '', /^https:\/\/github\.com\/D-Robotics\/rdk_model_zoo\/tree\/[0-9a-f]{40}\/samples\//, `${model.id}: pending models must link a sample tree`);
      if (model.licenseUrl) assert.match(model.licenseUrl, /^https:\/\//, `${model.id}: licenseUrl must use HTTPS`);
      continue;
    }
    assert(model.licenseName, `${model.id}: licenseName is required`);
    assert.match(model.licenseUrl || '', /^https:\/\//, `${model.id}: licenseUrl must use HTTPS`);
    for (const asset of model.assets || []) {
      assert(asset.url && asset.sha256 && asset.sizeBytes, `${model.id}: assets need url, sha256 and size_bytes`);
    }
    if (model.reportDataUrl) {
      const dataPath = path.join(root, model.reportDataUrl);
      assert(fs.statSync(dataPath).size > 0, `${model.id}: report data payload is missing`);
      const oeData = JSON.parse(fs.readFileSync(dataPath, 'utf8'));
      assert.equal(oeData.schema_version, 1, `${model.id}: unsupported OE data schema_version`);
      assert.equal(
        oeData.provenance?.artifact_sha256,
        model.assets[0]?.sha256,
        `${model.id}: OE data provenance does not match the shipped artifact`,
      );
    }
  }

  for (const entry of reportEntries) {
    assert(modelIds.has(entry.modelIds[0]), `Report ${entry.id} references an unknown model`);
    if (entry.dataUrl) {
      assert(fs.statSync(path.join(root, entry.dataUrl)).size > 0, `Report ${entry.id}: data payload is missing`);
    } else {
      assert(entry.path, `Report ${entry.id} needs either dataUrl or an original-report path`);
    }
  }
  const modelsWithData = data.models.filter(model => model.reportDataUrl).length;
  assert(
    modelsWithData === 0 || reportEntries.some(entry => entry.dataUrl),
    'Models reference report data but no report entry carries it',
  );
} else {
  assert.deepEqual(Array.from(data.models), [], 'The pre-migration catalog must contain no models');
  assert.equal(data.catalog.status, 'migration-pending');
  assert(Object.values(data.catalog.summary).every(value => value === 0));
  assert.deepEqual(reportEntries, [], 'The pre-migration site must contain no OE reports');
  assert.deepEqual(Object.keys(context.window.MODEL_PROPERTIES), [], 'Demo model properties must not be bundled');
}

const inventory = JSON.parse(fs.readFileSync(path.join(root, 'reports/inventory.json'), 'utf8'));
if (preview) {
  assert.equal(inventory.reports.length, reportEntries.length, 'inventory.json must match the report registry');
} else {
  assert.deepEqual(inventory.reports, []);
}

const assetFiles = [];
function walk(directory) {
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const fullPath = path.join(directory, entry.name);
    if (entry.isDirectory()) walk(fullPath);
    else assetFiles.push(path.relative(path.join(root, 'assets'), fullPath));
  }
}
assertLocalImageFile(path.join(root, 'assets', 'pending-cover.svg'), 'assets/pending-cover.svg');

walk(path.join(root, 'assets'));
assert(assetFiles.includes(path.join('rdk-brand', 'logo.png')));
const allowedAssets = filename =>
  filename.startsWith(`icons${path.sep}`)
  || filename === 'pending-cover.svg'
  || filename === path.join('rdk-brand', 'logo.png')
  || (preview && filename.startsWith(`models${path.sep}`));
assert(
  assetFiles.every(allowedAssets),
  `Unexpected model/demo asset in build: ${assetFiles.join(', ')}`,
);

for (const filename of ['app.js', 'ask-ai.js', 'i18n.js', 'facets.js', 'model-properties.js', 'detail-view.js', 'reports.js']) {
  const source = fs.readFileSync(path.join(root, filename), 'utf8');
  new vm.Script(source, { filename });
  if (!preview) {
    assert.doesNotMatch(source, /clip-cover|demo-v1/i, `${filename} still contains demo-model data`);
  }
}

const appSource = fs.readFileSync(path.join(root, 'app.js'), 'utf8');
const detailSource = fs.readFileSync(path.join(root, 'detail-view.js'), 'utf8');
assertDetailHeroMarkup(detailSource);
assert.match(detailSource, /class="mz-report-placeholder" aria-disabled="true">OE 转换报告/);
if (!preview) {
  assert.match(appSource, /当前目录暂未发布任何模型。/);
  assert.doesNotMatch(appSource, /OE 报告|正式模型迁移尚未开始|模型迁移尚未开始|nav-placeholder/);
  assert.doesNotMatch(appSource, /href="reports\.html"/);
}
assert.doesNotMatch(detailSource, /external\(['"]reports\.html['"]/);

// Every local script and stylesheet carries a version derived from the
// built file's content (stamped by scripts/asset-versions.mjs).
for (const page of ['index.html', 'reports.html']) {
  const html = fs.readFileSync(path.join(root, page), 'utf8');
  for (const [, asset, version] of html.matchAll(/(?:href|src)="((?!https?:|\/\/)[^"?#]+\.(?:css|js))(?:\?v=([^"]*))?"/g)) {
    const digest = crypto.createHash('sha256').update(fs.readFileSync(path.join(root, asset))).digest('hex').slice(0, 10);
    assert.equal(version, digest, `${page}: ${asset} must carry its content version ?v=${digest}`);
  }
}

// Catalog labels come from data, not from the page templates, so a new task
// can ship without an English string. Check them against the dictionary.
const i18nContext = { window: {} };
vm.createContext(i18nContext);
vm.runInContext(fs.readFileSync(path.join(root, 'i18n.js'), 'utf8'), i18nContext);
vm.runInContext(fs.readFileSync(path.join(root, 'facets.js'), 'utf8'), i18nContext);
const { english } = i18nContext.window.HubI18n;
const catalogLabels = new Set([
  ...i18nContext.window.CatalogFacets.groups.flatMap(group => [group.label, ...group.tasks.map(([, label]) => label)]),
  ...data.models.map(model => model.task),
]);
const untranslated = [...catalogLabels].filter(label => /[\u4e00-\u9fff]/.test(label) && /[\u4e00-\u9fff]/.test(english(label)));
assert.deepEqual(untranslated, [], `i18n.js is missing English labels: ${untranslated.join(', ')}`);

const packageJson = JSON.parse(fs.readFileSync(path.join(webRoot, 'package.json'), 'utf8'));
assert.equal(packageJson.dependencies, undefined);
assert.doesNotMatch(fs.readFileSync(path.join(webRoot, 'scripts/generate.mjs'), 'utf8'), /\.\.\/\.\.\/|docs\/release/);

const localReportPayloads = fs.readdirSync(path.join(root, 'reports')).filter(filename => filename.endsWith('.html'));
assert.deepEqual(localReportPayloads, [], 'Full OE HTML reports must remain outside the website artifact root');
const localReportDirs = fs.readdirSync(path.join(root, 'reports'), { withFileTypes: true })
  .filter(entry => entry.isDirectory()).map(entry => entry.name);
if (preview) {
  assert(localReportDirs.includes('data'), 'The preview build must ship its report data payloads');
  const coverCount = assertGeneratedCoverSizes();
  console.log(`PASS: ${coverCount} generated covers are ${HERO_WIDTH}x${HERO_HEIGHT}, and detail heroes have no is-frame`);
} else {
  assert.deepEqual(localReportDirs, [], 'The pre-migration site must not ship report payloads');
}

console.log(preview
  ? `PASS: Model Zoo preview with ${data.models.length} model(s), ${reportEntries.length} report(s), data payloads verified.`
  : 'PASS: Model Zoo builds with 0 models, 0 reports, no catalog OE entry, and no demo assets.');
