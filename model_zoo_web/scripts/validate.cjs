const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const preview = process.argv.includes('--preview');
const webRoot = path.resolve(__dirname, '..');
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

  for (const model of data.models) {
    assert(model.coverImage, `${model.id}: coverImage is required`);
    assert(fs.statSync(path.join(root, model.coverImage)).size > 0, `${model.id}: cover image is missing`);
    assert(model.licenseName, `${model.id}: licenseName is required`);
    assert.match(model.licenseUrl || '', /^https:\/\//, `${model.id}: licenseUrl must use HTTPS`);
    for (const asset of model.assets || []) {
      if (model.entryKind === 'sample') {
        // Sample artifacts come from publisher manifests; a digest may be
        // legitimately absent (sha256_provenance: not-provided upstream).
        assert(asset.url && asset.filename, `${model.id}: sample assets need url and filename`);
        assert(!asset.sha256 || /^[0-9a-f]{64}$/.test(asset.sha256), `${model.id}: sample asset sha256 must be hex`);
      } else {
        assert(asset.url && asset.sha256 && asset.sizeBytes, `${model.id}: assets need url, sha256 and size_bytes`);
      }
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

  // Sample-profile entries render their own support payload instead of the
  // released-model benchmark data; hold them to the sample contract.
  const sampleEntries = data.models.filter(model => model.entryKind === 'sample');
  assert.equal(
    data.catalog.summary.sample_profile_count, sampleEntries.length,
    'catalog.summary.sample_profile_count must match the sample entries',
  );
  const neuralFormats = ['bin', 'hbm', 'onnx', 'pt'];
  for (const model of sampleEntries) {
    assert(model.sampleProfile, `${model.id}: sample entries carry a sampleProfile payload`);
    assert(!model.benchmark, `${model.id}: sample entries must not carry benchmark data`);
    assert.match(model.releaseStatus, /available/, `${model.id}: sample platforms are available, never released`);
    const availability = model.sampleProfile.availability || {};
    if (availability.status === 'manual-compile') {
      assert.match(availability.guideUrl || '', /^https:\/\//, `${model.id}: manual-compile needs an HTTPS guide`);
      // data.js runs in a vm context; copy into a host array before comparing.
      assert.deepEqual(Array.from(model.assets || []), [], `${model.id}: manual-compile publishes no artifact`);
    } else if (availability.status === 'published-download') {
      assert(model.assets.length >= 1, `${model.id}: published-download lists its artifacts`);
      // Mirror the catalog's split-detector contract in the built preview: a
      // pair ships both halves, and neural artifacts keep the delivery line's
      // published format (bin on X5, hbm on the S series).
      const roles = model.assets.map(asset => asset.role);
      const pairPresent = !roles.includes('parta') && !roles.includes('partb')
        || (roles.includes('parta') && roles.includes('partb'));
      assert(pairPresent, `${model.id}: a published split detector carries both parta and partb`);
      const expectedFormat = model.releasePlatform === 'X5' ? 'bin' : 'hbm';
      for (const asset of model.assets) {
        if (['parta', 'partb', 'detector'].includes(asset.role)) {
          assert.equal(asset.format, expectedFormat,
            `${model.id}: ${asset.role} must be a ${expectedFormat} file on ${model.releasePlatform}`);
        }
      }
    } else if (availability.status === 'no-neural-artifact') {
      assert(model.assets.every(asset => asset.role === 'upstream-source' && !neuralFormats.includes(asset.format)),
        `${model.id}: a platform without a neural artifact only ships its pinned source`);
    }
    for (const asset of model.assets || []) {
      assert(asset.format !== 'pt' || availability.status !== 'no-neural-artifact',
        `${model.id}: no fake neural artifacts on CPU trackers`);
    }
    for (const link of model.sampleProfile.evidence || []) {
      if (link.url.startsWith('sample-evidence/')) {
        assert(fs.statSync(path.join(root, link.url)).size > 0,
          `${model.id}: evidence asset ${link.url} is missing from the build`);
      }
    }
  }
  // ByteTrack S100P has no published detector: the page must say so.
  const s100p = sampleEntries.find(model => /^bytetrack-bytetrack-tracking-.*-s100p$/.test(model.id));
  if (s100p) {
    assert.equal(s100p.sampleProfile.availability.status, 'manual-compile');
    assert.match(s100p.sampleProfile.availability.guideUrl, /conversion$/, 'the S100P guide points at the conversion directory');
  }
  // Evidence stays a summary layer: small text payloads only, no log dumps.
  const evidenceDir = path.join(root, 'sample-evidence');
  if (fs.existsSync(evidenceDir)) {
    for (const filename of fs.readdirSync(evidenceDir)) {
      assert(/[.]json$|[.]log$/.test(filename), `sample-evidence ships only json/log summaries: ${filename}`);
      const size = fs.statSync(path.join(evidenceDir, filename)).size;
      assert(size > 0 && size < 64 * 1024, `sample-evidence payload is not a small summary: ${filename}`);
    }
  }
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
walk(path.join(root, 'assets'));
assert(assetFiles.includes(path.join('rdk-brand', 'logo.png')));
const allowedAssets = filename =>
  filename.startsWith(`icons${path.sep}`)
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
} else {
  assert.deepEqual(localReportDirs, [], 'The pre-migration site must not ship report payloads');
}

console.log(preview
  ? `PASS: Model Zoo preview with ${data.models.length} model(s), ${reportEntries.length} report(s), data payloads verified.`
  : 'PASS: Model Zoo builds with 0 models, 0 reports, no catalog OE entry, and no demo assets.');
