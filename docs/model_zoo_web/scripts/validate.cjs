const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const preview = process.argv.includes('--preview');
const webRoot = path.resolve(__dirname, '..');
const root = path.join(webRoot, 'dist');
const requiredFiles = [
  'index.html', 'app.js', 'data.js', 'style.css', 'gallery.css',
  'reference.css', 'detail.css', 'i18n.js', 'facets.js',
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

const reportPage = fs.readFileSync(path.join(root, 'reports.html'), 'utf8');
assert.match(reportPage, /reports\/reports-data\.js/);
assert.match(reportPage, /reports\.js/);

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

for (const filename of ['app.js', 'i18n.js', 'facets.js', 'model-properties.js', 'detail-view.js', 'reports.js']) {
  const source = fs.readFileSync(path.join(root, filename), 'utf8');
  new vm.Script(source, { filename });
  if (!preview) {
    assert.doesNotMatch(source, /yolo|convnext|efficient[-_ ]?sam|clip-cover|demo-v1/i, `${filename} still contains demo-model data`);
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
