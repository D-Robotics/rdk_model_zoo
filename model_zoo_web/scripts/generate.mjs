import { cp, mkdir, rm, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptRoot = dirname(fileURLToPath(import.meta.url));
const webRoot = resolve(scriptRoot, '..');
const sourceRoot = resolve(webRoot, 'src');
const outputRoot = resolve(webRoot, 'dist');
const repositoryUrl = 'https://github.com/D-Robotics/rdk_model_zoo';

const data = {
  schemaVersion: 1,
  catalog: {
    status: 'migration-pending',
    summary: {
      sample_count: 0,
      asset_count: 0,
      downloadable_asset_count: 0,
      benchmark_count: 0,
    },
  },
  repository: {
    url: repositoryUrl,
  },
  release: {
    status: 'unpublished',
    compatibility: {},
  },
  models: [],
};

const sourceFiles = [
  'index.html',
  'app.js',
  'style.css',
  'gallery.css',
  'reference.css',
  'detail.css',
  'i18n.js',
  'facets.js',
  'model-properties.js',
  'detail-view.js',
  'reports.html',
  'reports.js',
  'reports.css',
];

await rm(outputRoot, { recursive: true, force: true });
await mkdir(outputRoot, { recursive: true });

for (const filename of sourceFiles) {
  await cp(resolve(sourceRoot, filename), resolve(outputRoot, filename));
}

await cp(resolve(webRoot, 'public'), outputRoot, { recursive: true });
await writeFile(resolve(outputRoot, 'data.js'), `window.MODEL_DATA = ${JSON.stringify(data, null, 2)};\n`, 'utf8');
await mkdir(resolve(outputRoot, 'reports'), { recursive: true });
await writeFile(resolve(outputRoot, 'reports/reports-data.js'), 'window.OE_REPORTS = [];\n', 'utf8');
await writeFile(
  resolve(outputRoot, 'reports/inventory.json'),
  `${JSON.stringify({ schemaVersion: 1, reports: [] }, null, 2)}\n`,
  'utf8',
);

console.log('Built the Model Zoo shell with 0 models and 0 OE reports.');
