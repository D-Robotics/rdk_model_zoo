import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { createRequire } from 'node:module';
import path from 'node:path';
import { test } from 'node:test';
import { buildPendingModels } from '../pending-preview.mjs';

const require = createRequire(import.meta.url);
const { assertCoverResolves, assertDistinctModelCovers, assertLocalImageFile, assertWellFormedSvg } = require('../cover-assets.cjs');

const webRoot = path.resolve(import.meta.dirname, '../..');
const placeholder = path.join(webRoot, 'public/assets/pending-cover.svg');
const revision = '7a97c1780b0cfa131ff0007d662e255e5e41c12e';

test('shared placeholder is a relative SVG a browser can paint', () => {
  const text = readFileSync(placeholder, 'utf8');
  assertWellFormedSvg(text, 'pending-cover.svg');
  assertLocalImageFile(placeholder, 'pending-cover.svg');
  assert.match(text, /<svg[\s>]/);
  assert.doesNotMatch(text, /^\s*<\?xml/m);
  for (const source of ['src/app.js', 'src/detail-view.js', 'scripts/pending-preview.mjs']) {
    const page = readFileSync(path.join(webRoot, source), 'utf8');
    assert.match(page, /assets\/pending-cover\.svg/);
    assert.doesNotMatch(page, /["']\/assets\/pending-cover\.svg/);
  }
});

test('catalog and detail covers reject remote images and missing local files', () => {
  const present = `https://raw.githubusercontent.com/D-Robotics/rdk_model_zoo/${revision}/samples/vision/bytetrack/test_data/bus.jpg`;
  assert.throws(() => assertCoverResolves(present, webRoot, 'bytetrack'), /local/);
  assert.throws(
    () => assertCoverResolves('assets/missing-cover.webp', webRoot, 'missing hero'),
    /cover 404/,
  );
  assert.throws(
    () => assertCoverResolves('/assets/pending-cover.svg', webRoot, 'pages path'),
    /GitHub Pages/,
  );
  assert.throws(
    () => assertWellFormedSvg('<svg aria-label="\u0001broken"></svg>', 'broken'),
    /illegal XML control characters/,
  );
});

test('different model families cannot ship the same cover bytes under different filenames', () => {
  const directory = mkdtempSync(path.join(tmpdir(), 'model-covers-'));
  try {
    writeFileSync(path.join(directory, 'one.webp'), 'one recorded banner');
    writeFileSync(path.join(directory, 'two.webp'), 'one recorded banner');
    const rows = [
      { catalogId: 'one/one/cls', coverImage: 'one.webp' },
      { catalogId: 'two/two/cls', coverImage: 'two.webp' },
    ];
    assert.throws(() => assertDistinctModelCovers(rows, directory), /same cover/);
    assertDistinctModelCovers([rows[0], { ...rows[1], catalogId: rows[0].catalogId }], directory);
    writeFileSync(path.join(directory, 'two.webp'), 'another recorded banner');
    assertDistinctModelCovers(rows, directory);
    assertDistinctModelCovers(rows.map(row => ({ ...row, coverImage: 'assets/pending-cover.svg' })), directory);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('a source example image uses the shared placeholder until a local hero is supplied', () => {
  const record = {
    registration: 'pending', id: 'fixture/fixture/track', source: 'fixture', family: 'fixture',
    name: 'Fixture', task: 'track', sample_revision: revision, sample_path: 'samples/vision/fixture',
    supported_platforms: [{ platform: 'x5' }], example_image: { path: 'samples/vision/fixture/bus.jpg' },
  };
  const options = {
    officialRepositoryUrl: 'https://github.com/D-Robotics/rdk_model_zoo',
    taskRegistry: JSON.parse(readFileSync(path.join(webRoot, 'task-registry.json'), 'utf8')),
    safeId: (...parts) => parts.join('-'),
  };
  const [model] = buildPendingModels([record], options);
  assert.equal(model.coverImage, 'assets/pending-cover.svg');
  assert.equal(model.coverLabel, '封面待补充');
  const inferenceCommit = 'a'.repeat(40);
  record.hero_visual = {
    file: 'model_zoo_web/release/assets/pending-heroes/fixture.webp',
    kind: 'similarity-panel', input: 'samples/vision/fixture/input.png',
    commit: inferenceCommit, runtime: 'host-float-reference',
    inference_receipt_version: 2,
  };
  const [withHero] = buildPendingModels([record], options);
  assert.equal(withHero.coverImage, 'assets/models/pending-heroes/fixture.webp');
  assert.equal(withHero.heroProvenance.revision, inferenceCommit);
  assert.equal(withHero.heroProvenance.receiptVersion, 2);
});
