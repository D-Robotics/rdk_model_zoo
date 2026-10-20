import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { test } from 'node:test';

const require = createRequire(import.meta.url);
const { assertCoverResolves, assertLocalImageFile, assertWellFormedSvg } = require('../cover-assets.cjs');

const webRoot = path.resolve(import.meta.dirname, '../..');
const repoRoot = path.resolve(webRoot, '..');
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

test('catalog and detail covers 404 when the blob or local file is missing', () => {
  const present = `https://raw.githubusercontent.com/D-Robotics/rdk_model_zoo/${revision}/samples/vision/bytetrack/test_data/bus.jpg`;
  assert.doesNotThrow(() => assertCoverResolves(present, webRoot, repoRoot, 'bytetrack'));
  const missing = `https://raw.githubusercontent.com/D-Robotics/rdk_model_zoo/${revision}/samples/vision/bytetrack/test_data/missing-cover.jpg`;
  assert.throws(
    () => assertCoverResolves(missing, webRoot, repoRoot, 'missing hero'),
    /cover would 404/,
  );
  assert.throws(
    () => assertCoverResolves('/assets/pending-cover.svg', webRoot, repoRoot, 'pages path'),
    /GitHub Pages/,
  );
  assert.throws(
    () => assertWellFormedSvg('<svg aria-label="\u0001broken"></svg>', 'broken'),
    /illegal XML control characters/,
  );
});
