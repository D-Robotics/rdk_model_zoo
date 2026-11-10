import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import vm from 'node:vm';

const context = { window: {} };
vm.runInNewContext(readFileSync(new URL('../../src/facets.js', import.meta.url), 'utf8'), context);
const count = context.window.CatalogFacets.modelCount;
const models = [
  { catalogId: 'vision/a/cls', sample: 'vision/a/cls/n', platforms: ['X5', 'S100'] },
  { catalogId: 'vision/a/cls', sample: 'vision/a/cls/s', platforms: ['S100'] },
  { catalogId: 'vision/b/seg', sample: 'vision/b/seg/tiny', platforms: ['S600'] },
];

test('counts model variants once across their supported chips', () => {
  assert.equal(count(models), 3);
  assert.equal(count(models, new Set(['S100'])), 2);
  assert.equal(count(models, new Set(['X5', 'S100'])), 2);
  assert.equal(count(models, new Set(['X5', 'S600'])), 2);
  assert.equal(count(models, new Set(['S100P'])), 0);
  assert.equal(count([]), 0);
});

test('per-chip rows for the same model count once, including when multiple chips are selected', () => {
  const releases = ['X5', 'S100', 'S100P', 'S600'].map(releasePlatform => ({
    catalogId: 'vision/a/cls', sample: 'vision/a/cls/n', releasePlatform,
  }));
  assert.equal(count(releases), 1);
  assert.equal(count(releases, ['S100']), 1);
  assert.equal(count(releases, ['S100', 'S600']), 1);
});

test('duplicate rows do not inflate the count and distinct quantized variants remain distinct', () => {
  const small = { catalogId: 'llm/a/text', modelSize: '2B', platforms: ['S600'] };
  assert.equal(count([
    { ...small, sample: 'llm/a/text/2b-w4' },
    { ...small, sample: 'llm/a/text/2b-w4' },
    { ...small, sample: 'llm/a/text/2b-w8' },
  ]), 2);
});
