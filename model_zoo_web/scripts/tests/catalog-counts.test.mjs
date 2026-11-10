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

test('counts supported variant/chip combinations, not a Cartesian product of four chips', () => {
  assert.equal(count(models), 4);
  assert.equal(count(models, new Set(['S100'])), 2);
  assert.equal(count(models, new Set(['X5', 'S600'])), 2);
  assert.equal(count(models, new Set(['S100P'])), 0);
  assert.equal(count([]), 0);
});

test('duplicate rows do not inflate the count and distinct quantized variants remain distinct', () => {
  const small = { catalogId: 'llm/a/text', modelSize: '2B', platforms: ['S600'] };
  assert.equal(count([
    { ...small, sample: 'llm/a/text/2b-w4' },
    { ...small, sample: 'llm/a/text/2b-w4' },
    { ...small, sample: 'llm/a/text/2b-w8' },
  ]), 2);
});
