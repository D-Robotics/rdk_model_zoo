import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const webRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const script = resolve(webRoot, 'scripts/generate-preview.mjs');

test('local drafts cannot enter technical promotion mode', () => {
  const result = spawnSync(process.execPath, [script, '--local-candidate', '/tmp/draft',
    '--technical-passed', '--snapshot-dir', '/tmp/snapshot', '--audit-path', '/tmp/audit',
    '--review-path', '/tmp/review'], { encoding: 'utf8' });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /cannot be combined/);
});

test('local drafts reject a catalog claiming released status before generating output', async () => {
  const directory = await mkdtemp(resolve(tmpdir(), 'mobilenet-preview-test-'));
  try {
    await writeFile(resolve(directory, 'catalog.json'), JSON.stringify({
      source: 'local-candidate', status: 'released', models: [],
    }));
    const result = spawnSync(process.execPath, [script, '--local-candidate', directory], { encoding: 'utf8' });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /Catalog status does not match candidate/);
  } finally {
    await rm(directory, { recursive: true });
  }
});

test('local drafts cannot be combined with snapshot candidate previews', () => {
  const result = spawnSync(process.execPath, [script, '--local-candidate', '/tmp/draft', '--candidates'],
    { encoding: 'utf8' });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /cannot be combined/);
});

test('local drafts cannot replace a released catalog model', async () => {
  const directory = await mkdtemp(resolve(tmpdir(), 'mobilenet-preview-test-'));
  try {
    const record = { id: 'timm/mobilenetv4/cls', variants: [] };
    await writeFile(resolve(directory, 'catalog.json'), JSON.stringify({
      source: 'local-candidate', status: 'candidate', models: [record],
    }));
    await writeFile(resolve(directory, 'released.json'), JSON.stringify({
      source: 'model_zoo_web/data', models: [record],
    }));
    const result = spawnSync(process.execPath, [script, '--local-candidate', directory], {
      encoding: 'utf8',
      env: { ...process.env, MODEL_ZOO_CATALOG: resolve(directory, 'released.json') },
    });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /collides with a released catalog model/);
  } finally {
    await rm(directory, { recursive: true });
  }
});
