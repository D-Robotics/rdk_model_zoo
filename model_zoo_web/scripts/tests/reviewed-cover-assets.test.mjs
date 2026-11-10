import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { applyReviewedCovers } from '../reviewed-cover-assets.mjs';

test('one reviewed cover replaces every variant without promoting pending models or changing facts', async () => {
  const root = await mkdtemp(join(tmpdir(), 'reviewed-covers-'));
  try {
    const id = 'example/model/seg';
    const entry = { cover: 'art.webp', kind: 'task-illustration', cover_label: '任务示意图', provenance: 'provenance.json' };
    await writeFile(join(root, 'art.webp'), 'test artwork bytes');
    await writeFile(join(root, 'provenance.json'), JSON.stringify({ schema_version: 1, reviewed_at: '2026-11-11', items: { [id]: { ...entry, note: 'Illustrative artwork, not model output.' } } }));
    const models = [
      { id: 'a', catalogId: id, releaseStatus: 'converted', coverImage: 'old.webp', heroProvenance: { runtime: 'host-float-reference' }, assets: [{ sha256: 'unchanged' }] },
      { id: 'b', catalogId: id, releaseStatus: 'data-pending', dataPending: true, coverImage: 'assets/pending-cover.svg', benchmark: { documentation: true } },
      { id: 'c', catalogId: 'other/model/cls', coverImage: 'other.webp', releaseStatus: 'released' },
    ];
    const facts = value => value.map(({ coverImage, coverLabel, heroProvenance, coverProvenance, ...rest }) => rest);
    const before = structuredClone(facts(models));
    await applyReviewedCovers({ models, covers: { [id]: entry }, inputRoot: root, assetsRoot: join(root, 'built') });
    assert.deepEqual(facts(models), before);
    assert.equal(models[0].coverImage, models[1].coverImage);
    assert.equal(models[0].heroProvenance, undefined);
    assert.equal(models[1].coverProvenance.kind, 'task-illustration');
    assert.equal(models[2].coverImage, 'other.webp');
    assert.equal(await readFile(join(root, 'built/reviewed-art.webp'), 'utf8'), 'test artwork bytes');
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('unknown model artwork fails instead of silently skipping an approved cover', async () => {
  await assert.rejects(applyReviewedCovers({ models: [], covers: { unknown: {} }, inputRoot: '.', assetsRoot: '.' }), /unknown model/);
});
