import { cp, mkdir, readFile } from 'node:fs/promises';
import { basename, extname, resolve } from 'node:path';

const kinds = new Set(['task-illustration', 'host-float-reference', 'recorded-board-reference']);

/** Apply explicitly reviewed artwork after all family/official/pending defaults. */
export async function applyReviewedCovers({ models, covers = {}, inputRoot, assetsRoot }) {
  if (!covers || typeof covers !== 'object' || Array.isArray(covers)) {
    throw new Error('MODEL_ZOO_INPUTS.reviewed_covers must be keyed by catalog model id');
  }
  const byCatalog = new Map();
  for (const model of models) {
    if (!byCatalog.has(model.catalogId)) byCatalog.set(model.catalogId, []);
    byCatalog.get(model.catalogId).push(model);
  }
  for (const [id, entry] of Object.entries(covers)) {
    const targets = byCatalog.get(id);
    if (!targets?.length) throw new Error(`Reviewed cover references unknown model ${id}`);
    if (!entry || !kinds.has(entry.kind) || !entry.cover_label?.trim()
        || typeof entry.cover !== 'string' || extname(entry.cover) !== '.webp'
        || typeof entry.provenance !== 'string') {
      throw new Error(`${id}: reviewed cover needs WebP artwork, label, kind and provenance`);
    }
    const provenance = JSON.parse(await readFile(resolve(inputRoot, entry.provenance), 'utf8'));
    const receipt = provenance.items?.[id];
    if (provenance.schema_version !== 1 || receipt?.kind !== entry.kind
        || receipt.cover !== entry.cover || typeof provenance.reviewed_at !== 'string' || !provenance.reviewed_at) {
      throw new Error(`${id}: reviewed cover differs from its artwork record`);
    }
    const filename = `reviewed-${basename(entry.cover)}`;
    await mkdir(assetsRoot, { recursive: true });
    await cp(resolve(inputRoot, entry.cover), resolve(assetsRoot, filename));
    for (const model of targets) {
      model.coverImage = `assets/models/${filename}`;
      model.coverLabel = entry.cover_label;
      // The previous hero receipt describes the previous picture, not this one.
      delete model.heroProvenance;
      model.coverProvenance = {
        kind: entry.kind, reviewedAt: provenance.reviewed_at,
        source: `model_zoo_web/release/${entry.provenance}`, note: receipt.note,
      };
    }
  }
}
