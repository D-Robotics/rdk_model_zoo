import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

// Local scripts and stylesheets referenced by the HTML pages, with or without
// an existing "?v=" query.
export const localAssetPattern = /((?:href|src)=")((?!https?:|\/\/)[^"?#]+\.(?:css|js))(?:\?v=[^"]*)?(")/g;
export const pages = ['index.html', 'reports.html'];
export const assetVersion = bytes => createHash('sha256').update(bytes).digest('hex').slice(0, 10);

// Stamp every local asset URL with a digest of the built file, so browsers
// refetch exactly the files that changed. Run after the last write to the
// output directory; stamping again replaces the earlier versions.
export async function stampAssetVersions(outputRoot) {
  for (const page of pages) {
    const path = resolve(outputRoot, page);
    const html = await readFile(path, 'utf8');
    const versions = new Map();
    for (const [, , asset] of html.matchAll(localAssetPattern)) {
      if (!versions.has(asset)) versions.set(asset, assetVersion(await readFile(resolve(outputRoot, asset))));
    }
    const stamped = html.replace(localAssetPattern, (_, prefix, asset, suffix) => `${prefix}${asset}?v=${versions.get(asset)}${suffix}`);
    if (stamped !== html) await writeFile(path, stamped, 'utf8');
  }
}
