import { rename, rm, stat } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const finalRoot = resolve(root, 'dist-candidates-passed');
const nextRoot = resolve(root, '.dist-candidates-passed-next');
const backupRoot = resolve(root, '.dist-candidates-passed-previous');
if (!(await stat(nextRoot).then(result => result.isDirectory()).catch(() => false))) {
  throw new Error('Validated staging preview directory is missing');
}
await rm(backupRoot, { recursive: true, force: true });
const hadPrevious = await stat(finalRoot).then(result => result.isDirectory()).catch(() => false);
if (hadPrevious) await rename(finalRoot, backupRoot);
try {
  await rename(nextRoot, finalRoot);
} catch (error) {
  if (hadPrevious) await rename(backupRoot, finalRoot).catch(() => {});
  throw error;
}
await rm(backupRoot, { recursive: true, force: true });
console.log('Promoted the validated mixed preview to dist-candidates-passed.');
