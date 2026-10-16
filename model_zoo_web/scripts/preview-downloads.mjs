import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, isAbsolute, relative, resolve } from 'node:path';

const manifestSpecs = [
  { path: 'release/preview-downloads/yolo26-cls-seg-20260930.json', tasks: ['cls', 'seg'] },
  { path: 'release/preview-downloads/yolo26-pose-obb-20260930.json', tasks: ['pose', 'obb'] },
];
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const requireMatch = (condition, message) => {
  if (!condition) throw new Error(`Preview downloads: ${message}`);
};

async function loadManifest({ webRoot, snapshotId, auditSha, catalog, selectedReleaseIds }, spec) {
  const manifestPath = resolve(webRoot, spec.path);
  const bytes = await readFile(manifestPath).catch(error => {
    if (error.code === 'ENOENT') return null;
    throw error;
  });
  if (!bytes) return null;
  const manifest = JSON.parse(bytes.toString('utf8'));
  requireMatch(manifest.schema_version === 1 && manifest.kind === 'verified-preview-model-downloads',
    'unsupported manifest schema');
  requireMatch(manifest.snapshot_id === snapshotId, 'manifest belongs to another snapshot');
  requireMatch(manifest.source_hashes?.authoritative_audit_sha256 === auditSha,
    'manifest audit SHA does not match frozen evidence');
  if (manifest.tasks) {
    requireMatch(JSON.stringify(manifest.tasks) === JSON.stringify(spec.tasks),
      'manifest task scope does not match its download group');
  }
  const expected = new Map((catalog.models || []).filter(record => spec.tasks.includes(record.task))
    .flatMap(record => record.variants.flatMap(variant => variant.platforms.map(release =>
      [`${record.id}/${variant.size}/${release.platform}`, release.artifact])))
    .filter(([id]) => !selectedReleaseIds || selectedReleaseIds.has(id)));
  requireMatch(expected.size > 0, 'no selected models exist for this download group');
  requireMatch(manifest.artifacts && Object.keys(manifest.artifacts).length === expected.size,
    'manifest must cover exactly the selected models in its task group');
  for (const [id, artifact] of expected) {
    const download = manifest.artifacts[id];
    requireMatch(download?.status === 'verified', `${id} is not verified`);
    for (const field of ['filename', 'format', 'sha256', 'size_bytes']) {
      requireMatch(download[field] === artifact[field], `${id} ${field} differs from frozen artifact`);
    }
    requireMatch(/^[a-f0-9]{64}$/.test(download.sha256) && download.size_bytes > 0,
      `${id} is missing model identity`);
    const key = `models/${id}/${artifact.filename}`;
    requireMatch(download.url === `https://rdk-model-zoo.oss-cn-beijing.aliyuncs.com/${key}`,
      `${id} must use its stable public OSS model URL`);
    requireMatch(typeof download.upload_receipt_path === 'string'
      && !isAbsolute(download.upload_receipt_path), `${id} has an invalid receipt path`);
    const receiptPath = resolve(dirname(manifestPath), download.upload_receipt_path);
    const receiptRelative = relative(dirname(manifestPath), receiptPath);
    requireMatch(receiptRelative && !receiptRelative.startsWith('..') && !isAbsolute(receiptRelative),
      `${id} receipt escapes the manifest directory`);
    const receiptBytes = await readFile(receiptPath);
    requireMatch(sha256(receiptBytes) === download.upload_receipt_sha256,
      `${id} upload receipt SHA mismatch`);
    const receipt = JSON.parse(receiptBytes.toString('utf8'));
    requireMatch(receipt.status === 'verified' && receipt.acl === 'public-read'
      && receipt.key === key && receipt.url === download.url
      && receipt.sha256 === download.sha256 && receipt.size_bytes === download.size_bytes,
    `${id} upload receipt does not verify this public model`);
  }
  return {
    artifacts: manifest.artifacts,
    selection: {
      manifest_path: spec.path,
      manifest_sha256: sha256(bytes),
      verified_models: expected.size,
    },
  };
}

export async function loadVerifiedPreviewDownloads(params) {
  const loaded = [];
  for (const spec of manifestSpecs) {
    const manifest = await loadManifest(params, spec);
    if (manifest) loaded.push(manifest);
  }
  if (!loaded.length) return null;
  const artifacts = {};
  for (const manifest of loaded) {
    for (const [id, artifact] of Object.entries(manifest.artifacts)) {
      requireMatch(!Object.hasOwn(artifacts, id), `duplicate download identity: ${id}`);
      artifacts[id] = artifact;
    }
  }
  return {
    artifacts,
    selection: loaded.length === 1 ? loaded[0].selection : {
      manifests: loaded.map(manifest => manifest.selection),
      verified_models: Object.keys(artifacts).length,
    },
  };
}

export function previewDownloadAssets(downloads, releaseId, platform) {
  const asset = downloads?.artifacts[releaseId];
  return asset ? [{
    role: `${platform} 部署模型`,
    format: asset.format,
    filename: asset.filename,
    url: asset.url,
    sha256: asset.sha256,
    sizeBytes: asset.size_bytes,
  }] : [];
}
