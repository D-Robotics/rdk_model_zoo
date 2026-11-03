// Compiled text demos have source and preparation instructions, without board metrics.
import { cp, mkdir, readFile, readdir } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { extname, resolve } from 'node:path';

async function sampleFiles(root, prefix) {
  const files = [];
  for (const entry of await readdir(resolve(root, prefix), { withFileTypes: true })) {
    if (['prepared', 'build', 'results', '__pycache__'].includes(entry.name)) continue;
    const name = `${prefix}/${entry.name}`;
    if (entry.isDirectory()) files.push(...await sampleFiles(root, name));
    else if (entry.isFile() && (['.md', '.py', '.cc', '.hpp', '.json', '.sh', '.txt'].includes(extname(name)) || ['.gitignore', 'LICENSE'].includes(entry.name))) files.push(name);
  }
  return files.sort();
}

export async function textDemoEntry({ record, variant, release, webRoot, outputRoot, id, coverFilename }) {
  const root = resolve(webRoot, '..');
  const manifest = JSON.parse(await readFile(resolve(root, release.model_manifest), 'utf8'));
  const board = JSON.parse(await readFile(resolve(root, release.board_result), 'utf8'));
  if (board.status !== 'not-run' || ['ppl', 'ttft_ms', 'decode_tokens_per_second', 'peak_memory_bytes', 'generation_checks', 'environment']
    .some(field => board[field] !== null)) throw new Error('Untested text demos must retain null board measurements');
  const docsRoot = resolve(outputRoot, 'docs', id);
  await mkdir(docsRoot, { recursive: true });
  const documents = {};
  const documentsEn = {};
  for (const [key, directory] of [['sample', ''], ['runtime', 'runtime/cpp/'], ['conversion', 'conversion/'], ['evaluation', 'evaluator/'], ['model', 'model/']]) {
    const dest = `${key}.md`;
    await cp(resolve(root, record.sample_path, directory, 'README_cn.md'), resolve(docsRoot, dest));
    documents[key] = `docs/${id}/${dest}`;
    await cp(resolve(root, record.sample_path, directory, 'README.md'), resolve(docsRoot, `${key}-en.md`));
    documentsEn[key] = `docs/${id}/${key}-en.md`;
  }
  const downloadDir = resolve(outputRoot, 'downloads', id);
  await mkdir(downloadDir, { recursive: true });
  const archive = 'gemma4-text-sample-source.tar.gz';
  // Bundle the current sample source so an uncommitted integration never gets a fake GitHub link.
  const files = await sampleFiles(root, record.sample_path);
  execFileSync('tar', ['-czf', resolve(downloadDir, archive), '-C', root, ...files]);
  const policy = manifest.conversion;
  const published = manifest.availability?.package_url ? manifest.availability : null;
  const assets = published ? [{ role: 'S600 部署模型包', format: 'tar.gz',
    filename: published.name, url: published.package_url, sha256: published.sha256,
    sizeBytes: published.size_bytes, checksumsUrl: published.checksums_url }] : [];
  const preparation = published ? [
    'mkdir -p /data/gemma4-e2b-sdk2',
    `curl -fL --retry 3 -o ${published.name} ${published.package_url}`,
    `printf '%s  %s\\n' '${published.sha256}' '${published.name}' | sha256sum -c -`,
    `tar -xzf ${published.name} --strip-components=1 -C /data/gemma4-e2b-sdk2`,
    `python3 ${record.sample_path}/model/prepare.py \\`,
    '  --source-dir /data/gemma4-e2b-sdk2 \\',
    '  --output-dir /data/gemma4-e2b-sdk2',
  ].join('\n') : null;
  return {
    model: {
      id, catalogId: record.id, name: `${record.name} ${variant.size.toUpperCase()}`,
      variantName: `${record.name} ${variant.size.toUpperCase()} W4/W8`,
      family: record.family, modelSize: variant.size, releaseStatus: 'demo', releasePlatform: 'S600', hardware: 'RDK S600',
      task: '文本生成', taskId: 'text-generation', tasks: ['text-generation'], platforms: ['S600'],
      description: record.description.zh, descriptionEn: record.description.en,
      coverImage: `assets/models/${coverFilename}`, coverLabel: '文本对话示意',
      sample: `${record.id}/${variant.size}`, samplePath: record.sample_path,
      source: null, sourceArchive: `downloads/${id}/${archive}`, documents, documentsEn,
      upstreamModelUrl: `https://huggingface.co/${manifest.source.model_id}/tree/${manifest.source.revision}`,
      licenseName: record.license.name, licenseUrl: record.license.url,
      shape: 'Text', precisionPolicy: policy.precision_policy,
      reportUrl: null, reportDataUrl: null, reportSourceUrl: null,
      assets, benchmark: null, boardMeasurements: board,
      deployment: {
        sdk: policy.sdk, runtime: manifest.runtime.type, target: policy.march, cores: policy.cores,
        contextLength: policy.context_length, prefillChunkSize: policy.prefill_chunk_size,
        compileStatus: policy.status, boardStatus: board.status,
        fileBytes: manifest.files.reduce((sum, file) => sum + file.bytes, 0),
        visionCompanionRequired: manifest.runtime.vision_companion_required,
        artifactManifest: manifest,
      },
      quickStart: [
        { title: '准备模型文件', command: preparation || `python3 ${record.sample_path}/model/prepare.py \\\n  --source-dir /data/gemma4-language \\\n  --companion-dir /data/Gemma4-E2B_w4 \\\n  --output-dir /data/gemma4-e2b-sdk2` },
        { title: '运行文本生成', command: `export OELLM_RUNTIME_ROOT=/opt/oellm_runtime_s600\nbash ${record.sample_path}/runtime/cpp/run.sh \\\n  --config_path /data/gemma4-e2b-sdk2/oellm_config.json \\\n  --prompt "请把 hello 翻译成中文，只回答译文。" --max_new_tokens 64` },
      ],
    },
    properties: {
      inputKind: 'text', contextLength: policy.context_length, prefillChunkSize: policy.prefill_chunk_size,
      sdkRuntime: `${policy.sdk} · ${manifest.runtime.type}`, precisionPolicy: policy.precision_policy,
      deploymentBytes: manifest.files.reduce((sum, file) => sum + file.bytes, 0),
      visionCompanionRequired: manifest.runtime.vision_companion_required,
      modelSource: manifest.source.model_id, sourceRevision: manifest.source.revision,
    },
  };
}
