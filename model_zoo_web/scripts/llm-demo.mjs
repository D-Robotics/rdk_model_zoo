// SDK references remain separate from local board measurements.
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

const tasks = {
  'text-generation': { id: 'text-generation', label: '文本生成', kind: 'text', cover: '文本对话示意' },
  'text-gen': { id: 'text-generation', label: '文本生成', kind: 'text', cover: '文本对话示意' },
  vlm: { id: 'vision-language-model', label: '视觉语言模型', kind: 'image-text', cover: '图文理解示意' },
  asr: { id: 'speech-recognition', label: '语音识别', kind: 'audio', cover: '语音识别示意' },
};

export async function textDemoEntry({ record, variant, release, webRoot, outputRoot, id, coverFilename }) {
  const root = resolve(webRoot, '..');
  const manifest = JSON.parse(await readFile(resolve(root, release.model_manifest), 'utf8'));
  const official = release.status === 'official';
  const sdkCatalog = record.sample_path === 'samples/llm/oellm';
  const task = tasks[record.task];
  if (!task) throw new Error('Unsupported SDK model task: ' + record.task);
  const references = official ? JSON.parse(await readFile(resolve(root, release.official_benchmark), 'utf8')) : null;
  const reference = references?.profiles[manifest.profile];
  const officialBenchmark = official ? (reference.metrics ? reference : { source: references.source, metrics: reference }) : null;
  const board = JSON.parse(await readFile(resolve(root, release.board_result), 'utf8'));
  if (board.status !== 'not-run' || ['ppl', 'ttft_ms', 'decode_tokens_per_second', 'peak_memory_bytes', 'generation_checks', 'environment', ...(sdkCatalog ? ['rtf'] : [])]
    .some(field => board[field] !== null)) throw new Error('Local board measurements must remain not-run and null');
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
  const archive = sdkCatalog ? 'oellm-sample-source.tar.gz' : 'gemma4-text-sample-source.tar.gz';
  const files = await sampleFiles(root, record.sample_path);
  execFileSync('tar', ['-czf', resolve(downloadDir, archive), '-C', root, ...files]);
  const policy = manifest.conversion;
  const published = manifest.availability?.package_url ? manifest.availability : null;
  const assets = published ? [{ role: official ? 'S600 官方模型包' : 'S600 部署模型包', format: official ? 'zip' : 'tar.gz',
    filename: published.name, url: published.package_url, sha256: published.sha256,
    sizeBytes: published.size_bytes, checksumsUrl: published.checksums_url }] : [];
  const preparation = official ? [
    `MODEL_DIR=/data/${published.archive_root || published.name.slice(0, -4)}`,
    `python3 ${record.sample_path}/model/prepare.py ${sdkCatalog ? '' : '--profile ' + manifest.profile + ' '}\\`,
    `  ${sdkCatalog ? '--profile ' + manifest.profile + ' ' : ''}--source-dir "$MODEL_DIR"${sdkCatalog ? '' : ' --output-dir "$MODEL_DIR"'}`,
  ].join('\n') : published ? [
    'mkdir -p /data/gemma4-e2b-sdk2',
    `curl -fL --retry 3 -o ${published.name} ${published.package_url}`,
    `printf '%s  %s\\n' '${published.sha256}' '${published.name}' | sha256sum -c -`,
    `tar -xzf ${published.name} --strip-components=1 -C /data/gemma4-e2b-sdk2`,
    `python3 ${record.sample_path}/model/prepare.py \\`,
    '  --source-dir /data/gemma4-e2b-sdk2 \\',
    '  --output-dir /data/gemma4-e2b-sdk2',
  ].join('\n') : null;
  const sdkRun = [
    'export OELLM_RUNTIME_ROOT=/opt/oellm_runtime_s600',
    `bash ${record.sample_path}/runtime/cpp/run.sh --config_path "$MODEL_DIR/oellm_config.json" \\`,
    task.kind === 'audio' ? '  --audio_path "$OELLM_RUNTIME_ROOT/resource/test_asr.flac"'
      : task.kind === 'image-text' ? '  --image_path "$OELLM_RUNTIME_ROOT/resource/image.jpg" --prompt "描述图片"'
        : '  --prompt "你好"',
  ].join('\n');
  const license = manifest.source.license || record.license;
  return {
    model: {
      id, catalogId: record.id, name: official ? record.name : `${record.name} ${(manifest.model_size || variant.size).toUpperCase()}`,
      variantName: official ? (sdkCatalog ? manifest.display_name : `${record.name} ${manifest.model_size.toUpperCase()} ${policy.precision_policy.toUpperCase()}`) : `${record.name} ${variant.size.toUpperCase()} W4/W8`,
      family: record.family, modelSize: manifest.model_size || variant.size, releaseStatus: official ? 'official' : 'demo', releasePlatform: 'S600', hardware: 'RDK S600',
      task: task.label, taskId: task.id, tasks: [task.id], platforms: ['S600'],
      description: record.description.zh, descriptionEn: record.description.en,
      coverImage: `assets/models/${coverFilename}`, coverLabel: task.cover,
      sample: `${record.id}/${variant.size}`, samplePath: record.sample_path,
      source: official ? `https://github.com/D-Robotics/rdk_model_zoo/tree/${record.sample_revision}/${record.sample_path}` : null,
      sampleRevision: official ? record.sample_revision : null, sourceArchive: `downloads/${id}/${archive}`, documents, documentsEn,
      upstreamModelUrl: `https://huggingface.co/${manifest.source.model_id}${manifest.source.revision ? '/tree/' + manifest.source.revision : ''}`,
      licenseName: license.name, licenseUrl: license.url,
      shape: task.kind === 'audio' ? 'Audio' : task.kind === 'image-text' ? 'Image + Text' : 'Text', precisionPolicy: policy.precision_policy,
      reportUrl: null, reportDataUrl: null, reportSourceUrl: null,
      assets, benchmark: null, boardMeasurements: board, officialBenchmark,
      deployment: {
        sdk: policy.sdk, runtime: manifest.runtime.type, target: policy.march, cores: policy.cores,
        contextLength: policy.context_length, prefillChunkSize: policy.prefill_chunk_size,
        compileStatus: policy.status, boardStatus: board.status,
        fileBytes: manifest.files.reduce((sum, file) => sum + file.bytes, 0),
        visionCompanionRequired: manifest.runtime.vision_companion_required,
        acceleration: policy.acceleration || 'standard', artifactManifest: manifest,
      },
      quickStart: [
        { title: official ? '下载解压后准备模型' : '准备模型文件', command: preparation || `python3 ${record.sample_path}/model/prepare.py \\\n  --source-dir /data/gemma4-language \\\n  --companion-dir /data/Gemma4-E2B_w4 \\\n  --output-dir /data/gemma4-e2b-sdk2` },
        { title: task.kind === 'audio' ? '运行语音识别' : task.kind === 'image-text' ? '运行图文理解' : '运行文本生成', command: sdkCatalog ? sdkRun : official
          ? `export OELLM_RUNTIME_ROOT=/opt/oellm_runtime_s600\nbash ${record.sample_path}/runtime/cpp/run.sh \\\n  --config_path "$MODEL_DIR/oellm_config.json" --prompt "你好" --max_new_tokens 64`
          : `export OELLM_RUNTIME_ROOT=/opt/oellm_runtime_s600\nbash ${record.sample_path}/runtime/cpp/run.sh \\\n  --config_path /data/gemma4-e2b-sdk2/oellm_config.json \\\n  --prompt "请把 hello 翻译成中文，只回答译文。" --max_new_tokens 64` },
      ],
    },
    properties: {
      inputKind: manifest.input?.kind || task.kind, contextLength: policy.context_length, prefillChunkSize: policy.prefill_chunk_size,
      sampleRate: manifest.input?.sample_rate, sdkRuntime: official ? 'OellmRuntime' : `${policy.sdk} · ${manifest.runtime.type}`, precisionPolicy: policy.precision_policy,
      deploymentBytes: manifest.files.reduce((sum, file) => sum + file.bytes, 0),
      visionCompanionRequired: manifest.runtime.vision_companion_required,
      modelSource: manifest.source.model_id, sourceRevision: manifest.source.revision,
    },
  };
}
