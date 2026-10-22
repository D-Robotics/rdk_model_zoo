import { execFileSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const blobCache = new Map();
const SIZE_TOKENS = [
  'xx-small', 'x-small', 'atto', 'femto', 'pico', 'nano', 'tiny', 'small', 'base', 'large',
  'b0', 'b1', 'b2', 'b3', 'b4', 'b5', 's0', 's1', 's2', 's3', 's4',
  'n', 's', 'm', 'l', 'x',
].sort((a, b) => b.length - a.length);

const STAGE_METRICS = new Map([
  ['pre_process_latency', '前处理'],
  ['preprocess-latency', '前处理'],
  ['post_process_latency', '后处理'],
  ['postprocess-latency', '后处理'],
  ['end-to-end-latency', '端到端'],
  ['end_to_end_latency', '端到端'],
  ['encoder-latency', '编码器'],
  ['decoder-latency', '解码器'],
  ['predictor-latency', '预测器'],
  ['cpu-cif-latency', 'CPU CIF'],
  ['cpu-inference-time', 'CPU 推理'],
  ['tracker-update-latency', '跟踪更新'],
]);

const ACCURACY_METRICS = new Set([
  'top-1', 'top-5', 'bbox-all-map-50-95', 'mask-all-map-50-95', 'keypoints-all-map-50-95',
  'obb-map-50', 'miou', 'pixel-accuracy', 'cosine-similarity', 'cosine_similarity', 'mse',
  'ptq-output-cosine', 'cer', 'keyword-confidence', 'trajectory-cosine-similarity',
  'agent-state-cosine-similarity', 'bev-cosine-similarity', 'bev-pixel-agreement', 'bev-mean-iou',
]);

function blobText(revision, path) {
  const key = `${revision}:${path}`;
  if (blobCache.has(key)) return blobCache.get(key);
  let text = '';
  try {
    text = execFileSync('git', ['show', `${revision}:${path}`], {
      cwd: repoRoot,
      encoding: 'utf8',
      maxBuffer: 64 * 1024 * 1024,
    });
  } catch {
    text = '';
  }
  blobCache.set(key, text);
  return text;
}

function entryWindow(text, id) {
  const needle = `id: ${id}`;
  let from = 0;
  while (from < text.length) {
    const start = text.indexOf(needle, from);
    if (start < 0) return '';
    const next = text.indexOf('\n  - id:', start + needle.length);
    const window = text.slice(start, next < 0 ? Math.min(text.length, start + 20000) : next);
    if (window.startsWith(needle) || window.includes(`\n${needle}`) || window.slice(0, needle.length) === needle) return window;
    from = start + needle.length;
  }
  return '';
}

function parseFlat(body) {
  const object = {};
  let current = '';
  let quote = '';
  const parts = [];
  for (const character of body) {
    if (quote) {
      current += character;
      if (character === quote) quote = '';
      continue;
    }
    if (character === '"' || character === "'") {
      quote = character;
      current += character;
      continue;
    }
    if (character === ',') {
      parts.push(current);
      current = '';
      continue;
    }
    current += character;
  }
  if (current.trim()) parts.push(current);
  for (const part of parts) {
    const match = part.match(/^\s*([A-Za-z0-9_]+)\s*:\s*(.*)$/);
    if (!match) continue;
    object[match[1]] = match[2].trim().replace(/^['"]|['"]$/g, '');
  }
  return object;
}

function metricObjects(windowText) {
  const objects = [];
  for (const match of windowText.matchAll(/\{[^{}]*\bmetric\s*:[\s\S]*?\}/g)) {
    const parsed = parseFlat(match[0].slice(1, -1));
    if (parsed.metric) objects.push(parsed);
  }
  const lines = windowText.split('\n');
  for (let index = 0; index < lines.length; index += 1) {
    const metricLine = lines[index].match(/^(\s*)-\s+metric:\s*(.+?)\s*$/);
    if (!metricLine) continue;
    const indent = metricLine[1].length;
    const object = { metric: metricLine[2].replace(/['"]/g, '').trim() };
    for (let next = index + 1; next < lines.length; next += 1) {
      const line = lines[next];
      if (!line.trim()) continue;
      const field = line.match(/^(\s*)([A-Za-z0-9_]+):\s*(.*)$/);
      if (!field || field[1].length <= indent) break;
      object[field[2]] = field[3].trim().replace(/^['"]|['"]$/g, '');
    }
    objects.push(object);
  }
  return objects;
}

function sameNumber(left, right) {
  const a = Number(left);
  const b = Number(right);
  return Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) < 1e-6;
}

function annotate(row, revision) {
  const annotated = { ...row };
  const text = blobText(revision, row.source);
  const windowText = entryWindow(text, row.id);
  const matches = metricObjects(windowText).filter(object => object.metric === row.metric
    && sameNumber(object.value, row.value)
    && String(object.unit || '') === String(row.unit || ''));
  const scoped = row.condition
    ? matches.filter(object => object.scope === row.condition || object.condition === row.condition)
    : matches;
  const chosen = scoped.length === 1 ? scoped : matches;
  const unique = field => {
    const values = [...new Set(chosen.map(object => object[field]).filter(value => value !== undefined && value !== ''))];
    return values.length === 1 ? values[0] : undefined;
  };
  const stage = unique('model_stage');
  if (stage === 'float' || stage === 'quantized' || stage === 'board') annotated.modelStage = stage;
  const concurrency = unique('concurrency');
  if (concurrency !== undefined && /^\d+$/.test(String(concurrency))) annotated.concurrency = Number(concurrency);
  const qualifier = unique('qualifier');
  if (qualifier === 'exact' || qualifier === 'lower-bound' || qualifier === 'approximate') annotated.qualifier = qualifier;
  const statistic = unique('statistic');
  if (statistic && !annotated.statistic) annotated.statistic = statistic;
  return annotated;
}

// Words that show up in benchmark prose but not in artifact filenames.
// Dataset alias "pascal" is the same release as the VOC bins. "prompt" is
// the MobileSAM box-prompt label; the bin is named "box" and "decoder".
const LABEL_NOISE = new Set([
  'on', 'rdk', 'pascal', 'prompt', 'nv12', 'bpu', 'default', 'the', 'and', 'for', 'with',
  'x5', 's100', 's100p', 's600', 'hbm', 'bin',
]);

function tokensOf(name) {
  const text = String(name || '').toLowerCase()
    .replace(/3d[_\-\s]?resnet[_\-\s]?18/g, ' r3d18 ')
    .replace(/vit[_\-]?s[_\-\/]?14/g, ' vits14 ')
    .replace(/r3d[_\-]?18/g, ' r3d18 ')
    .replace(/xx[_\-\s]?small/g, ' qxxs ')
    .replace(/x[_\-\s]?small/g, ' qxs ')
    .replace(/m(\d+)[._](\d+)/g, ' m$1p$2 ');
  const alias = { qxxs: 'xx-small', qxs: 'x-small' };
  return text.split(/[^a-z0-9]+/)
    .filter(Boolean)
    .map(token => alias[token] || token)
    .map(token => {
      const square = token.match(/^(\d+)x\1$/);
      return square ? square[1] : token;
    })
    .filter(token => !LABEL_NOISE.has(token));
}

// "efficientnet" + "b0" is the same identity as the filename token "efficientnetb0".
function glueTrailingSize(tokens) {
  const glued = [];
  for (const token of tokens) {
    const size = SIZE_TOKENS.includes(token) ? token.replace(/-/g, '') : '';
    const previous = glued[glued.length - 1];
    if (size && previous && !SIZE_TOKENS.includes(previous) && !/\d/.test(previous)) glued[glued.length - 1] = previous + size;
    else glued.push(token);
  }
  return glued;
}

function sameVariant(left, right, modelName = '') {
  const drop = new Set(modelName ? tokensOf(modelName) : []);
  const prepare = tokens => {
    const joined = [];
    for (const token of tokens) {
      const previous = joined[joined.length - 1];
      // "ResNeXt" + "50" is the filename token resnext50. Resolutions stay separate.
      if (/^\d{1,2}$/.test(token) && previous && /^[a-z]+$/.test(previous) && !SIZE_TOKENS.includes(previous)) joined[joined.length - 1] = previous + token;
      else joined.push(token);
    }
    const kept = joined.filter(token => !drop.has(token));
    return glueTrailingSize(kept);
  };
  const a = prepare(tokensOf(left));
  const b = prepare(tokensOf(right));
  if (!a.length && !b.length) return true;
  if (!a.length) return b.every(token => /^\d+$/.test(token));
  if (!b.length) return a.every(token => /^\d+$/.test(token));
  const [small, large] = a.length <= b.length ? [a, b] : [b, a];
  return small.every(token => large.includes(token));
}

// "RepVGG B1g2 224x224 Multi-thread" is the same variant measured under
// another runtime mode, not a second model.
function canonicalVariant(label) {
  return String(label || '').replace(/\s+multi[-\s]?thread\s*$/i, '').trim();
}

function sizeToken(label) {
  const tokens = String(label || '').toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
  const compact = tokens.join('');
  for (const token of SIZE_TOKENS) {
    if (tokens.includes(token)) return token;
  }
  const tail = compact.match(/(?:yolo)?(?:v?\d+)?(?:e)?(n|s|m|l|x)$/);
  return tail ? tail[1] : '';
}

function variantGroups(record) {
  const groups = [];
  for (const supported of record.supported_platforms || []) {
    const platform = String(supported.platform || '').toUpperCase();
    const rows = (record.sourced_benchmarks || []).filter(row => !row.platform || String(row.platform).toUpperCase() === platform);
    const grouped = new Map();
    for (const row of rows) {
      const label = canonicalVariant(row.variant || record.name);
      const key = tokensOf(label).join('-') || 'model';
      if (!grouped.has(key)) grouped.set(key, { key, label, rows: [] });
      grouped.get(key).rows.push(row);
    }
    for (const published of record.published_variants || []) {
      const compact = String(published.name || '').toLowerCase().replace(/[^a-z0-9]/g, '');
      if (['x5', 's100', 's100p', 's600', 'rdkx5', 'rdks100', 'rdks100p', 'rdks600'].includes(compact)) continue;
      const matches = [...grouped.values()].filter(group => sameVariant(group.label, published.name, record.name));
      if (matches.length === 1) continue;
      if (matches.length > 1) continue;
      const key = tokensOf(published.name).join('-') || 'model';
      if (!grouped.has(key)) grouped.set(key, { key, label: published.name, rows: [] });
    }
    if (!grouped.size) grouped.set(tokensOf(record.name).join('-') || 'model', { key: 'model', label: record.name, rows: [] });
    const drafts = [...grouped.values()];
    const counts = new Map();
    for (const group of drafts) {
      const token = sizeToken(group.label) || group.key;
      counts.set(token, (counts.get(token) || 0) + 1);
    }
    for (const group of drafts) {
      const token = sizeToken(group.label) || group.key;
      groups.push({
        platform,
        key: group.key,
        label: group.label,
        modelSize: counts.get(token) > 1 ? group.key : token,
        rows: group.rows,
      });
    }
  }
  return groups;
}

function distinctiveTokens(label, name) {
  const drop = new Set(tokensOf(name));
  return tokensOf(label).filter(token => !drop.has(token) && (token.length > 1 || SIZE_TOKENS.includes(token)));
}

// Size words must not leak across siblings ("small" is not "x-small").
// A size written into a longer filename token still counts ("b0" inside
// "efficientnetb0"). Digits stay exact so 224 does not match 384.
function tokenSatisfied(token, fileTokens, compact) {
  if (fileTokens.has(token)) return true;
  if (/^\d+$/.test(token)) {
    for (const other of fileTokens) {
      if (other === token || other.startsWith(`${token}x`) || other.endsWith(`x${token}`)) return true;
    }
  }
  if (token.length < 2) return false;
  if (SIZE_TOKENS.includes(token)) {
    for (const other of fileTokens) {
      if (other !== token && SIZE_TOKENS.includes(other) && (other.includes(token) || token.includes(other))) return false;
    }
    return compact.includes(token.replace(/-/g, ''));
  }
  if (/\d/.test(token)) return false;
  return compact.includes(token);
}

function namedSubset(items, label, name) {
  const tokens = distinctiveTokens(label, name);
  if (!tokens.length) return [];
  return items.filter(item => {
    // nash-m / nash-e / nash-p are BPU march directories, not model sizes.
    const cleaned = `${item.filename || ''} ${item.path || ''}`.replace(/nash-[emp]/gi, ' ').replace(/bayes-?e/gi, ' ');
    const fileTokens = new Set(tokensOf(cleaned));
    const compact = [...fileTokens].join('');
    return tokens.every(token => tokenSatisfied(token, fileTokens, compact));
  });
}

function isCommonDownload(item) {
  return String(item.filename || '').toLowerCase().startsWith('common/');
}

function conditionKey(row) {
  if (Number.isInteger(row.concurrency)) return `c:${row.concurrency}`;
  return `t:${row.condition || 'recorded'}`;
}

function conditionOptions(rows) {
  const options = [];
  const seen = new Set();
  for (const row of rows.filter(row => !ACCURACY_METRICS.has(row.metric))) {
    const key = conditionKey(row);
    if (seen.has(key)) continue;
    seen.add(key);
    options.push(Number.isInteger(row.concurrency)
      ? { key, concurrency: row.concurrency }
      : { key, label: row.condition || '' });
  }
  return options.sort((a, b) => {
    const aNumber = Number.isInteger(a.concurrency);
    const bNumber = Number.isInteger(b.concurrency);
    if (aNumber && bNumber) return a.concurrency - b.concurrency;
    if (aNumber !== bNumber) return aNumber ? -1 : 1;
    return String(a.label || '').localeCompare(String(b.label || ''));
  });
}

function measurement(row, sourceUrl) {
  return {
    metric: row.metric,
    value: Number(row.value),
    unit: row.unit,
    ...(row.qualifier ? { qualifier: row.qualifier } : {}),
    ...(row.statistic ? { statistic: row.statistic } : {}),
    ...(row.modelStage ? { model_stage: row.modelStage } : {}),
    ...(Number.isInteger(row.concurrency) ? { concurrency: row.concurrency } : {}),
    conditionKey: conditionKey(row),
    ...(row.condition ? { condition: row.condition } : {}),
    source: sourceUrl,
    ...(STAGE_METRICS.has(row.metric) ? { stageLabel: STAGE_METRICS.get(row.metric) } : {}),
  };
}

function documentationBenchmark(rows, blobUrl) {
  const measurements = rows.map(row => measurement(row, blobUrl(row.source)));
  return {
    documentation: true,
    performance: measurements.filter(row => row.metric === 'latency' || row.metric === 'throughput'),
    accuracy: measurements.filter(row => ACCURACY_METRICS.has(row.metric)),
    stages: measurements.filter(row => STAGE_METRICS.has(row.metric)),
    extras: measurements.filter(row => row.metric !== 'latency' && row.metric !== 'throughput'
      && !ACCURACY_METRICS.has(row.metric) && !STAGE_METRICS.has(row.metric)),
    conditionOptions: conditionOptions(rows),
    endToEnd: [],
  };
}

const PLATFORM_TOKENS = ['s100p', 's100', 's600', 'x5'];

function expandCommand(text, index) {
  let start = text.lastIndexOf('\n', Math.max(0, index - 1)) + 1;
  while (start > 0) {
    const previousStart = text.lastIndexOf('\n', start - 2) + 1;
    const previous = text.slice(previousStart, start - 1);
    if (!/\\\s*$/.test(previous)) break;
    start = previousStart;
  }
  const lines = [];
  let position = start;
  while (position < text.length) {
    const lineEnd = text.indexOf('\n', position);
    const end = lineEnd < 0 ? text.length : lineEnd;
    const line = text.slice(position, end);
    lines.push(line);
    position = end + 1;
    if (!/\\\s*$/.test(line) || lineEnd < 0) break;
  }
  return { command: lines.join('\n').replace(/\s+$/, ''), start, end: position };
}

// Join the cited lines back into the shell commands that contain them.
// Continuation lines stay exactly as they appear in the pinned file.
function verbatimCommands(text, lines) {
  if (!text) return [];
  const commands = [];
  const spans = [];
  let from = 0;
  for (const line of lines || []) {
    const needle = String(line || '');
    if (!needle.trim()) continue;
    let index = text.indexOf(needle, from);
    const covered = spans.find(span => index >= span.start && index < span.end);
    if (covered) {
      from = covered.end;
      index = text.indexOf(needle, from);
    }
    if (index < 0) continue;
    const span = expandCommand(text, index);
    if (span.command && !commands.includes(span.command)) {
      commands.push(span.command);
      spans.push(span);
    }
    from = Math.max(from, span.end);
  }
  return commands;
}

function commandWords(command) {
  return command.replace(/\\\s*\n\s*/g, ' ').replace(/\s+/g, ' ').trim();
}

function mentionedPlatforms(command) {
  const found = [];
  const lower = command.toLowerCase();
  for (const token of PLATFORM_TOKENS) {
    const pattern = new RegExp(`(?:^|[\\s=/:_'"\`-])${token}(?=$|[\\s=/:_.'"\\\`-])`);
    if (pattern.test(lower)) found.push(token);
  }
  return found;
}

function isAuxiliaryCommand(command) {
  return /(?:^|\s)--(?:dry-run|list-models)(?:\s|$)/.test(commandWords(command));
}

function isDownloadCommand(command) {
  if (isAuxiliaryCommand(command)) return false;
  const flat = commandWords(command);
  if (/(?:^|\s)(?:wget|curl)(?:\s|$)/.test(flat)) {
    return /\.(?:bin|hbm|onnx)(?:\s|['"]|$)/.test(flat);
  }
  return /(?:^|[\s/.])(?:download(?:\.(?:py|sh))?|download_model\.sh|model\.download)(?:\s|$)/.test(flat);
}

function isRunCommand(command) {
  if (isAuxiliaryCommand(command) || isDownloadCommand(command)) return false;
  const flat = commandWords(command);
  return /(?:^|\s)(?:\S*\/runtime\/\S*(?:main\.py|run\.sh)|\S*runtime\.python\.main|run\.sh)(?:\s|$)/.test(flat);
}

function pickCommand(commands, platform) {
  const token = String(platform || '').toLowerCase();
  const eligible = commands.filter(command => {
    const mentioned = mentionedPlatforms(command);
    return !mentioned.length || mentioned.includes(token);
  });
  return eligible.find(command => mentionedPlatforms(command).includes(token)) || eligible[0] || '';
}

function quickstartForUsage(text, lines, platform) {
  const commands = verbatimCommands(text, lines);
  return {
    download: pickCommand(commands.filter(isDownloadCommand), platform),
    run: pickCommand(commands.filter(isRunCommand), platform),
  };
}

function quickstartForRecord(record, platform) {
  const commands = [];
  for (const item of record.usage || []) {
    const text = blobText(record.sample_revision, item.source);
    commands.push(...verbatimCommands(text, item.lines));
  }
  return {
    download: pickCommand(commands.filter(isDownloadCommand), platform),
    run: pickCommand(commands.filter(isRunCommand), platform),
  };
}

function commandSizeTokens(command) {
  const found = [];
  for (const token of tokensOf(command)) {
    if (SIZE_TOKENS.includes(token) && !found.includes(token)) found.push(token);
  }
  return found;
}

function variantSizeTokens(label) {
  const tokens = tokensOf(label);
  return SIZE_TOKENS.filter(token => tokens.includes(token));
}

// A command that names atto stays on Atto. An unsized download stays on the
// variants that have a documented file, unless no sibling has one.
function scopeQuickstart(quickstart, label, variantHasDownload, downloadsAreSpecific) {
  const variantSizes = variantSizeTokens(label);
  const fit = (command, isDownload) => {
    if (!command) return '';
    const named = commandSizeTokens(command);
    if (named.length) return named.every(token => variantSizes.includes(token)) ? command : '';
    if (isDownload && downloadsAreSpecific && !variantHasDownload) return '';
    return command;
  };
  return {
    download: fit(quickstart?.download, true),
    run: fit(quickstart?.run, false),
  };
}

function buildPendingModels(records, { officialRepositoryUrl, taskRegistry, safeId }) {
  const models = [];
  for (const record of records) {
    if (record.registration !== 'pending') continue;
    const taskInfo = taskRegistry.tasks[record.task];
    if (!taskInfo) throw new Error(`Pending model ${record.id} uses unknown task ${record.task}`);
    const facetTasks = Array.isArray(record.facet_tasks) && record.facet_tasks.length
      ? record.facet_tasks
      : [taskInfo.facet_id];
    const revision = record.sample_revision;
    const sourceUrl = `${officialRepositoryUrl}/tree/${revision}/${record.sample_path}`;
    const blobUrl = path => `${officialRepositoryUrl}/blob/${revision}/${path}`;
    const heroFile = String(record.hero_visual?.file || '');
    const heroName = heroFile ? heroFile.split('/').pop() : '';
    const heroCover = heroName ? `assets/models/pending-heroes/${heroName}` : '';
    const platformDownloads = new Map();
    const platformConfigs = new Map();
    const groups = variantGroups(record).map(group => {
      const downloads = (record.documented_downloads || []).filter(item => !item.platform || String(item.platform).toUpperCase() === group.platform);
      const configs = record.conversion_configs || [];
      if (!platformDownloads.has(group.platform)) platformDownloads.set(group.platform, downloads);
      if (!platformConfigs.has(group.platform)) platformConfigs.set(group.platform, configs);
      return group;
    });
    const downloadIsSpecific = new Map();
    const downloadMatches = new Map();
    const configIsSpecific = new Map();
    for (const [platform, downloads] of platformDownloads) {
      const platformGroups = groups.filter(group => group.platform === platform);
      const matched = new Map(platformGroups.map(group => [group, namedSubset(downloads, group.label, record.name)]));
      const covered = new Set([...matched.values()].flat().map(item => item.url));
      const complete = [...matched.values()].some(list => list.length)
        && downloads.every(item => covered.has(item.url) || isCommonDownload(item));
      downloadIsSpecific.set(platform, complete);
      downloadMatches.set(platform, matched);
    }
    for (const [platform, configs] of platformConfigs) {
      configIsSpecific.set(platform, groups.some(group => group.platform === platform && namedSubset(configs, group.label, record.name).length));
    }
    for (const group of groups) {
      const annotated = group.rows.map(row => annotate(row, revision));
      const downloads = platformDownloads.get(group.platform) || [];
      const configs = platformConfigs.get(group.platform) || [];
      const matchedDownloads = downloadMatches.get(group.platform)?.get(group) || [];
      const commonDownloads = downloads.filter(isCommonDownload);
      const selectedDownloads = downloadIsSpecific.get(group.platform)
        ? [...matchedDownloads, ...commonDownloads.filter(item => !matchedDownloads.includes(item))]
        : downloads;
      const selectedConfigs = configIsSpecific.get(group.platform) ? namedSubset(configs, group.label, record.name) : configs;
      const rowIds = new Set(annotated.map(row => row.id));
      let inputs = (record.inputs || []).filter(item => !item.platform || String(item.platform).toUpperCase() === group.platform);
      const matchedInputs = inputs.filter(item => rowIds.has(item.source_id));
      if (matchedInputs.length) inputs = matchedInputs;
      models.push({
        id: safeId(record.source, record.family, record.task, group.platform, group.modelSize),
        catalogId: record.id,
        name: record.name,
        variantName: group.label,
        family: record.family,
        modelSize: group.modelSize,
        releaseStatus: 'data-pending',
        dataPending: true,
        releasePlatform: group.platform,
        hardware: `RDK ${group.platform}`,
        task: taskInfo.label_zh,
        taskId: taskInfo.facet_id,
        tasks: facetTasks,
        description: record.description?.zh || '',
        descriptionEn: record.description?.en || '',
        platforms: [group.platform],
        coverImage: heroCover || 'assets/pending-cover.svg',
        ...(heroCover ? {
          heroProvenance: {
            kind: record.hero_visual.kind,
            input: record.hero_visual.input,
            predictionsSource: record.hero_visual.predictions_source || record.hero_visual.input,
            revision,
            runtime: record.hero_visual.runtime,
            note: record.hero_visual.note || '',
            modelUrl: record.hero_visual.model_url || '',
            framework: record.hero_visual.framework || '',
            ...(record.hero_visual.confidence_threshold === undefined ? {} : {
              confidenceThreshold: record.hero_visual.confidence_threshold,
            }),
          },
        } : {}),
        sourceInput: record.source_input ? {
          format: record.source_input.format || '',
          dtype: record.source_input.dtype || '',
          layout: record.source_input.layout || '',
          shape: record.source_input.shape || '',
        } : null,
        coverLabel: heroCover
          ? (record.hero_visual.runtime === 'host-float-reference' ? '浮点参考' : '参考结果')
          : '封面待补充',
        sample: `${record.id}/${group.modelSize}`,
        source: sourceUrl,
        sourceType: 'repository',
        licenseName: record.license?.name || '',
        licenseUrl: record.license?.url || '',
        licenseScope: record.license?.scope || '',
        upstream: (record.upstream || []).map(item => ({ name: item.name, url: item.url, kind: item.kind })),
        inputs: inputs.map(item => ({
          shape: item.shape || '',
          layout: item.layout || '',
          format: item.format || '',
          source: blobUrl(item.source),
        })),
        conversions: selectedConfigs.map(item => ({
          path: item.path,
          url: blobUrl(item.path),
          march: item.march || '',
          runtimeInput: item.runtime_input || '',
          trainInput: item.train_input || '',
          trainLayout: item.train_layout || '',
          resolution: item.resolution || '',
        })),
        usage: (record.usage || []).flatMap(item => item.lines || []),
        usageSource: record.usage?.[0]?.source ? blobUrl(record.usage[0].source) : '',
        quickstart: scopeQuickstart(
          quickstartForRecord(record, group.platform),
          group.label,
          selectedDownloads.length > 0,
          Boolean(downloadIsSpecific.get(group.platform)),
        ),
        documentedDownloads: selectedDownloads.map(item => ({
          filename: item.filename,
          url: item.url,
          format: item.format || '',
          sha256: item.sha256 || '',
        })),
        assets: [],
        benchmark: documentationBenchmark(annotated, blobUrl),
      });
    }
  }
  return models;
}

export { buildPendingModels, variantGroups, annotate, quickstartForUsage, scopeQuickstart, tokensOf, namedSubset, sameVariant };
