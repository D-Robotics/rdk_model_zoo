import { cp, mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { basename, dirname, extname, isAbsolute, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptRoot = dirname(fileURLToPath(import.meta.url));
const webRoot = resolve(scriptRoot, '..');
const outputRoot = resolve(webRoot, 'dist');
const catalogPath = process.env.MODEL_ZOO_CATALOG;
const inputsPath = process.env.MODEL_ZOO_INPUTS;
const repositoryUrl = process.env.MODEL_ZOO_REPOSITORY_URL || 'https://github.com/D-Robotics/rdk_model_zoo';

if (!catalogPath) throw new Error('MODEL_ZOO_CATALOG must point to the samples-only catalog.json');
if (!inputsPath) throw new Error('MODEL_ZOO_INPUTS must point to the reviewed multi-model input manifest');

await import('./generate.mjs');

const catalog = JSON.parse(await readFile(resolve(catalogPath), 'utf8'));
if (catalog.source !== 'model_zoo_web/data') {
  throw new Error(`Unexpected catalog source: ${catalog.source || 'missing'}`);
}

const resolvedInputsPath = resolve(inputsPath);
const inputRoot = dirname(resolvedInputsPath);
const inputs = JSON.parse(await readFile(resolvedInputsPath, 'utf8'));
if (inputs.schema_version !== 1) {
  throw new Error(`Unsupported MODEL_ZOO_INPUTS schema_version: ${inputs.schema_version || 'missing'}`);
}
if (!inputs.models || typeof inputs.models !== 'object' || Array.isArray(inputs.models)) {
  throw new Error('MODEL_ZOO_INPUTS.models must be an object keyed by catalog model id');
}
if (!inputs.releases || typeof inputs.releases !== 'object' || Array.isArray(inputs.releases)) {
  throw new Error('MODEL_ZOO_INPUTS.releases must be an object keyed by release id');
}

const resolveInputFile = async (value, label) => {
  if (typeof value !== 'string' || !value.trim()) throw new Error(`${label} must be a non-empty path`);
  const path = isAbsolute(value) ? value : resolve(inputRoot, value);
  const details = await stat(path).catch(() => null);
  if (!details?.isFile()) throw new Error(`${label} does not exist or is not a file: ${path}`);
  return path;
};
const oeDataCache = new Map();
const loadOeData = async (value, label) => {
  const path = await resolveInputFile(value, label);
  if (oeDataCache.has(path)) return oeDataCache.get(path);
  const oeData = JSON.parse(await readFile(path, 'utf8'));
  if (oeData.schema_version !== 1) {
    throw new Error(`${label} has unsupported schema_version: ${oeData.schema_version}`);
  }
  if (!oeData.provenance?.artifact_sha256) {
    throw new Error(`${label} is missing provenance.artifact_sha256`);
  }
  oeDataCache.set(path, oeData);
  return oeData;
};
const usedModelInputs = new Set();
const usedReleaseInputs = new Set();

const taskMetadata = {
  detect: { id: 'object-detection', label: '目标检测', description: 'COCO 目标检测模型' },
  classify: { id: 'image-classification', label: '图像分类', description: '图像分类模型' },
  cls: { id: 'image-classification', label: '图像分类', description: '图像分类模型' },
  segment: { id: 'instance-segmentation', label: '实例分割', description: '实例分割模型' },
  seg: { id: 'instance-segmentation', label: '实例分割', description: '实例分割模型' },
  pose: { id: 'pose-estimation', label: '姿态估计', description: '人体姿态估计模型' },
  obb: { id: 'object-detection', label: '旋转框检测', description: '旋转框目标检测模型' },
};

const safeId = (...parts) => parts.join('-').toLowerCase().replace(/[^a-z0-9-]+/g, '-').replace(/-+/g, '-');
const displayNames = (family, size, task) => {
  const familyName = family.replace(/^yolo/i, 'YOLO');
  const taskName = task === 'detect' ? 'Detect' : task[0].toUpperCase() + task.slice(1);
  return {
    model: `${familyName} ${taskName}`,
    variant: `${familyName}${size} ${taskName}`,
  };
};
const metric = (record, name) => Number(record?.[name]);

const models = [];
const properties = {};
const reportEntries = [];
const assetsRoot = resolve(outputRoot, 'assets', 'models');
const reportsRoot = resolve(outputRoot, 'reports', 'models');
await mkdir(assetsRoot, { recursive: true });
await mkdir(reportsRoot, { recursive: true });

for (const record of catalog.models || []) {
  const modelInput = inputs.models[record.id];
  if (!modelInput || typeof modelInput !== 'object' || Array.isArray(modelInput)) {
    throw new Error(`MODEL_ZOO_INPUTS.models is missing ${record.id}`);
  }
  usedModelInputs.add(record.id);
  const coverPath = await resolveInputFile(modelInput.cover, `MODEL_ZOO_INPUTS.models.${record.id}.cover`);
  const coverExtension = extname(coverPath).toLowerCase();
  if (!['.jpg', '.jpeg', '.png', '.webp'].includes(coverExtension)) {
    throw new Error(`Unsupported cover image extension for ${record.id}: ${coverExtension || 'missing'}`);
  }
  const normalizedCoverExtension = coverExtension === '.jpeg' ? '.jpg' : coverExtension;
  const coverFilename = `${safeId(record.source, record.family, record.task)}${normalizedCoverExtension}`;
  await cp(coverPath, resolve(assetsRoot, coverFilename));

  const task = taskMetadata[record.task] || {
    id: record.task,
    label: record.task,
    description: `${record.family} ${record.task} model`,
  };
  for (const variant of record.variants || []) {
    for (const release of variant.platforms || []) {
      if (release.status !== 'released') continue;
      const platform = String(release.platform).toUpperCase();
      const id = safeId(record.source, record.family, record.task, variant.size, release.platform);
      const releaseId = `${record.id}/${variant.size}/${release.platform}`;
      const names = displayNames(record.family, variant.size, record.task);

      const artifact = release.artifact;
      const sourceUrl = `${repositoryUrl}/tree/${release.provenance.repository_commit}/${record.sample_path}`;
      const floatAccuracy = release.accuracy?.float_onnx;
      const runtimeAccuracy = release.accuracy?.runtime;
      const performance = release.performance || {};
      const measurements = performance.measurements || [];
      const endToEndSource = performance.end_to_end;
      const endToEndRecords = Array.isArray(endToEndSource)
        ? endToEndSource
        : endToEndSource ? [endToEndSource] : [];
      const endToEndMetric = (endToEnd, name) => {
        const summary = endToEnd?.metrics_ms?.[name];
        if (!summary || !Number.isFinite(Number(summary.mean))) return null;
        return {
          value: Number(summary.mean),
          unit: 'ms',
          p50: Number(summary.p50),
          p95: Number(summary.p95),
          min: Number(summary.min),
          max: Number(summary.max),
        };
      };
      const reportSourceUrl = release.reports?.oe_conversion_url;
      const releaseInput = inputs.releases[releaseId];
      let oeData;
      if (releaseInput !== undefined) {
        if (!releaseInput || typeof releaseInput !== 'object' || Array.isArray(releaseInput)) {
          throw new Error(`MODEL_ZOO_INPUTS.releases.${releaseId} must be an object`);
        }
        usedReleaseInputs.add(releaseId);
        oeData = await loadOeData(
          releaseInput.oe_data,
          `MODEL_ZOO_INPUTS.releases.${releaseId}.oe_data`,
        );
        if (oeData.provenance.artifact_sha256 !== artifact.sha256) {
          throw new Error(
            `${releaseId}: OE data artifact sha256 does not match the catalog artifact`,
          );
        }
      }
      let reportUrl;
      let reportDataUrl;
      if (oeData) {
        // Structured path: ship the small extracted JSON. Full HTML payloads
        // stay out of dist and are only retained as a report fallback source.
        const dataRoot = resolve(outputRoot, 'reports', 'data');
        await mkdir(dataRoot, { recursive: true });
        const dataFilename = `${id}-oe-data.json`;
        const dataBytes = JSON.stringify(oeData, null, 2);
        await writeFile(resolve(dataRoot, dataFilename), dataBytes, 'utf8');
        reportDataUrl = `reports/data/${dataFilename}`;
      } else if (reportSourceUrl) {
        const response = await fetch(reportSourceUrl);
        if (!response.ok) throw new Error(`Unable to fetch OE report: ${response.status} ${reportSourceUrl}`);
        const reportBytes = Buffer.from(await response.arrayBuffer());
        if (!/<(?:!doctype\s+html|html)[\s>]/i.test(reportBytes.subarray(0, 512).toString('utf8'))) {
          throw new Error(`OE report is not an HTML document: ${reportSourceUrl}`);
        }
        const reportFilename = `${id}-oe-report.html`;
        await writeFile(resolve(reportsRoot, reportFilename), reportBytes);
        reportUrl = `reports/models/${reportFilename}`;
      }
      if (reportDataUrl || reportUrl) {
        reportEntries.push({
          id,
          name: names.variant,
          hardware: platform,
          march: artifact.march,
          kind: 'conversion',
          bytes: reportDataUrl
            ? Buffer.byteLength(JSON.stringify(oeData), 'utf8')
            : 0,
          path: reportDataUrl ? reportSourceUrl : reportUrl,
          dataUrl: reportDataUrl,
          modelIds: [id],
          sources: [oeData?.generated_from_run].filter(Boolean),
        });
      }
      const accuracyRecords = [];
      if (Number.isFinite(metric(floatAccuracy, 'map_50_95'))) {
        accuracyRecords.push({
          metric: 'bbox-all-map-50-95',
          value: metric(floatAccuracy, 'map_50_95'),
          unit: 'ratio',
          model_stage: 'float',
        });
      }
      if (Number.isFinite(metric(runtimeAccuracy, 'map_50_95'))) {
        accuracyRecords.push({
          metric: 'bbox-all-map-50-95',
          value: metric(runtimeAccuracy, 'map_50_95'),
          unit: 'ratio',
          model_stage: 'quantized',
        });
      }

      models.push({
        id,
        catalogId: record.id,
        name: names.model,
        variantName: names.variant,
        family: record.family,
        modelSize: variant.size,
        releasePlatform: platform,
        task: task.label,
        taskId: task.id,
        tasks: [task.id],
        description: record.description?.zh || task.description,
        descriptionEn: record.description?.en || `${record.family} ${record.task} model.`,
        platforms: [platform],
        coverImage: `assets/models/${coverFilename}`,
        coverLabel: '模型参考推理结果',
        sample: `${record.source}/${record.family}/${record.task}/${variant.size}`,
        source: sourceUrl,
        licenseName: record.license?.name,
        licenseUrl: record.license?.url,
        shape: (variant.input.source?.shape || [1, 3, variant.input.height, variant.input.width]).join(' × '),
        reportUrl,
        reportDataUrl,
        reportSourceUrl,
        assets: [
          {
            role: `${platform} 部署模型`,
            format: artifact.format,
            filename: basename(new URL(artifact.url).pathname),
            url: artifact.url,
            sha256: artifact.sha256,
            sizeBytes: Number(artifact.size_bytes),
          },
        ],
        benchmark: {
          precision: 'int8',
          environment: { hardware: `RDK ${platform}` },
          timing: {
            scope: performance.timing_scope,
            tool: performance.tool,
            implementation: performance.implementation,
            threadSemantics: performance.thread_semantics,
            stages: performance.stages,
            runsPerCondition: Number(performance.runs_per_condition),
            framesPerRun: Number(performance.frames_per_run),
            warmupFramesPerCondition: Number(performance.warmup_frames_per_condition),
          },
          endToEnd: endToEndRecords.map(endToEnd => ({
            timing: {
              scope: endToEnd.timing_scope,
              tool: endToEnd.tool,
              implementation: endToEnd.implementation,
              pipelineStreams: Number(endToEnd.pipeline_streams),
              runtimeSubmissionThreads: Number(endToEnd.runtime_submission_threads),
              cpuThreadPolicy: endToEnd.cpu_thread_policy,
              onlineCpuThreads: Number(endToEnd.online_cpu_threads),
              opencvThreads: Number(endToEnd.opencv_threads),
              cpuGovernor: endToEnd.cpu_governor,
              cpuFrequencyMhz: Number(endToEnd.cpu_frequency_mhz),
              bpuFrequencyMhz: Number(endToEnd.bpu_frequency_mhz),
              warmupFramesPerRound: Number(endToEnd.warmup_frames_per_round),
              rounds: Number(endToEnd.rounds),
              framesPerRound: Number(endToEnd.frames_per_round),
              timedFrames: Number(endToEnd.timed_frames),
              aggregateWallMs: Number(endToEnd.aggregate_wall_ms),
            },
            metrics: {
              preprocess: endToEndMetric(endToEnd, 'preprocess'),
              runtime: endToEndMetric(endToEnd, 'runtime'),
              postprocess: endToEndMetric(endToEnd, 'postprocess'),
              endToEnd: endToEndMetric(endToEnd, 'end_to_end'),
            },
            throughputFps: Number(endToEnd.throughput_fps),
          })),
          performance: measurements.flatMap(measurement => [
            {
              metric: 'latency',
              value: Number(measurement.average_latency_ms),
              unit: 'ms',
              concurrency: Number(measurement.threads),
              observedMin: Number(measurement.observed_min_latency_ms),
              observedMax: Number(measurement.observed_max_latency_ms),
            },
            {
              metric: 'throughput',
              value: Number(measurement.aggregate_fps),
              unit: 'fps',
              concurrency: Number(measurement.threads),
            },
          ]),
          accuracy: accuracyRecords,
        },
      });
      properties[id] = {
        parameterCount: Number(variant.model?.parameter_count),
        gflops: Number(variant.model?.gflops),
        inputShape: variant.input.source?.shape || [1, 3, variant.input.height, variant.input.width],
        sourceInput: variant.input.source,
        runtimeInput: {
          format: artifact.runtime_input,
          resolution: [variant.input.height, variant.input.width],
        },
      };
    }
  }
}

if (!models.length) throw new Error('The samples-only catalog contains no released models');
for (const modelId of Object.keys(inputs.models)) {
  if (!usedModelInputs.has(modelId)) throw new Error(`MODEL_ZOO_INPUTS references unknown model ${modelId}`);
}
for (const releaseId of Object.keys(inputs.releases)) {
  if (!usedReleaseInputs.has(releaseId)) throw new Error(`MODEL_ZOO_INPUTS references unknown release ${releaseId}`);
}

const platformNames = [...new Set(models.flatMap(model => model.platforms))];
const data = {
  schemaVersion: 1,
  catalog: {
    status: 'published',
    summary: {
      sample_count: new Set(models.map(model => model.sample)).size,
      asset_count: models.reduce((count, model) => count + model.assets.length, 0),
      downloadable_asset_count: models.reduce((count, model) => count + model.assets.length, 0),
      benchmark_count: models.filter(model => model.benchmark).length,
    },
  },
  repository: { url: repositoryUrl },
  release: {
    status: 'published',
    compatibility: { hardware: platformNames.map(name => `RDK ${name}`).join(' / ') },
  },
  models,
};

await writeFile(resolve(outputRoot, 'data.js'), `window.MODEL_DATA = ${JSON.stringify(data, null, 2)};\n`, 'utf8');
await writeFile(
  resolve(outputRoot, 'model-properties.js'),
  `window.MODEL_PROPERTIES = Object.freeze(${JSON.stringify(properties, null, 2)});\n`,
  'utf8',
);
await writeFile(
  resolve(outputRoot, 'reports', 'reports-data.js'),
  `window.OE_REPORTS = ${JSON.stringify(reportEntries, null, 2)};\n`,
  'utf8',
);
await writeFile(
  resolve(outputRoot, 'reports', 'inventory.json'),
  `${JSON.stringify({ reports: reportEntries.map(entry => ({ id: entry.id, kind: entry.kind })) }, null, 2)}\n`,
  'utf8',
);

console.log(`Built local Model Zoo preview with ${models.length} released model entry.`);
