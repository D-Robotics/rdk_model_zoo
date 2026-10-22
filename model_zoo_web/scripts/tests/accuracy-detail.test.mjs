import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import vm from 'node:vm';

import {
  boardAccuracyRecords,
  accuracyRecords,
  accuracyMetricLabels,
  accuracyMetricsSchema,
} from '../accuracy-metrics.mjs';

const detailSource = readFileSync(new URL('../../src/detail-view.js', import.meta.url), 'utf8');

function renderAccuracy(records, metricLabels, taskId = 'pose-estimation', catalogId = 'fixture/pose') {
  const performanceRoot = { innerHTML: '' };
  const root = {
    addEventListener() {},
    querySelectorAll: () => [],
    querySelector: selector => {
      if (selector === '#benchmark-condition') return { dataset: { value: '1' }, addEventListener() {} };
      if (selector === '#detail-performance') return performanceRoot;
      return null;
    },
  };
  const document = {
    addEventListener() {},
    documentElement: { clientWidth: 1200, classList: { add() {}, remove() {} } },
  };
  const context = {
    window: { HubI18n: { locale: 'en', apply() {} } },
    document,
    AbortController,
    console,
  };
  vm.runInNewContext(detailSource, context);
  context.window.ModelDetail.bind(root, {
    id: 'fixture',
    sample: 'fixture',
    task: 'Pose estimation',
    taskId,
    catalogId,
    benchmark: { accuracy: records },
  }, [], { accuracyMetricLabels: metricLabels, release: { compatibility: { hardware: 'RDK S100' } } });
  return performanceRoot.innerHTML;
}

function renderDetail(model) {
  const context = {
    window: { HubI18n: { locale: 'zh', apply() {} }, MODEL_PROPERTIES: {} },
    document: {},
    AbortController,
    console,
  };
  vm.runInNewContext(detailSource, context);
  return context.window.ModelDetail.render({
    model,
    models: [model],
    data: { release: { compatibility: { hardware: 'RDK S600' } } },
  });
}

test('sample-code licenses stay separate from model licenses', () => {
  const model = {
    id: 'license-fixture', catalogId: 'fixture/fixture/cls', name: 'Fixture',
    task: '图像分类', dataPending: true, hardware: 'RDK X5', assets: [],
    licenseName: 'Apache-2.0', licenseUrl: 'https://www.apache.org/licenses/LICENSE-2.0',
    licenseScope: 'sample-code',
  };
  const html = renderDetail(model);
  const modelSection = html.match(/<section><h2>模型许可<\/h2>([\s\S]*?)<\/section>/)?.[1];
  assert.match(modelSection, /未声明/);
  assert.doesNotMatch(modelSection, /Apache-2.0/);
  assert.match(html, /<h2>示例代码许可<\/h2>[\s\S]*?Apache-2.0/);

  for (const scope of [{ licenseScope: 'model' }, { dataPending: false, licenseScope: undefined }]) {
    const released = renderDetail({ ...model, ...scope });
    assert.match(released, /<h2>模型许可<\/h2>[\s\S]*?Apache-2.0/);
    assert.doesNotMatch(released, /示例代码许可/);
  }
});

test('detail accuracy tiles use task labels and show float/runtime comparison values', () => {
  const records = accuracyRecords('seg',
    { box_ap: 0.7, mask_ap: 0.6 },
    { box_ap: 0.68, mask_ap: 0.57 },
  );
  const html = renderAccuracy(records, accuracyMetricLabels());

  assert.match(html, /data-i18n-en="Box AP50–95"/);
  assert.match(html, /data-i18n-en="Mask AP50–95"/);
  assert.match(html, /68<small>%<\/small>/);
  assert.match(html, /57<small>%<\/small>/);
  assert.doesNotMatch(html, /mAP50-95/);
});

test('detail labels cover classification, segmentation, pose, and oriented boxes', () => {
  const labels = accuracyMetricLabels();
  for (const task of ['cls', 'seg', 'pose', 'obb']) {
    const profile = accuracyMetricsSchema.tasks[task];
    const source = Object.fromEntries(profile.required.map(field => [field, 0.75]));
    const html = renderAccuracy(accuracyRecords(task, source, source, {
      dataset: profile.dataset,
      evaluation_scope: profile.evaluation_scope?.value,
    }), labels);
    for (const { metric } of profile.display) {
      assert.ok(html.includes(`data-i18n-en="${labels[metric].en}"`), `${task}: ${metric}`);
    }
    if (profile.evaluation_scope) {
      assert.ok(!html.includes(profile.evaluation_scope.labels.en));
      assert.ok(!html.includes(profile.evaluation_scope.labels.zh));
    }
  }
});

test('board receipt metrics render without float comparison or a local DOTA scope caption', () => {
  const mappings = {
    cls: { receipt: { top1: 0.71, top5: 0.9 }, web: ['top1', 'top5'] },
    seg: { receipt: { 'bbox/AP': 0.4, 'segm/AP': 0.35 }, web: ['box_ap', 'mask_ap'] },
    pose: { receipt: { 'keypoints/AP': 0.52 }, web: ['keypoints_ap'] },
    obb: { receipt: { mAP50: 0.51 }, web: ['map_50'] },
  };
  const fieldMaps = {
    cls: { top1: 'top1', top5: 'top5' },
    seg: { 'bbox/AP': 'box_ap', 'segm/AP': 'mask_ap' },
    pose: { 'keypoints/AP': 'keypoints_ap' },
    obb: { mAP50: 'map_50' },
  };

  for (const [task, fixture] of Object.entries(mappings)) {
    const sourceValues = Object.fromEntries(Object.entries(fixture.receipt).map(([key, value]) => [fieldMaps[task][key], value]));
    const metadata = {
      dataset: accuracyMetricsSchema.tasks[task].dataset,
      evaluation_scope: accuracyMetricsSchema.tasks[task].evaluation_scope?.value,
    };
    const records = boardAccuracyRecords(task, sourceValues, metadata);
    assert.deepEqual(records.map(record => record.model_stage), fixture.web.map(() => 'board'));
    assert.deepEqual(records.map(record => record.metric), accuracyMetricsSchema.tasks[task].display.map(row => row.metric));
    const html = renderAccuracy(records, accuracyMetricLabels());
    assert.doesNotMatch(html, /浮点参考|差距|quantized/);
    assert.match(html, /Standalone board evaluation/);
    if (task === 'obb') assert.doesNotMatch(html, /本地 DOTA val|Local DOTA val/);
  }
});

test('all five YOLO26 tasks use the same end-to-end percent delta without explanatory text', () => {
  const pairs = [
    { task: 'detect', taskId: 'object-detection', catalogId: 'ultralytics_yolo/yolo26/detect', float: { map_50_95: 0.42 }, board: { map_50_95: 0.406 } },
    { task: 'cls', taskId: 'image-classification', catalogId: 'ultralytics_yolo/yolo26/cls', float: { top1: 0.4989, top5: 0.7444 }, board: { top1: 0.4712, top5: 0.7192 } },
    { task: 'seg', taskId: 'instance-segmentation', catalogId: 'ultralytics_yolo/yolo26/seg', float: { box_ap: 0.3986, mask_ap: 0.3397 }, board: { box_ap: 0.3724, mask_ap: 0.3225 } },
    { task: 'pose', taskId: 'pose-estimation', catalogId: 'ultralytics_yolo/yolo26/pose', float: { keypoints_ap: 0.7 }, board: { keypoints_ap: 0.65 } },
    { task: 'obb', taskId: 'object-detection', catalogId: 'ultralytics_yolo/yolo26/obb', float: { map_50: 0.62 }, board: { map_50: 0.57 }, metadata: { dataset: 'DOTA val', evaluation_scope: 'local_dota_val_single_scale' } },
  ];

  for (const pair of pairs) {
    const records = accuracyRecords(pair.task, pair.float, pair.board, pair.metadata);
    const html = renderAccuracy(records, accuracyMetricLabels(), pair.taskId, pair.catalogId);
    assert.match(html, /aria-haspopup="dialog"/);
    assert.match(html, /端到端差值/);
    assert.match(html, /−\d+(?:\.\d+)?%/);
    if (pair.task === 'detect') assert.match(html, /−1\.4%/);
    assert.doesNotMatch(html, /个百分点|RGB|NV12|纯量化|preprocessing|差距/);
    assert.doesNotMatch(html, /Standalone board evaluation|板端独立评测/);
    if (pair.task === 'obb') assert.doesNotMatch(html, /本地 DOTA val|Local DOTA val/);
  }
});

test('candidate detail keeps the model repository link and download without an upstream-weight link', () => {
  const model = {
    id: 'yolo26-cls-n-s600',
    catalogId: 'ultralytics_yolo/yolo26/cls',
    name: 'YOLO26 Cls',
    variantName: 'YOLO26n Cls',
    task: '图像分类',
    taskId: 'image-classification',
    description: '图像分类',
    source: 'https://github.com/D-Robotics/rdk_model_zoo/tree/main/samples/vision/ultralytics_yolo',
    upstreamWeightUrl: 'https://huggingface.co/Ultralytics/YOLO26/blob/main/yolo26n-cls.pt',
    releaseStatus: 'candidate',
    coverImage: 'assets/cls.webp',
    coverLabel: '分类示意图',
    shape: '1 × 3 × 224 × 224',
    assets: [{ role: '部署模型', format: 'hbm', filename: 'yolo26n-cls-s600.hbm', url: 'https://example.invalid/yolo26n-cls-s600.hbm', sizeBytes: 1234 }],
    benchmark: { performance: [], accuracy: [] },
  };
  const html = renderDetail(model);

  assert.match(html, /data-open-download/);
  // Files are offered through the get-model picker rather than a separate list.
  assert.match(html, /<dialog class="mz-download-dialog mz-store"/);
  assert.ok(html.includes(`href="${model.source}" target="_blank" rel="noopener">模型仓库`));
  assert.doesNotMatch(html, /上游权重|huggingface\.co/);
  assert.ok(html.includes(model.assets[0].filename));
  assert.doesNotMatch(html, /href="https:\/\/github\.com[^\"]*">上游权重/);

  const detectHtml = renderDetail({
    ...model,
    id: 'yolo11-detect-n-s600',
    catalogId: 'ultralytics_yolo/yolo11/detect',
    name: 'YOLO11 Detect',
    task: '目标检测',
    taskId: 'object-detection',
    releaseStatus: 'released',
    upstreamWeightUrl: undefined,
  });
  assert.ok(detectHtml.includes(`href="${model.source}" target="_blank" rel="noopener">模型仓库`));
  assert.ok(detectHtml.includes(`${model.source}/runtime/python`));
});

test('technical Pose and OBB models link to the shared sample without exposing checkpoint provenance', () => {
  const sampleCommit = 'eed26ce610d7fba03a68d1c0ee6e62603cd9b85d';
  for (const task of ['pose', 'obb']) {
    const source = `https://github.com/D-Robotics/rdk_model_zoo/tree/${sampleCommit}/samples/vision/ultralytics_yolo`;
    const html = renderDetail({
      id: `yolo26-${task}-n-s600`,
      catalogId: `ultralytics_yolo/yolo26/${task}`,
      name: `YOLO26 ${task.toUpperCase()}`,
      variantName: `YOLO26n ${task.toUpperCase()}`,
      task: task === 'pose' ? '姿态估计' : '旋转框检测',
      taskId: task === 'pose' ? 'pose-estimation' : 'object-detection',
      description: task,
      source,
      sourceRepositoryCommit: sampleCommit,
      conversionRepositoryCommit: 'becb068806fb5a1c3291be544ea5e54194777847',
      checkpointProvenance: {
        sourceWeightUrl: `https://huggingface.co/Ultralytics/YOLO26/blob/main/yolo26n-${task}.pt`,
        sourceCheckpointSha256: '0123456789abcdef',
      },
      releaseStatus: 'candidate',
      coverImage: 'assets/task.webp',
      coverLabel: '任务示意图',
      shape: '1 × 3 × 640 × 640',
      assets: [],
      benchmark: { performance: [], accuracy: [] },
    });

    assert.ok(html.includes(`href="${source}" target="_blank" rel="noopener">模型仓库`));
    assert.ok(html.includes(`href="${source}/runtime/python" target="_blank" rel="noopener">运行文档`));
    assert.ok(html.includes(`href="${source}/conversion" target="_blank" rel="noopener">模型转换`));
    assert.doesNotMatch(html, /上游权重/);
    assert.doesNotMatch(html, /huggingface\.co/);
  }
});

test('pending detail uses the published task tiles, parameter rows, and sidebar links', () => {
  const source = 'https://github.com/D-Robotics/rdk_model_zoo/tree/abc/samples/vision/convnext';
  const model = {
    id: 'convnext-cls-nano-x5',
    catalogId: 'convnext/convnext/cls',
    name: 'ConvNeXt',
    variantName: 'ConvNeXt Nano',
    task: '图像分类',
    taskId: 'image-classification',
    description: 'ConvNeXt 面向 ImageNet-1k 分类。',
    source,
    sourceType: 'repository',
    releaseStatus: 'data-pending',
    dataPending: true,
    hardware: 'RDK X5',
    licenseName: 'Apache-2.0',
    licenseUrl: 'https://www.apache.org/licenses/LICENSE-2.0',
    usageSource: `${source.replace('/tree/', '/blob/')}/README.md`,
    upstream: [{ kind: 'paper', name: 'arxiv', url: 'https://arxiv.org/abs/2201.03545' }],
    inputs: [{ format: 'NV12', shape: '[ 224, 224 ]' }],
    sourceInput: { format: 'rgb', dtype: 'float32', layout: 'NCHW', shape: '1x3x224x224' },
    conversions: [{
      path: 'samples/vision/convnext/conversion/ConvNeXt_nano.yaml',
      url: 'https://github.com/D-Robotics/rdk_model_zoo/blob/abc/samples/vision/convnext/conversion/ConvNeXt_nano.yaml',
      runtimeInput: 'nv12',
      trainInput: 'rgb',
      trainLayout: 'NCHW',
      resolution: '224x224',
    }],
    benchmark: {
      conditionOptions: [{ key: 'single', concurrency: 1, label: '单线程' }],
      performance: [{ metric: 'latency', value: 5.71, unit: 'ms', conditionKey: 'single', condition: 'single-thread', source: 'https://example.invalid/bench' }],
      accuracy: [{ metric: 'top-1', value: 71.75, unit: 'percent', model_stage: 'quantized', conditionKey: 'single', source: 'https://example.invalid/acc' }],
      stages: [],
      extras: [{ metric: 'bpu_usage', value: 80, unit: 'percent', conditionKey: 'single' }],
    },
  };
  const html = renderDetail(model);
  for (const label of ['参数量', '浮点计算量', '源模型输入', 'Runtime 输入', '量化精度']) {
    assert.match(html, new RegExp(`<dt>${label}</dt>`));
  }
  assert.match(html, /RGB · FLOAT32 · NCHW · 1 × 3 × 224 × 224/);
  assert.match(html, /NV12 · 224 × 224/);
  assert.doesNotMatch(html, /记录的输入|转换输入|参数待补充|<h2>上游<\/h2>|<h2>转换配置<\/h2>|<h2>运行命令<\/h2>|<h2>文档中的下载<\/h2>/);
  assert.match(html, /模型仓库/);
  assert.match(html, /运行文档/);
  assert.match(html, /\/tree\/abc\/samples\/vision\/convnext\/conversion/);
  assert.match(html, />论文/);
  assert.match(html, /<div class="mz-kicker"><span class="mz-task">图像分类<\/span><span class="pending-badge">数据待补充<\/span><\/div><h1>/);
  assert.match(html, /title="模型产物待发布"/);
  assert.match(html, /获取模型/);
  assert.match(html, /class="mz-demo"/);
  assert.doesNotMatch(html, /is-frame/);
  assert.match(html, /Runtime 单路/);
  assert.match(html, /mz-report-link is-disabled/);
  assert.match(html, /查看转换详情/);
  assert.doesNotMatch(html, /data-open-download|OE 转换报告|>精度<\/|快速上手/);

  const performanceRoot = { innerHTML: '' };
  const root = {
    addEventListener() {},
    querySelectorAll: () => [],
    querySelector: selector => {
      if (selector === '#benchmark-condition') return { dataset: { value: 'single' }, addEventListener() {} };
      if (selector === '#detail-performance') return performanceRoot;
      return null;
    },
  };
  const context = {
    window: { HubI18n: { locale: 'zh', apply() {} } },
    document: { addEventListener() {}, documentElement: { clientWidth: 1200, classList: { add() {}, remove() {} } } },
    AbortController,
    console,
  };
  vm.runInNewContext(detailSource, context);
  context.window.ModelDetail.bind(root, model, [model], { release: { compatibility: { hardware: 'RDK X5' } } });
  const tiles = performanceRoot.innerHTML;
  assert.match(tiles, /Top-1 准确率/);
  assert.match(tiles, /Top-5 准确率/);
  assert.match(tiles, /检测 mAP50–95|Runtime 延迟/);
  assert.doesNotMatch(tiles, /bpu_usage/);
  const e2e = tiles.split('aria-label="端到端流水线条件">')[1].split('</span></button>')[0];
  for (const label of ['实现', 'CPU 线程', '流水线', 'Runtime 提交', '前处理', '后处理', '端到端吞吐量']) {
    assert.match(e2e, new RegExp(label));
  }
  assert.doesNotMatch(e2e, /条件|来源|文档说明|Runtime 延迟/);
  assert.match(tiles, /aria-label="Runtime 延迟：查看来源和条件"/);
  assert.match(tiles, /single-thread/);
});

test('pending pages without a published task profile use named accuracy tiles and related models', () => {
  const clip = {
    id: 'clip-demo',
    catalogId: 'clip/clip/similarity',
    name: 'CLIP',
    task: '图文匹配',
    taskId: 'image-text-similarity',
    tasks: ['image-text-similarity'],
    family: 'clip',
    description: '图文匹配。',
    source: 'https://github.com/D-Robotics/rdk_model_zoo/tree/abc/samples/vision/clip',
    sourceType: 'repository',
    releaseStatus: 'data-pending',
    dataPending: true,
    hardware: 'RDK X5',
    documentedDownloads: [
      { filename: 'img_encoder.bin', url: 'https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/clip/img_encoder.bin' },
      { filename: 'text_encoder.onnx', url: 'https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/clip/text_encoder.onnx' },
    ],
    sourceInput: { format: 'RGB', dtype: 'float32', layout: 'NCHW', shape: '(1,3,224,224)' },
    quickstart: {
      download: 'python3 samples/vision/clip/model/download.py --target x5',
      run: 'python3 samples/vision/clip/runtime/python/main.py --target x5',
    },
    benchmark: { documentation: true, performance: [], accuracy: [], stages: [], conditionOptions: [] },
  };
  const peer = { ...clip, id: 'siglip-demo', name: 'SigLIP', family: 'siglip', catalogId: 'siglip/siglip/embed', task: '视觉特征提取', taskId: 'image-embedding', tasks: ['image-embedding'] };
  const context = {
    window: {
      HubI18n: { locale: 'zh', apply() {} },
      CatalogFacets: { groups: [{ id: 'multimodal', tasks: [['image-text-similarity', '图文匹配'], ['image-embedding', '视觉特征提取']] }] },
    },
    document: { addEventListener() {}, documentElement: { classList: { add() {}, remove() {} } } },
    AbortController,
    console,
  };
  vm.runInNewContext(detailSource, context);
  const html = context.window.ModelDetail.render({ model: clip, models: [clip, peer], data: { release: { compatibility: { hardware: 'RDK X5' } } } });
  assert.doesNotMatch(html, />精度</);
  assert.match(html, /href="https:\/\/archive\.d-robotics\.cc\/downloads\/rdk_model_zoo\/rdk_x5\/clip\/img_encoder\.bin"/);
  assert.match(html, /<h2>相关模型<\/h2>/);
  assert.match(html, /SigLIP/);
  assert.match(html, /Runtime 单路/);
  assert.match(html, /RGB · FLOAT32 · NCHW · 1 × 3 × 224 × 224/);
  assert.match(html, /<h2>快速上手<\/h2><a href="https:\/\/github.com\/D-Robotics\/rdk_model_zoo\/tree\/abc\/samples\/vision\/clip#quickstart"/);
  assert.match(html, /<span>1<\/span>下载模型/);
  assert.match(html, /<span>2<\/span>运行推理/);
  assert.match(html, /python3 samples\/vision\/clip\/model\/download\.py --target x5/);
  assert.match(html, /python3 samples\/vision\/clip\/runtime\/python\/main\.py --target x5/);
  assert.match(html, /aria-label="复制命令"/);
  assert.doesNotMatch(html, /wget /);

  const performanceRoot = { innerHTML: '' };
  context.window.ModelDetail.bind({
    addEventListener() {},
    querySelectorAll: () => [],
    querySelector: selector => {
      if (selector === '#benchmark-condition') return { dataset: { value: '1' }, addEventListener() {} };
      if (selector === '#detail-performance') return performanceRoot;
      return null;
    },
  }, clip, [clip, peer], { release: { compatibility: { hardware: 'RDK X5' } } });
  const tiles = performanceRoot.innerHTML;
  assert.match(tiles, /Runtime 延迟/);
  assert.match(tiles, /端到端延迟/);
  assert.match(tiles, /吞吐量/);
  assert.match(tiles, /CLIP zero-shot Top-1/);
  assert.match(tiles, /图文检索 R@1/);
  assert.equal(tiles.split('<strong>').length - 1, 5);
});

test('published quick start stays on YOLO pages and a pending step without a command is omitted', () => {
  const published = {
    id: 'resnet-demo',
    catalogId: 'resnet/resnet/cls',
    name: 'ResNet',
    task: '图像分类',
    description: '分类。',
    source: 'https://github.com/D-Robotics/rdk_model_zoo/tree/abc/samples/vision/resnet',
    sourceType: 'repository',
    releasePlatform: 'S100',
    assets: [{ url: 'https://example.invalid/resnet.hbm', filename: 'resnet.hbm', format: 'hbm' }],
    benchmark: { performance: [], accuracy: [] },
  };
  assert.doesNotMatch(renderDetail(published), /快速上手/);

  const yolo = {
    ...published,
    id: 'yolo11-demo',
    catalogId: 'ultralytics_yolo/yolo11/detect',
    name: 'YOLO11 Detect',
    source: 'https://github.com/D-Robotics/rdk_model_zoo/tree/abc/samples/vision/ultralytics_yolo',
    assets: [{ url: 'https://example.invalid/yolo11.bin', filename: 'yolo11n.bin', format: 'bin' }],
  };
  const yoloHtml = renderDetail(yolo);
  assert.match(yoloHtml, /<h2>快速上手<\/h2>/);
  assert.match(yoloHtml, /<span>1<\/span>下载模型/);
  assert.match(yoloHtml, /<span>2<\/span>运行推理/);
  assert.match(yoloHtml, /wget -O samples\/vision\/ultralytics_yolo\/model\/yolo11n\.bin/);

  const pending = {
    ...published,
    dataPending: true,
    releaseStatus: 'data-pending',
    assets: [],
    quickstart: { download: '', run: 'python3 samples/vision/convnext/runtime/python/main.py --target x5' },
  };
  const pendingHtml = renderDetail(pending);
  assert.match(pendingHtml, /<span>1<\/span>运行推理/);
  assert.doesNotMatch(pendingHtml, /下载模型/);
  assert.match(pendingHtml, /python3 samples\/vision\/convnext\/runtime\/python\/main\.py --target x5/);
});
