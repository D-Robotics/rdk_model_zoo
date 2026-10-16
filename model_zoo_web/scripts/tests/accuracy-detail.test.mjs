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
