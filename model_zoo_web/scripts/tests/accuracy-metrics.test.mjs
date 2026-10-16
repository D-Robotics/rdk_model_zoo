import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  accuracyMetricLabels,
  accuracyMetricsSchema,
  comparableAccuracyRecords,
  accuracyRecords,
} from '../accuracy-metrics.mjs';

const values = fields => Object.fromEntries(fields.map((field, index) => [field, (index + 1) / 10]));

test('each YOLO26 task maps its measured fields to float and quantized preview metrics', () => {
  const receiptFields = {
    cls: [['top1', 'top1'], ['top5', 'top5']],
    seg: [['bbox/AP', 'box_ap'], ['segm/AP', 'mask_ap']],
    pose: [['keypoints/AP', 'keypoints_ap']],
    obb: [['mAP50', 'map_50']],
  };
  for (const [task, profile] of Object.entries(accuracyMetricsSchema.tasks)) {
    if (receiptFields[task]) {
      assert.deepEqual(profile.display.map(item => [item.receipt_field, item.field]), receiptFields[task], task);
    }
    const fields = profile.required;
    const float = values(fields);
    const runtime = Object.fromEntries(fields.map(field => [field, float[field] - 0.05]));
    const records = accuracyRecords(task, float, runtime, {
      dataset: profile.dataset,
      evaluation_scope: profile.evaluation_scope?.value,
    });

    assert.equal(records.length, profile.display.length * 2, task);
    for (const item of profile.display) {
      assert.ok(records.some(record => record.metric === item.metric && record.model_stage === 'float'), task);
      assert.ok(records.some(record => record.metric === item.metric && record.model_stage === 'quantized'), task);
    }
  }
});

test('classification aliases and localized metric labels are available to preview and detail pages', () => {
  assert.deepEqual(
    accuracyRecords('classify', { top1: 0.8, top5: 0.95 }, { top1: 0.79, top5: 0.94 })
      .map(record => record.metric),
    ['top-1', 'top-1', 'top-5', 'top-5'],
  );
  assert.equal(accuracyMetricLabels()['top-5'].en, 'Top-5 accuracy');
  assert.equal(accuracyMetricLabels()['obb-map-50'].zh, '旋转框 mAP50');
});

test('preview accuracy serialization rejects missing or out-of-range task metrics', () => {
  assert.throws(
    () => accuracyRecords('pose', { keypoints_ap: 0.8 }, {}),
    /quantized\.keypoints_ap/,
  );
  assert.throws(
    () => accuracyRecords('obb', { map_50: 1.01 }, { map_50: 0.9 }, {
      dataset: 'DOTA val',
      evaluation_scope: 'local_dota_val_single_scale',
    }),
    /float\.map_50/,
  );
  assert.throws(
    () => accuracyRecords('obb', { map_50: 0.8 }, { map_50: 0.79 }, {
      dataset: 'DOTA test',
      evaluation_scope: 'official_test',
    }),
    /dataset must be DOTA val/,
  );
});

test('valid comparable evidence pairs float/runtime metrics for all four YOLO26 tasks', () => {
  for (const task of ['cls', 'seg', 'pose', 'obb']) {
    const profile = accuracyMetricsSchema.tasks[task];
    const float = values(profile.required);
    const runtime = Object.fromEntries(profile.required.map(field => [field, float[field] - 0.05]));
    const metadata = {
      dataset: profile.dataset || `${task} evaluation split`,
      evaluation_scope: profile.evaluation_scope?.value,
    };
    const comparisonScope = {
      comparable: true,
      kind: 'end-to-end-metric-comparison',
      meaning: 'Same evaluation split measured through float and board paths.',
      not_a_pure_quantization_loss_estimate: true,
      reason: 'Input adapter differs.',
    };
    const comparison = {
      status: 'valid-evidence',
      direct_metric_comparison: 'valid',
      comparison_scope: comparisonScope,
      metrics: Object.fromEntries(profile.display.map(({ field }) => [field, {
        float: float[field],
        board: runtime[field],
        board_minus_float: runtime[field] - float[field],
      }])),
    };

    const records = comparableAccuracyRecords(task, float, runtime, metadata, comparison);
    assert.equal(records.length, profile.display.length * 2, task);
    assert.ok(records.every(record => record.dataset === metadata.dataset), task);
    assert.ok(records.every(record => record.comparison_scope.comparable === true), task);
    assert.ok(records.every(record => record.comparison_scope.not_a_pure_quantization_loss_estimate), task);
    if (profile.evaluation_scope) {
      assert.ok(records.every(record => record.scope === profile.evaluation_scope.labels), task);
    }
  }
});

test('paired accuracy rejects invalid scope flags and mismatched comparison metrics', () => {
  const task = 'pose';
  const float = { keypoints_ap: 0.7 };
  const runtime = { keypoints_ap: 0.65 };
  const metadata = { dataset: 'COCO val2017 full 5000' };
  const comparison = {
    status: 'valid-evidence',
    direct_metric_comparison: 'valid',
    comparison_scope: {
      comparable: true,
      kind: 'end-to-end-metric-comparison',
      not_a_pure_quantization_loss_estimate: true,
    },
    metrics: { keypoints_ap: { float: 0.7, board: 0.65, board_minus_float: -0.05 } },
  };

  assert.throws(
    () => comparableAccuracyRecords(task, float, runtime, metadata, {
      ...comparison,
      comparison_scope: { ...comparison.comparison_scope, comparable: false },
    }),
    /valid comparable end-to-end evidence/,
  );
  assert.throws(
    () => comparableAccuracyRecords(task, float, runtime, metadata, {
      ...comparison,
      metrics: { keypoints_ap: { float: 0.7, board: 0.6, board_minus_float: -0.1 } },
    }),
    /do not match float\/runtime receipts/,
  );
});
