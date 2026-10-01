import test from 'node:test';
import assert from 'node:assert/strict';
import { selectTechnicalPassedEntries } from '../technical-passed-selection.mjs';

const auditPath = '/workbench/release-audit.json';
const auditSha = 'a'.repeat(64);
const requiredChecks = ['batch8_conversion', 'board_full_dataset', 'runtime_performance'];

function fixture({ declareRequired = true } = {}) {
  const audit = { entries: [] };
  if (declareRequired) audit.required_technical_checks = requiredChecks;
  const review = {
    audit_report: { path: auditPath, sha256: auditSha },
    rows: [],
  };
  for (const task of ['cls', 'seg', 'pose', 'obb']) {
    for (const size of ['n', 's', 'm', 'l', 'x']) {
      for (const platform of ['s600', 's100p', 's100']) {
        const name = `${task}-${size}-${platform}`;
        const datasetCheck = task === 'obb' ? 'dota_remote_image_byte_identity' : 'dataset_image_byte_identity';
        const platformChecks = platform === 's600' ? [] : [
          { name: 'onnx_semantic_equivalence_certificate', passed: true },
          { name: 'shared_float_reference_and_target_conversion', passed: true },
        ];
        audit.entries.push({
          name, task, size, platform,
          checks: [
            ...requiredChecks.map(checkName => ({ name: checkName, passed: true })),
            ...platformChecks,
            { name: datasetCheck, passed: true },
            { name: 'validation_file', passed: false },
            { name: 'manual_review_receipt_file', passed: false },
            { name: 'human_accuracy_delta_review', passed: false },
            { name: 'human_release_approval', passed: false },
          ],
        });
        const anomaly = (task === 'pose' && size === 's' && platform === 's100p')
          || (task === 'obb' && size === 'x') ? 'retention anomaly' : '';
        review.rows.push({
          variant: name, task, size, platform,
          missing_technical_evidence: ['validation.json (11 validation gates)'],
          anomaly,
          technical_audit: {
            passed: true,
            checks_passed: 29,
            checks_total: 29,
            failed_checks: [],
            audit_report_path: auditPath,
            audit_report_sha256: auditSha,
          },
        });
      }
    }
  }
  return { audit, review };
}

test('selects all 56 S-platform rows that pass technical gates and have no review anomaly', () => {
  const { selected, excluded } = selectTechnicalPassedEntries({
    ...fixture(), auditPath, auditSha,
  });
  assert.equal(selected.length, 56);
  assert.deepEqual(Object.fromEntries(['cls', 'seg', 'pose', 'obb'].map(task =>
    [task, selected.filter(row => row.task === task).length])),
  { cls: 15, seg: 15, pose: 14, obb: 12 });
  assert.deepEqual(Object.fromEntries(['s600', 's100p', 's100'].map(platform =>
    [platform, selected.filter(row => row.platform === platform).length])),
  { s600: 19, s100p: 18, s100: 19 });
  assert.equal(selected.find(row => row.platform === 's600').required_technical_checks.length, 4);
  assert.equal(selected.find(row => row.platform === 's100').required_technical_checks.length, 6);
  assert.ok(selected.every(row => row.validation_json_pending && row.pending_review_gates.length > 0));
  assert.equal(excluded.length, 4);
  assert.equal(excluded.find(row => row.audit_name === 'pose-s-s100p').reason, 'review_anomaly');
  assert.equal(excluded.filter(row => row.task === 'obb' && row.size === 'x').length, 3);
  assert.ok(excluded.filter(row => row.reason === 'review_anomaly').every(row => row.anomaly));
});

test('a failed or missing required technical check excludes only that row', () => {
  const input = fixture();
  input.audit.entries.find(row => row.name === 'cls-n-s600').checks
    .find(check => check.name === 'batch8_conversion').passed = false;
  input.audit.entries.find(row => row.name === 'seg-s-s100p').checks = input.audit.entries
    .find(row => row.name === 'seg-s-s100p').checks
    .filter(check => check.name !== 'runtime_performance');
  const { selected, excluded } = selectTechnicalPassedEntries({ ...input, auditPath, auditSha });
  assert.equal(selected.length, 54);
  assert.equal(excluded.find(row => row.audit_name === 'cls-n-s600').reason, 'technical_check_failed');
  assert.equal(excluded.find(row => row.audit_name === 'seg-s-s100p').reason, 'missing_required_technical_checks');
});

test('platform scoped float-reference checks are required on S100/P but not S600', () => {
  const input = fixture();
  const s100 = input.audit.entries.find(row => row.name === 'cls-n-s100');
  s100.checks = s100.checks.filter(check => check.name !== 'onnx_semantic_equivalence_certificate');
  const s600 = input.audit.entries.find(row => row.name === 'cls-n-s600');
  assert.ok(!s600.checks.some(check => check.name === 'onnx_semantic_equivalence_certificate'));
  const { selected, excluded } = selectTechnicalPassedEntries({ ...input, auditPath, auditSha });
  assert.equal(selected.length, 55);
  assert.equal(excluded.find(row => row.audit_name === 'cls-n-s100').reason, 'missing_required_technical_checks');
  assert.ok(selected.some(row => row.audit_name === 'cls-n-s600'));
});

test('audit-wide technical check union catches a missing common gate on each board', () => {
  const input = fixture({ declareRequired: false });
  for (const platform of ['s600', 's100p', 's100']) {
    const row = input.audit.entries.find(item => item.name === `seg-n-${platform}`);
    row.checks = row.checks.filter(check => check.name !== 'board_full_dataset');
  }
  const { selected, excluded } = selectTechnicalPassedEntries({ ...input, auditPath, auditSha });
  assert.equal(selected.length, 53);
  for (const platform of ['s600', 's100p', 's100']) {
    const row = excluded.find(item => item.audit_name === `seg-n-${platform}`);
    assert.equal(row.reason, 'missing_required_technical_checks');
    assert.ok(row.required_check_failures.includes('board_full_dataset'));
  }
});

test('review matrix audit and per-row technical evidence must bind to the supplied audit', () => {
  const input = fixture();
  input.review.rows[0].technical_audit.audit_report_sha256 = 'b'.repeat(64);
  assert.throws(() => selectTechnicalPassedEntries({ ...input, auditPath, auditSha }), /does not bind/);
  const topLevelMismatch = fixture();
  topLevelMismatch.review.audit_report.path = '/workbench/other-audit.json';
  assert.throws(() => selectTechnicalPassedEntries({ ...topLevelMismatch, auditPath, auditSha }), /different authoritative audit path/);
  assert.throws(() => selectTechnicalPassedEntries({ ...fixture(), auditPath, auditSha: 'b'.repeat(64) }), /different authoritative audit SHA/);
});

test('duplicate checks and mismatched audit/review identities fail closed', () => {
  const duplicate = fixture();
  duplicate.audit.entries[0].checks.push({ ...duplicate.audit.entries[0].checks[0] });
  assert.throws(() => selectTechnicalPassedEntries({ ...duplicate, auditPath, auditSha }), /duplicate check/);
  const mismatch = fixture();
  mismatch.review.rows.pop();
  assert.throws(() => selectTechnicalPassedEntries({ ...mismatch, auditPath, auditSha }), /identities do not match/);
});

test('review technical gate failure and missing evidence are excluded', () => {
  const input = fixture();
  const row = input.review.rows.find(item => item.variant === 'cls-n-s600');
  row.technical_audit.failed_checks = ['cpp_e2e_logs'];
  row.technical_audit.passed = false;
  row.anomaly = '';
  const missing = input.review.rows.find(item => item.variant === 'seg-n-s600');
  missing.missing_technical_evidence = ['conversion receipt SHA is missing'];
  const { selected, excluded } = selectTechnicalPassedEntries({ ...input, auditPath, auditSha });
  assert.equal(selected.length, 54);
  assert.equal(excluded.find(item => item.audit_name === 'cls-n-s600').reason, 'review_technical_audit_failed');
  assert.equal(excluded.find(item => item.audit_name === 'seg-n-s600').reason, 'review_missing_technical_evidence');
});
