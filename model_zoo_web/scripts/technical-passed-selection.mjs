import { resolve } from 'node:path';

const isReviewGate = name => /^(?:validation|manual|human)_/i.test(name);
const isDatasetIdentity = name => /^(?:dataset|dota.+)_image_byte_identity$/.test(name);
const rowKey = row => `${row.task}/${row.size}/${row.platform}`.toLowerCase();
const PLATFORM_SCOPED_TECHNICAL_CHECKS = new Map([
  ['onnx_semantic_equivalence_certificate', new Set(['s100p', 's100'])],
  ['shared_float_reference_and_target_conversion', new Set(['s100p', 's100'])],
]);

function rowsOf(document, keys, label) {
  for (const key of keys) {
    if (Array.isArray(document?.[key])) return document[key];
  }
  throw new Error(`${label} does not contain a row array`);
}

function hasValue(value) {
  if (value === null || value === undefined || value === false) return false;
  if (typeof value === 'string') return value.trim().length > 0;
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === 'object') return Object.keys(value).length > 0;
  return true;
}

function evidenceText(value) {
  return typeof value === 'string' ? value : JSON.stringify(value);
}

function assertReviewAuditBinding(review, expectedAuditPath, expectedAuditSha) {
  const report = review?.audit_report;
  if (!report || typeof report.path !== 'string' || typeof report.sha256 !== 'string') {
    throw new Error('review matrix must bind audit_report path and SHA-256');
  }
  if (expectedAuditPath && resolve(report.path) !== resolve(expectedAuditPath)) {
    throw new Error('review matrix binds a different authoritative audit path');
  }
  if (expectedAuditSha && report.sha256 !== expectedAuditSha) {
    throw new Error('review matrix binds a different authoritative audit SHA');
  }
}

function uniqueRows(rows, label, keyField) {
  if (!Array.isArray(rows) || rows.length === 0) {
    throw new Error(`${label} must contain at least one row`);
  }
  const map = new Map();
  for (const row of rows) {
    const key = row && row.task && row.size && row.platform ? rowKey(row) : '';
    if (!key || (keyField && typeof row[keyField] !== 'string') || map.has(key)) {
      throw new Error(`${label} contains a malformed or duplicate row`);
    }
    map.set(key, row);
  }
  return map;
}

export function selectTechnicalPassedEntries({ audit, review, auditPath, auditSha }) {
  assertReviewAuditBinding(review, auditPath, auditSha);
  const audits = uniqueRows(rowsOf(audit, ['entries'], 'authoritative audit'), 'authoritative audit', 'name');
  const reviewRows = rowsOf(review, ['rows', 'entries'], 'review matrix');
  const reviews = uniqueRows(reviewRows.map(row => ({ ...row, variant: row.variant || row.name || rowKey(row) })),
    'review matrix', 'variant');
  if (audits.size !== reviews.size || [...audits.keys()].some(key => !reviews.has(key))) {
    throw new Error('authoritative audit and review matrix identities do not match');
  }

  const declared = audit.required_technical_checks || audit.scope?.required_technical_checks;
  const technicalNamesByRow = [...audits.values()].map(row => {
    if (!Array.isArray(row.checks)) throw new Error(`Audit row ${row.name} has no check list`);
    return new Set(row.checks
      .filter(check => check && typeof check.name === 'string' && !isReviewGate(check.name) && !isDatasetIdentity(check.name))
      .map(check => check.name));
  });
  const commonTechnicalChecks = [...new Set(technicalNamesByRow.flatMap(names => [...names]))]
    .filter(name => !PLATFORM_SCOPED_TECHNICAL_CHECKS.has(name)).sort();
  if (!commonTechnicalChecks.length && !Array.isArray(declared)) {
    throw new Error('audit has no authoritative required technical checks');
  }

  const selected = [];
  const excluded = [];
  for (const [key, row] of audits) {
    const reviewRow = reviews.get(key);
    if (!Array.isArray(row.checks)) throw new Error(`Audit row ${key} has no check list`);
    const checkMap = new Map();
    for (const check of row.checks) {
      if (!check || typeof check.name !== 'string' || checkMap.has(check.name)) {
        throw new Error(`Audit row ${key} has a malformed or duplicate check`);
      }
      checkMap.set(check.name, check);
    }
    const datasetIdentityChecks = [...checkMap.keys()].filter(isDatasetIdentity);
    const technicalChecks = [...checkMap.values()].filter(check => !isReviewGate(check.name));
    const failedTechnical = technicalChecks.filter(check => check.passed !== true).map(check => check.name);
    const anomaly = hasValue(reviewRow.anomaly) ? evidenceText(reviewRow.anomaly) : '';
    if (!Array.isArray(reviewRow.missing_technical_evidence)) {
      throw new Error(`Review row ${key} has no missing_technical_evidence list`);
    }
    const missingReviewEvidence = (reviewRow.missing_technical_evidence || [])
      .map(evidenceText)
      .filter(value => !/(?:validation\.json|validation_|human|manual|approval)/i.test(value));
    const reviewAudit = reviewRow.technical_audit;
    if (!reviewAudit || reviewAudit.audit_report_path !== review.audit_report.path
        || reviewAudit.audit_report_sha256 !== review.audit_report.sha256) {
      throw new Error(`Review row ${key} does not bind the review's authoritative audit`);
    }
    const reviewTechnicalFailure = reviewAudit && (
      reviewAudit.passed !== true
      || !Number.isInteger(reviewAudit.checks_total)
      || reviewAudit.checks_total < 1
      || reviewAudit.checks_passed !== reviewAudit.checks_total
      || !Array.isArray(reviewAudit.failed_checks)
      || reviewAudit.failed_checks.length > 0
    );
    const rowDeclared = row.required_technical_checks || reviewAudit.required_technical_checks || [];
    const requiredBase = [...new Set([
      ...commonTechnicalChecks,
      ...(Array.isArray(declared) ? declared : []),
      ...(Array.isArray(rowDeclared) ? rowDeclared : []),
    ].filter(name => typeof name === 'string' && !isReviewGate(name) && !isDatasetIdentity(name)))];
    const requiredTechnicalChecks = [...new Set([
      ...requiredBase.filter(name => !PLATFORM_SCOPED_TECHNICAL_CHECKS.has(name)
        || PLATFORM_SCOPED_TECHNICAL_CHECKS.get(name).has(String(row.platform).toLowerCase())),
      ...[...PLATFORM_SCOPED_TECHNICAL_CHECKS]
        .filter(([, platforms]) => platforms.has(String(row.platform).toLowerCase()))
        .map(([name]) => name),
    ])].sort();
    if (!requiredTechnicalChecks.length) {
      throw new Error(`Audit row ${key} has no applicable required technical checks`);
    }
    const rowMissingRequired = requiredTechnicalChecks.filter(name => !checkMap.has(name));
    if (datasetIdentityChecks.length !== 1) rowMissingRequired.push('exactly_one_dataset_identity_check');
    let reason;
    if (rowMissingRequired.length) reason = 'missing_required_technical_checks';
    else if (failedTechnical.length) reason = 'technical_check_failed';
    else if (reviewTechnicalFailure) reason = 'review_technical_audit_failed';
    else if (missingReviewEvidence.length) reason = 'review_missing_technical_evidence';
    else if (anomaly) reason = 'review_anomaly';

    const item = {
      audit_name: row.name,
      task: row.task,
      size: row.size,
      platform: row.platform,
      model_id: `ultralytics_yolo/yolo26/${row.task}/${row.size}/${row.platform}`,
      required_technical_checks: [...requiredTechnicalChecks, ...datasetIdentityChecks],
      technical_checks_passed: technicalChecks.filter(check => check.passed === true).map(check => check.name),
      required_check_failures: rowMissingRequired,
      failed_technical_checks: failedTechnical,
      review_missing_technical_evidence: missingReviewEvidence,
      pending_review_gates: [...checkMap.values()].filter(check => isReviewGate(check.name) && check.passed !== true).map(check => check.name),
      review_technical_audit: reviewAudit ? {
        checks_passed: reviewAudit.checks_passed,
        checks_total: reviewAudit.checks_total,
        failed_checks: reviewAudit.failed_checks || [],
      } : null,
      validation_json_pending: (reviewRow.missing_technical_evidence || []).some(value => /validation\.json/i.test(evidenceText(value))),
      anomaly,
    };
    if (reason) excluded.push({ ...item, reason });
    else selected.push(item);
  }
  return { selected, excluded };
}

export { isReviewGate, isDatasetIdentity };
