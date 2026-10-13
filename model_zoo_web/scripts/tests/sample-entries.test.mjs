import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import vm from 'node:vm';

const detailSource = readFileSync(new URL('../../src/detail-view.js', import.meta.url), 'utf8');
const appSource = readFileSync(new URL('../../src/app.js', import.meta.url), 'utf8');
const i18nSource = readFileSync(new URL('../../src/i18n.js', import.meta.url), 'utf8');

function renderDetail(model, models = [model]) {
  const context = {
    window: { HubI18n: { locale: 'zh', apply() {} }, MODEL_PROPERTIES: {} },
    document: {},
    AbortController,
    console,
  };
  vm.runInNewContext(detailSource, context);
  return context.window.ModelDetail.render({
    model, models,
    data: { release: { compatibility: { hardware: 'RDK X5' } } },
  });
}

const baseSample = {
  id: 'bytetrack-bytetrack-tracking-s100p',
  catalogId: 'bytetrack/bytetrack/tracking',
  name: 'ByteTrack',
  variantName: 'ByteTrack',
  family: 'bytetrack',
  modelSize: 'default',
  releaseStatus: 'available',
  entryKind: 'sample',
  releasePlatform: 'S100P',
  releaseHardware: 'RDK S100P',
  task: '目标跟踪',
  taskId: 'multi-object-tracking',
  tasks: ['multi-object-tracking'],
  description: 'BYTE 式多目标跟踪。',
  descriptionEn: 'BYTE-style tracking.',
  platforms: ['S100P'],
  coverImage: 'assets/models/bytetrack-bytetrack-tracking.webp',
  coverLabel: 'RDK X5 板端跟踪输出（track id）',
  sample: 'bytetrack/bytetrack/tracking/default',
  source: 'https://github.com/D-Robotics/rdk_model_zoo/tree/c8eb9d56/samples/vision/bytetrack',
  sourceType: 'repository',
  licenseName: 'MIT',
  licenseUrl: 'https://github.com/ifzhang/ByteTrack/blob/main/LICENSE',
  assets: [],
  sampleProfile: {
    provider: 'ifzhang/ByteTrack',
    sampleCommit: 'c8eb9d56264fbaf8f78265139d053e027bd32b28',
    pipeline: 'detector_cpu_tracker',
    pipelineNote: { zh: '检测器 + CPU 关联。', en: 'Detector plus CPU association.' },
    inputContract: [
      { label: { zh: '输出', en: 'Output' }, value: { zh: '逐帧 track id + 检测框 + 置信度', en: 'Per-frame track ids with boxes and confidences' } },
    ],
    runtimes: ['python', 'cpp'],
    availability: {
      status: 'manual-compile',
      statusLabel: '需自行编译',
      note: { zh: 'S100P 未发布检测模型。', en: 'No S100P detector is published.' },
      guideUrl: 'https://github.com/D-Robotics/rdk_model_zoo/tree/c8eb9d56/samples/vision/bytetrack/conversion',
    },
    verification: { zh: 'S100P 板端回放一致。', en: 'S100P board replay matches.' },
    evidence: [{ label: { zh: '证据摘要', en: 'Evidence summary' }, url: 'sample-evidence/bytetrack-s100p.json' }],
    quickstart: [
      { title: { zh: '确认已按转换指南准备 S100P 检测模型', en: 'Confirm the S100P detector is prepared' }, command: 'test -f samples/vision/bytetrack/model/s100p/yolov5x_672x672_nv12.hbm' },
      { title: { zh: '视频跟踪（S100P）', en: 'Track the video (S100P)' }, command: 'python3 -m samples.vision.bytetrack.runtime.python.main --target s100p --model-path samples/vision/bytetrack/model/s100p/yolov5x_672x672_nv12.hbm' },
    ],
    resources: [],
    missing: [
      { field: 'runtime-latency-1-and-2-threads', status: 'missing', title: { zh: '单路与双路性能', en: 'Single- and dual-stream performance' }, note: { zh: '待测量。', en: 'Pending.' } },
      { field: 'cpu-tracker-oe-report', status: 'not-applicable', title: { zh: 'CPU 跟踪 OE 转换报告', en: 'CPU tracker OE conversion report' }, note: { zh: '无 BPU 转换。', en: 'No BPU conversion.' } },
    ],
  },
};

test('sample detail renders the support matrix, not benchmarks', () => {
  const html = renderDetail(baseSample);
  assert.match(html, /板级支持与验证/);
  assert.match(html, /data-status="manual-compile"/);
  assert.match(html, /需自行编译/);
  assert.match(html, /conversion/, 'the manual-compile platform links its conversion guide');
  assert.doesNotMatch(html, /性能与精度/);
  assert.doesNotMatch(html, /Runtime 并发/);
  assert.doesNotMatch(html, /id="detail-performance"/);
  assert.match(html, /OE 转换报告/, 'the report row keeps its OE placeholder');
});

test('sample detail quickstart follows the selected platform', () => {
  const html = renderDetail(baseSample);
  assert.match(html, /--target s100p/);
  assert.match(html, /test -f samples\/vision\/bytetrack\/model\/s100p\//);
  assert.doesNotMatch(html, /--target x5\b/, 'the S100P page must not show X5 commands');
  assert.match(html, /输入与输出/, 'the contract section carries a sample-specific heading');
});

test('missing items show localized titles, never raw field ids', () => {
  const html = renderDetail(baseSample);
  assert.match(html, /单路与双路性能/);
  assert.match(html, /CPU 跟踪 OE 转换报告/);
  assert.doesNotMatch(html, />runtime-latency-1-and-2-threads</);
  assert.doesNotMatch(html, />cpu-tracker-oe-report</);
  assert.match(html, /data-i18n-en="Single- and dual-stream performance"/);
});

test('input contract labels are bilingual', () => {
  const html = renderDetail(baseSample);
  assert.match(html, /<dt data-i18n-zh="输出" data-i18n-en="Output">/);
  assert.match(html, /<dd data-i18n-zh="[^"]+" data-i18n-en="Per-frame track ids with boxes and confidences">/);
});

test('CPU-tracker entries fetch source, model entries fetch the model', () => {
  const sourceOnly = {
    ...baseSample,
    id: 'hobot-mot-hobot-mot-tracking-x5',
    releaseHardware: 'RDK X5',
    assets: [{ role: 'upstream-source', format: 'tar.gz', filename: 'mot.tar.gz', url: 'https://codeload.github.com/D-Robotics/mot/tar.gz/0120205', sha256: 'a'.repeat(64) }],
    sampleProfile: {
      ...baseSample.sampleProfile,
      availability: { status: 'no-neural-artifact', statusLabel: '无神经网络模型', note: { zh: '无神经网络模型。', en: 'No neural model.' } },
    },
  };
  const trackerHtml = renderDetail(sourceOnly);
  // The CTA button fetches source; the dialog header keeps its generic title.
  assert.match(trackerHtml, /获取源码<\/button>/);
  assert.doesNotMatch(trackerHtml, /获取模型<\/button>/);
  const detectorHtml = renderDetail(baseSample);
  // ByteTrack S100P ships no artifact at all: no primary download button.
  assert.doesNotMatch(detectorHtml, /data-open-download/);
  const published = {
    ...baseSample,
    assets: [{ role: 'detector', format: 'bin', filename: 'yolov5n.bin', url: 'https://archive.d-robotics.cc/x.bin', sha256: 'b'.repeat(64), sizeBytes: 100 }],
    sampleProfile: {
      ...baseSample.sampleProfile,
      availability: { status: 'published-download', statusLabel: '已发布下载', note: { zh: '已发布。', en: 'Published.' } },
    },
  };
  assert.match(renderDetail(published), /获取模型</);
});

test('the download dialog lists every published file of a sample platform', () => {
  const pair = {
    ...baseSample,
    assets: [
      { role: 'parta', format: 'bin', filename: 'parta.bin', url: 'https://archive.d-robotics.cc/parta.bin', sha256: 'c'.repeat(64), sizeBytes: 10 },
      { role: 'partb', format: 'bin', filename: 'partb.bin', url: 'https://archive.d-robotics.cc/partb.bin', sha256: 'd'.repeat(64), sizeBytes: 20 },
    ],
    sampleProfile: {
      ...baseSample.sampleProfile,
      availability: { status: 'published-download', statusLabel: '已发布下载', note: { zh: '已发布。', en: 'Published.' } },
    },
  };
  const html = renderDetail(pair);
  assert.match(html, /data-files=/, 'each choice carries its whole file list');
  assert.match(html, /data-download-links/, 'the footer renders one link per file');
  assert.match(html, /转换指南/, 'a manual-compile platform keeps its guide link in the matrix');
});

test('rebuilt download links re-apply the active locale', () => {
  // syncDownloadDialog inserts fresh zh markup when the dialog opens or the
  // platform changes; the i18n refresh must run after that insertion so an
  // English visitor sees English role labels.
  const inserted = detailSource.indexOf('linkHost.innerHTML');
  const refresh = detailSource.indexOf('refresh();', inserted);
  assert.ok(inserted >= 0 && refresh >= 0, 'refresh() runs after the links are rebuilt');
  assert.ok(detailSource.slice(inserted, refresh).includes(".join('');"),
    'the refresh call follows the completed innerHTML assignment');
});

test('cards badge sample-profile entries', () => {
  assert.match(appSource, /card-sample-badge/, 'the card template carries the sample badge');
  assert.match(appSource, /data-i18n-zh="样例" data-i18n-en="Sample"/);
  assert.match(appSource, /entryKind === 'sample'/);
});

test('i18n covers the new sample strings', () => {
  const context = { window: {}, localStorage: { getItem: () => null, setItem() {} }, document: { documentElement: {}, createTreeWalker: () => ({ nextNode: () => null }), querySelectorAll: () => [] } };
  vm.createContext(context);
  vm.runInNewContext(i18nSource, context);
  const { english } = context.window.HubI18n;
  const pairs = {
    '获取源码': 'Get source',
    '输入与输出': 'Input and output',
    '板级支持与验证': 'Board support and verification',
    '已发布下载': 'Published download',
    '需自行编译': 'Manual compile required',
    '无神经网络模型': 'No neural model',
    '目标跟踪': 'Object tracking',
    '样例': 'Sample',
  };
  for (const [zh, en] of Object.entries(pairs)) {
    assert.equal(english(zh), en, `${zh} must translate`);
  }
});
