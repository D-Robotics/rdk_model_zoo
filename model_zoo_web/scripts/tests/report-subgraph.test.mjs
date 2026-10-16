import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import vm from 'node:vm';

const reportsSource = readFileSync(new URL('../../src/reports.js', import.meta.url), 'utf8');

async function renderReport(data) {
  const nodeBody = { innerHTML: '' };
  const root = {
    innerHTML: '',
    querySelectorAll: () => [],
    querySelector: selector => selector === '#rd-node-body' ? nodeBody : null,
  };
  const reportApp = { innerHTML: '' };
  const document = {
    getElementById: id => id === 'report-app' ? reportApp : id === 'rd-root' ? root : null,
    querySelector: () => ({ addEventListener() {} }),
  };
  const window = {
    OE_REPORTS: [{ id: 'fixture', name: 'Fixture', modelIds: [], dataUrl: '/fixture.json' }],
    MODEL_DATA: { models: [] },
    HubI18n: { apply() {}, setLocale() {} },
    addEventListener() {},
  };
  const context = {
    window,
    document,
    URLSearchParams,
    location: { search: '?report=fixture' },
    fetch: async () => ({ ok: true, json: async () => data }),
    console,
  };
  vm.runInNewContext(reportsSource, context);
  await new Promise(resolve => setImmediate(resolve));
  return { bodyHtml: nodeBody.innerHTML, reportHtml: root.innerHTML };
}

test('S-series report hides the subgraph column when conversion data omits it', async () => {
  const s600Data = JSON.parse(readFileSync(
    new URL('../../release/reports/yolo11n-s600-oe-data.json', import.meta.url),
    'utf8',
  ));
  const { bodyHtml, reportHtml } = await renderReport(s600Data);

  assert.doesNotMatch(reportHtml, /<th>子图<\/th>/);
  assert.doesNotMatch(bodyHtml, /undefined/);
});

test('X5 report keeps the explicit subgraph index in the table', async () => {
  const x5Data = JSON.parse(readFileSync(
    new URL('../../release/reports/yolo11n-x5-oe-data.json', import.meta.url),
    'utf8',
  ));
  const { bodyHtml, reportHtml } = await renderReport(x5Data);

  assert.match(reportHtml, /<th>子图<\/th>/);
  assert.match(bodyHtml, /<td>0<\/td>/);
  assert.doesNotMatch(bodyHtml, /undefined/);
});

test('partial subgraph metadata renders a dash for nodes without an index', async () => {
  const { bodyHtml, reportHtml } = await renderReport({
    quantization: {
      nodes: [
        { name: 'node-with-index', on: 'BPU', subgraph: 0, type: 'Conv' },
        { name: 'node-without-index', on: 'BPU', type: 'Conv' },
      ],
    },
  });

  assert.match(reportHtml, /<th>子图<\/th>/);
  assert.match(bodyHtml, /<td>0<\/td>/);
  assert.match(bodyHtml, /<td>—<\/td>/);
  assert.doesNotMatch(bodyHtml, /undefined/);
});
