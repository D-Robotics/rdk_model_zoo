import { readFileSync } from 'node:fs';
import { expect, it } from 'vitest';
import type { Catalog } from '../src/catalog/types';
import { renderEditorialIntro } from '../src/ui/editorial-shell';
import { renderModelDetails } from '../src/ui/model-details';

const catalog = JSON.parse(readFileSync('public/data/catalog.json', 'utf8')) as Catalog;
it('uses the full catalog for family and platform configuration counts', () => {
  const page = renderEditorialIntro('zh', catalog);
  expect(page.querySelector('[data-count="families"]')?.textContent).toBe('54');
  expect(page.querySelector('[data-count="models"]')?.textContent).toBe('584');
  expect(page.querySelector('[data-platform="x5"]')?.textContent).toContain('197');
  expect(page.querySelector('[data-platform="s100"]')?.textContent).toContain('136');
  expect(page.textContent).toContain('按平台分别计数');
});
for (const [id, hardware] of [['paraformer','s100'],['himloco','x5'],['siglip','s100']] as const) {
  it(`separates ${id} performance and accuracy without losing source evidence`, () => {
    const model = catalog.models.find(m => m.id === id)!;
    const page = renderModelDetails(model, {hardware, locale:'zh', repositoryUrl:'https://github.com/D-Robotics/rdk_model_zoo',releaseTag:catalog.release.tag});
    expect(page.querySelector('.representative-performance')).not.toBeNull();
    expect(page.querySelector('.model-detail-conditions-host')).toBeNull();
    expect(page.querySelector('.representative-accuracy')).not.toBeNull();
    const panels = [...page.querySelectorAll<HTMLElement>('.representative-configuration')];
    expect(panels.filter(p=>!p.hidden)).toHaveLength(1);
    if (id === 'siglip') {
      const select = page.querySelector<HTMLSelectElement>('.representative-selector select')!;
      select.value = panels[1]!.dataset.variantId!;
      select.dispatchEvent(new Event('change'));
      expect(panels[0]!.hidden).toBe(true);
      expect(panels[1]!.hidden).toBe(false);
    }
    if(id === 'himloco') {
      expect(page.querySelector('.representative-performance')?.textContent).toContain('p95');
      expect(page.querySelector('.representative-performance')?.textContent).toContain('0.942');
      expect(page.querySelector('.representative-estimate')?.textContent).toContain('63 us');
      const runtime = page.querySelector<HTMLSelectElement>('.representative-performance select')!;
      runtime.value = 'C++';
      runtime.dispatchEvent(new Event('change'));
      const visible = [...page.querySelectorAll<HTMLTableRowElement>('.representative-performance tbody tr')].filter(row=>!row.hidden);
      expect(visible).toHaveLength(1);
      expect(visible[0]!.textContent).toContain('0.364 ms');
    }
    if(id === 'paraformer') {
      expect(page.querySelector('.representative-performance')?.textContent).toContain('33.63 ms');
      expect(page.querySelector('.representative-performance')?.textContent).toContain('33.15 ms');
    }
  });
}
it('lets YOLOv8 readers expand the other measured threads without duplicating configurations', () => {
  window.history.replaceState({}, '', '/');
  const model = catalog.models.find(m=>m.id==='yolov8')!;
  const page = renderModelDetails(model, {hardware:'x5',task:'object-detection',locale:'zh',repositoryUrl:'https://github.com/D-Robotics/rdk_model_zoo',releaseTag:catalog.release.tag});
  const rowCount = page.querySelectorAll('.model-detail-spec-row').length;
  // The X5 records do not state the profiling tool, so the conditions block
  // reports only what they do state: the timing scope every row shares is
  // stated once below the table instead of beside every value.
  expect(page.querySelector('.model-detail-test-conditions')?.textContent).toContain('BPU task');
  expect([...page.querySelectorAll('.model-detail-measurement-scope')].some(note => /BPU task/.test(note.textContent ?? ''))).toBe(false);
  expect(page.querySelector('.model-detail-test-conditions a')?.getAttribute('href')).toContain('/blob/' + model.variants!.find(v => v.hardware === 'x5' && v.task === 'object-detection')!.benchmarks[0]!.source.ref + '/');

  expect(page.querySelector('.model-detail-spec-row .model-detail-latency[data-concurrency="2"]')).toBeNull();
  page.querySelector<HTMLButtonElement>('[data-action="toggle-threads"]')!.click();
  expect(page.querySelectorAll('.model-detail-spec-row')).toHaveLength(rowCount);
  expect(page.querySelector('.model-detail-spec-row .model-detail-latency[data-concurrency="2"]')?.textContent).toContain('8 ms');
  window.history.replaceState({}, '', '/');
});
it('shows the S600 YOLOv8n source measurements in both thread modes', () => {
  window.history.replaceState({}, '', '/');
  const model = catalog.models.find(m=>m.id==='yolov8')!;
  const page = renderModelDetails(model, {hardware:'s600',task:'object-detection',locale:'zh',repositoryUrl:'https://github.com/D-Robotics/rdk_model_zoo',releaseTag:catalog.release.tag});
  const row = () => page.querySelector('[data-variant-id="yolov8n-detect-640-object-detection-s600"]')!;
  expect(row().querySelector('.model-detail-latency')?.textContent).toContain('0.776');
  expect(row().querySelector('.model-detail-throughput')?.textContent).toContain('1,258.503');
  page.querySelector<HTMLButtonElement>('[data-action="toggle-threads"]')!.click();
  expect(row().querySelector('.model-detail-latency[data-concurrency="12"]')?.textContent).toContain('1.674');
  expect(row().querySelector('.model-detail-throughput[data-concurrency="12"]')?.textContent).toContain('6,512.325');
});
