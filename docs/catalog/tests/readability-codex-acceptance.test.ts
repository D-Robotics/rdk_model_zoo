import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { expect, it } from 'vitest';
import type { Catalog } from '../src/catalog/types';
import { renderModelDetails } from '../src/ui/model-details';

it('independently accepts every sample/board/task without losing configuration evidence', () => {
  const catalog = JSON.parse(readFileSync('public/data/catalog.json', 'utf8')) as Catalog;
  const acceptance: object[] = [];
  window.history.replaceState({}, '', '/?threads=all');
  for (const model of catalog.models) {
    const variants = model.variants ?? [];
    for (const key of new Set(variants.map(v => `${v.hardware}|${v.task}`))) {
      const selected = variants.filter(v => `${v.hardware}|${v.task}` === key);
      const page = renderModelDetails(model, { hardware: selected[0]!.hardware, task: selected[0]!.task, locale: 'zh', repositoryUrl: 'https://github.com/D-Robotics/rdk_model_zoo', releaseTag: catalog.release.tag });
      const rows = [...page.querySelectorAll<HTMLElement>('.model-detail-spec-row')];
      expect(rows.length, `${model.id}/${key}`).toBe(selected.length);
      let metrics = 0;
      for (const variant of selected) {
        const own = rows.filter(row => row.dataset.variantId === variant.id);
        expect(own.length, variant.id).toBe(1);
        const evidence = own[0]!.textContent! + own[0]!.nextElementSibling!.textContent!;
        for (const record of variant.benchmarks) {
          for (const metric of [...record.performance ?? [], ...record.accuracy ?? []]) {
            expect(evidence, `${variant.id}/${record.id}/${metric.metric}`).toContain(metric.value.toLocaleString('en-US', {maximumFractionDigits:12}));
            metrics++;
          }
        }
      }
      acceptance.push({ family: model.id, board: selected[0]!.hardware, task: selected[0]!.task, samples: [...new Set(selected.map(v => v.sample_path))], configurations: selected.length, metrics, status: 'host-verified' });
    }
  }
  mkdirSync('audit', { recursive: true });
  writeFileSync('audit/readability-codex-acceptance.json', JSON.stringify(acceptance, null, 2));
  window.history.replaceState({}, '', '/');
}, 30000);
