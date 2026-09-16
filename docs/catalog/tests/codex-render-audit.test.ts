import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, it, expect } from 'vitest';
import { renderModelDetails } from '../src/ui/model-details';
import { buildModelCardViewModel } from '../src/catalog/card-view-model';
import type { Catalog, HardwareId } from '../src/catalog/types';

// Independent inspection artifact; successful rendering is NOT sample acceptance.
const auditEnabled = process.env.CODEX_RENDER_AUDIT === '1';
describe.skipIf(!auditEnabled)('independent per-family render inventory', () => {
  it('captures every hardware/task detail for manual source comparison', () => {
    const catalog = JSON.parse(readFileSync(resolve('public/data/catalog.json'), 'utf8')) as Catalog;
    const reports: unknown[] = [];
    for (const model of catalog.models) {
      const variants = model.variants ?? [];
      const combinations = [...new Set(variants.map(v => `${v.hardware}|${v.task}`))];
      for (const combination of combinations) {
        const [hardware, task] = combination.split('|');
        for (const locale of ['en', 'zh'] as const) {
          window.history.replaceState({}, '', '/?threads=all');
          const page = renderModelDetails(model, {locale, hardware:hardware as HardwareId, task,
            repositoryUrl:'https://github.com/D-Robotics/rdk_model_zoo', releaseTag:catalog.release.tag});
          document.body.replaceChildren(page);
          const rows = [...page.querySelectorAll('.model-detail-spec-row')].map(row => ({
            variant:(row as HTMLElement).dataset.variantId,
            cells:[...row.querySelectorAll('th,td')].map(cell=>cell.textContent),
            metrics:[...row.querySelectorAll('.model-detail-metric-value,.model-detail-accuracy-value')].map(metric=>({
              ...((metric as HTMLElement).dataset), text:metric.textContent
            }))
          }));
          const idList = [...page.querySelectorAll('[id]')].map(el=>el.id);
          reports.push({family:model.id,hardware,task,locale,
            card:buildModelCardViewModel(model,hardware),rows,
            repeatedDomIds:idList.filter((id,i)=>idList.indexOf(id)!==i),
            headers:[...page.querySelectorAll('thead th')].map(el=>el.textContent),
            text:page.textContent});
          expect(page.querySelector('h1')?.textContent).toBe(model.name);
        }
      }
    }
    const destination = resolve('../../audit');
    mkdirSync(destination,{recursive:true});
    writeFileSync(resolve(destination,'rendered-details.current.json'),JSON.stringify(reports,null,2)+'\n');
    expect(reports.length).toBeGreaterThan(0);
  }, 120000);
});
