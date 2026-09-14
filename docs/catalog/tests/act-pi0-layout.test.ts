import {readFileSync} from 'node:fs';
import {it,expect} from 'vitest';
import type {Catalog} from '../src/catalog/types';
import {renderModelDetails} from '../src/ui/model-details';
const catalog=JSON.parse(readFileSync('public/data/catalog.json','utf8')) as Catalog;
it('publishes ACT support for S100 and S600 with platform-specific sources',()=>{
 const act=catalog.models.find(m=>m.id==='act')!;
 expect(act.variants?.map(v=>v.hardware).sort()).toEqual(['s100','s600']);
 expect(act.variants?.find(v=>v.hardware==='s600')?.benchmarks[0]?.source.path).toBe('models/act/README.md');
});
it('shows Pi0 accuracy numbers without repeating the shared scope in every cell',()=>{
 const model=catalog.models.find(m=>m.id==='pi0')!;
 const page=renderModelDetails(model,{hardware:'s600',locale:'en',repositoryUrl:'https://github.com/D-Robotics/rdk_model_zoo',releaseTag:catalog.release.tag});
 const cells=[...page.querySelectorAll('.model-detail-accuracy-value')];
 expect(cells).toHaveLength(5);
 for(const cell of cells)expect(cell.textContent).not.toContain('fixed real input');
 const conditions=page.querySelector('.model-detail-test-conditions')!;
 expect(conditions.textContent?.match(/One fixed real input/g)).toHaveLength(1);
 for(const value of ['0.4405','0.5903','1.6888','0.852','0.999981858'])expect(cells.map(c=>c.textContent).join(' ')).toContain(value);
 expect(conditions.textContent).toContain('not task success rate');
});
