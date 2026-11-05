// @vitest-environment node
import {describe,expect,it} from 'vitest';
import {repositoryCatalog} from './helpers/repository';

describe('independent complete transformer conversions',()=>{
  it('retains an exact S100 artifact without inheriting historical measurements',async()=>{
    const catalog=await repositoryCatalog();
    const expected=[["asr", "https://archive.d-robotics.cc/downloads/rdk_model_zoo_web/rdk_s100/asr/asr_xlsr53_zh3503_30k_mask_int16_cpuptq_o1_20261105_nashe.hbm"]];
    for(const [source,url] of expected){
      const matches=catalog.models.flatMap(m=>m.variants??[])
        .filter(v=>v.assets.some(a=>a.url===url));
      expect(matches,source).toHaveLength(1);
      expect(matches[0]!.hardware).toBe('s100');
      expect(matches[0]!.benchmarks).toEqual([]);
    }
  });
});
