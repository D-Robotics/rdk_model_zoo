import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import vm from 'node:vm';

const facetsSource = readFileSync(new URL('../../src/facets.js',import.meta.url),'utf8');
const detailSource = readFileSync(new URL('../../src/detail-view.js',import.meta.url),'utf8');

test('availability separates hashed artifacts, historical links and verified board results',()=>{
  const context={window:{}};
  vm.runInNewContext(facetsSource,context);
  const facets=context.window.CatalogFacets;
  const legacy={releaseStatus:'data-pending',assets:[],documentedDownloads:[{url:'https://example.org/old.bin'}]};
  const converted={releaseStatus:'converted',assets:[{url:'https://example.org/new.bin',sha256:'a'.repeat(64),sizeBytes:1024}]};
  const verified={...converted,releaseStatus:'released'};
  assert.equal(facets.matchesAvailability(legacy,'downloadable'),false);
  assert.equal(facets.matchesAvailability(converted,'downloadable'),true);
  assert.equal(facets.matchesAvailability(converted,'verified'),false);
  assert.equal(facets.matchesAvailability(verified,'verified'),true);
});

test('a converted artifact has downloads and task-labelled unmeasured tiles without inherited metrics',()=>{
  const grid={innerHTML:''};
  const root={addEventListener(){},querySelectorAll:()=>[],querySelector:selector=>selector==='#detail-performance'?grid:null};
  const context={window:{HubI18n:{locale:'zh',apply(){}},MODEL_PROPERTIES:{}},
    document:{addEventListener(){},documentElement:{classList:{add(){},remove(){}}}},AbortController,console};
  vm.runInNewContext(detailSource,context);
  const model={id:'convnext-atto-x5',catalogId:'convnext/convnext/cls',name:'ConvNeXt',variantName:'Atto D2',
    releaseStatus:'converted',releasePlatform:'X5',modelSize:'atto-d2-224',task:'图像分类',taskId:'image-classification',
    hardware:'RDK X5',platforms:['X5'],assets:[{filename:'new.bin',url:'https://example.org/new.bin',sha256:'a'.repeat(64),sizeBytes:1024}],
    source:'https://github.com/D-Robotics/rdk_model_zoo/tree/'+'a'.repeat(40)+'/samples/vision/convnext',
    benchmark:{precision:'int8',performance:[],accuracy:[],endToEnd:[]},runtimeCommand:'python3 example.py --model-path new.bin'};
  const data={release:{compatibility:{hardware:'RDK X5'}}};
  const html=context.window.ModelDetail.render({model,models:[model],data});
  assert.doesNotMatch(html,/模型可下载 · 板端待测/);
  assert.match(html,/data-open-download/);
  assert.match(html,/python3 example.py --model-path new.bin/);
  assert.doesNotMatch(html,/data-ask-model/);
  context.window.ModelDetail.bind(root,model,[model],data);
  assert.match(grid.innerHTML,/Top-1/);
  assert.match(grid.innerHTML,/Top-5/);
  assert.equal((grid.innerHTML.match(/待实测/g)||[]).length,5);
  assert.doesNotMatch(grid.innerHTML,/5\.71|71\.75|NaN|undefined/);
});
