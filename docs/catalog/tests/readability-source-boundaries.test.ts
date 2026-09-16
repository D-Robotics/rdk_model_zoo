import { expect, it } from 'vitest';
import { uniformColumnConditions, summarizeConditions, renderTableConditions } from '../src/ui/table-conditions';
import { benchmarkFixture } from './fixtures/catalog';

it('does not promote a timing condition across observations where it is unknown', () => {
  const record = benchmarkFixture();
  const known = {record, metric: {metric:'latency',value:1,unit:'ms',scope:'BPU task'}};
  const unknown = {record, metric: {metric:'latency',value:2,unit:'ms'}};
  expect(uniformColumnConditions(new Map([['latency', [[known], [unknown]]]])).get('latency')?.scope).toBeUndefined();
});
it('does not infer a runtime from another board with the same relative source path', () => {
  const record = benchmarkFixture({environment:{hardware:'RDK X5'}});
  const sibling = benchmarkFixture({environment:{hardware:'RDK S100',runtime:'hrt_model_exec'}});
  const summary = summarizeConditions({records:[record],siblingRecords:[sibling],columns:new Map(),columnLabels:new Map(),threadCounts:[1],locale:'en'});
  expect(summary.environment.some(fact=>fact.value.includes('hrt_model_exec'))).toBe(false);
});
it('preserves an external evidence repository in the shared conditions source link', () => {
  const record = benchmarkFixture();
  record.source.repository_url = 'https://github.com/example/evidence';
  const summary = summarizeConditions({records:[record],siblingRecords:[],sourceRef:"wrong-release",columns:new Map(),columnLabels:new Map(),threadCounts:[1],locale:'en'});
  const section = renderTableConditions(summary,{locale:'en',repositoryUrl:'https://github.com/D-Robotics/rdk_model_zoo',releaseTag:'x5-v1.1.2'});
  expect(section.querySelector('a')?.href).toContain('https://github.com/example/evidence/blob/' + record.source.ref + '/');
});
