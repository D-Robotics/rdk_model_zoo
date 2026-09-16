import type { BenchmarkRecord, MetricRecord, ModelVariant } from '../catalog/types';
import { metricDisplayLabel, canonicalMetricName } from '../catalog/metric-identity';
import { cell, metricCellText, table, wrapper } from './detail-utils';
import { renderVariantBenchmarkTable, type VariantBenchmarkTableOptions } from './variant-benchmark-table';

type Observation = { metric: MetricRecord; record: BenchmarkRecord };
type Context = VariantBenchmarkTableOptions['context'];

function rowLabel(text: string): HTMLTableCellElement {
  const result = cell(text, true);
  result.scope = 'row';
  return result;
}

function observations(variant: ModelVariant, kind: 'performance' | 'accuracy'): Observation[] {
  return variant.benchmarks.flatMap(record => (record[kind] ?? []).map(metric => ({metric, record})));
}

function conditions(entries: Observation[], context: Context): HTMLElement {
  const details = document.createElement('details');
  details.className = 'measurement-conditions';
  const summary = document.createElement('summary');
  summary.textContent = context.locale === 'zh' ? '测试条件' : 'Test conditions';
  details.append(summary);
  for (const value of new Set(entries.map(({metric,record}) => [record.display_name, record.environment.runtime,
    metric.scope, metric.concurrency === undefined ? undefined : `${metric.concurrency} threads`].filter(Boolean).join(' · ')))) {
    const p = document.createElement('p');
    p.textContent = value;
    details.append(p);
  }
  return details;
}

function values(entries: Observation[], context: Context): string {
  return [...new Set(entries.map(({metric}) => metricCellText(metric, context.locale).replace(/ratio$/, "")))].join(' / ') || '—';
}

function measurementSection(title: string, className: string, headers: string[]): {section: HTMLElement; body: HTMLTableSectionElement} {
  const section = document.createElement('section');
  section.className = className;
  const heading = document.createElement('h3');
  heading.textContent = title;
  const grid = table(title);
  grid.className = 'representative-matrix';
  const row = document.createElement('tr');
  row.append(...headers.map(name=>cell(name,true)));
  grid.tHead!.append(row);
  section.append(heading, wrapper(grid,title));
  return {section, body:grid.tBodies[0]!};
}

function runtimeOf({record, metric}: Observation): string {
  // Use explicit source labels, never infer a runtime from the timing value.
  const label = `${record.display_name} ${metric.scope ?? ''}`;
  if (/C\+\+|cpp/i.test(label)) return 'C++';
  if (/Python|HB_HBMRuntime/i.test(label)) return 'Python';
  return record.environment.runtime ?? record.display_name;
}

function performance(id: string, entries: Observation[], context: Context): HTMLElement {
  const zh = context.locale === 'zh';
  if (id === 'paraformer') {
    const {section,body} = measurementSection(zh?'阶段性能':'Pipeline performance','representative-performance',
      [zh?'阶段 / 指标':'Stage / metric','Python hbm_runtime','C++ UCP']);
    for (const metric of new Set(entries.map(e=>e.metric.metric))) {
      const relevant = entries.filter(e=>e.metric.metric===metric);
      const row = document.createElement('tr');
      const labels: Record<string,string> = zh ? {'encoder-latency':'Encoder 编码器','predictor-latency':'Predictor 预测器','cpu-cif-latency':'CPU CIF','decoder-latency':'Decoder 解码器','end-to-end-latency':'端到端耗时','rtf':'RTF 实时率'} : {'encoder-latency':'Encoder','predictor-latency':'Predictor','cpu-cif-latency':'CPU CIF','decoder-latency':'Decoder','end-to-end-latency':'End-to-end latency','rtf':'RTF'};
      row.append(rowLabel(labels[metric] ?? metricDisplayLabel(canonicalMetricName(metric),context.locale)),
        cell(values(relevant.filter(e=>!e.metric.scope?.includes('C++')),context)),
        cell(values(relevant.filter(e=>e.metric.scope?.includes('C++')),context)));
      body.append(row);
    }
    const note = document.createElement('p');
    note.textContent = zh ? '端到端数值来自原始记录，计时不含 WAV 前端处理；各阶段不相加推导总耗时。' : 'Reported end-to-end timing excludes the WAV frontend. Stage timings are not summed to derive a total.';
    section.append(note,conditions(entries,context));
    return section;
  }
  if (id === 'himloco') {
    const measured = entries.filter(e=>!/compiler estimate/i.test(`${e.record.display_name} ${e.metric.scope}`));
    const estimated = entries.filter(e=>!measured.includes(e));
    const {section,body} = measurementSection(zh?'运行时性能':'Runtime performance','representative-performance',
      [zh?'运行时':'Runtime','mean','p50','p95','min','max','FPS']);
    const select = document.createElement('select');
    select.setAttribute('aria-label',zh?'运行时':'Runtime');
    for(const runtime of ['all',...new Set(measured.map(runtimeOf))]) {
      const option = document.createElement('option');
      option.value = runtime;
      option.textContent = runtime === 'all' ? (zh?'全部运行时':'All runtimes') : runtime;
      select.append(option);
    }
    for(const runtime of new Set(measured.map(runtimeOf))) {
      const relevant = measured.filter(e=>runtimeOf(e)===runtime);
      const row = document.createElement('tr');
      row.dataset.runtime = runtime;
      row.append(rowLabel(runtime));
      for(const stat of ['mean','p50','p95','min','max','FPS']) row.append(cell(values(relevant.filter(e=>stat==='FPS'
        ? e.metric.unit==='fps' : e.metric.statistic===stat && e.metric.metric==='latency'),context)));
      body.append(row);
    }
    select.addEventListener('change',()=>{for(const row of [...body.rows]) row.hidden=select.value!=='all' && row.dataset.runtime!==select.value;});
    section.insertBefore(select,section.children[1]!);
    section.append(conditions(measured,context));
    if(estimated.length) {
      const estimate = document.createElement('details');
      estimate.className='representative-estimate';
      const summary=document.createElement('summary');
      summary.textContent=zh?'编译器估算（非运行时实测）':'Compiler estimate (not runtime measurement)';
      estimate.append(summary,document.createTextNode(values(estimated,context)),conditions(estimated,context));
      section.append(estimate);
    }
    return section;
  }
  const {section,body} = measurementSection(zh?'性能':'Performance','representative-performance',
    [zh?'输出 / 计时范围':'Output / timing scope',zh?'指标':'Metric',zh?'数值':'Value']);
  for(const {metric} of entries) {
    const row=document.createElement('tr');
    row.append(rowLabel(metric.scope ?? (zh?'未记录':'Not recorded')),cell(metricDisplayLabel(canonicalMetricName(metric.metric),context.locale)),cell(metricCellText(metric,context.locale)));
    body.append(row);
  }
  section.append(conditions(entries,context));
  return section;
}

function accuracy(entries: Observation[],context: Context): HTMLElement {
  const zh=context.locale==='zh';
  const stages = ['float', 'quantized', 'reported'].filter(stage => entries.some(e => stage === 'reported'
    ? !['float','quantized'].includes(e.metric.model_stage ?? '') : e.metric.model_stage === stage));
  const hasDataset = entries.some(e => e.metric.dataset);
  const stageNames: Record<string,string> = zh ? {float:'浮点',quantized:'量化',reported:'实测值'} : {float:'Float',quantized:'Quantized',reported:'Reported value'};
  const headers = [zh?'指标 / 输出':'Metric / output', ...(hasDataset ? [zh?'数据集':'Dataset'] : []), ...stages.map(stage=>stageNames[stage]!)];
  const {section,body}=measurementSection(zh?'精度':'Accuracy','representative-accuracy',headers);
  const groups=new Map<string,Observation[]>();
  for(const entry of entries) {
    const key=JSON.stringify([entry.metric.metric,entry.metric.dataset,entry.metric.scope,entry.metric.statistic,entry.metric.unit]);
    groups.set(key,[...groups.get(key)??[],entry]);
  }
  for(const group of groups.values()) {
    const metric=group[0]!.metric;
    const row=document.createElement('tr');
    const label=rowLabel(metricDisplayLabel(canonicalMetricName(metric.metric),context.locale));
    if(metric.scope) { const scope=document.createElement('small');scope.textContent=metric.scope;label.append(document.createElement('br'),scope); }
    if(metric.statistic) label.append(document.createTextNode(' · ' + metric.statistic));
    if(metric.unit === 'ratio') label.append(document.createTextNode(zh ? ' · 原始值' : ' · raw value'));
    row.append(label);
    if(hasDataset) row.append(cell(metric.dataset ?? '—'));
    for(const stage of stages) row.append(cell(values(group.filter(e=>stage==='reported'
      ? !['float','quantized'].includes(e.metric.model_stage ?? '') : e.metric.model_stage===stage),context)));
    body.append(row);
  }
  if(!entries.length) {const row=document.createElement('tr');const empty=cell(zh?'未记录精度数据':'No accuracy recorded');empty.colSpan=headers.length;row.append(empty);body.append(row);}
  return section;
}

/** Representative layouts retain the original source details and asset actions. */
export function renderRepresentativeDetails(id: string, options: VariantBenchmarkTableOptions): HTMLElement {
  const {context}=options;
  const zh=context.locale==='zh';
  const host=document.createElement('div');
  host.className='representative-layout';
  host.dataset.layout=id;
  const selected=options.variants.filter(v=>v.hardware===options.hardware && v.task===options.task).sort((a,b)=>a.id.localeCompare(b.id, undefined, {numeric:true}));
  const selector=document.createElement('label');
  selector.className='representative-selector';
  selector.textContent=zh?'具体模型':'Model configuration';
  const select=document.createElement('select');
  selector.append(select);
  if(selected.length>1)host.append(selector);
  for(const [index,variant] of selected.entries()) {
    const option=document.createElement('option');option.value=variant.id;option.textContent=variant.name;select.append(option);
    const panel=document.createElement('article');panel.className='representative-configuration';panel.dataset.variantId=variant.id;panel.hidden=index!==0;
    const specs=renderVariantBenchmarkTable({...options,variants:[variant]});
    // The specification table owns only configuration, input and downloads.
    // Keep its adjacent expandable source row intact for evidence and audit.
    specs.querySelector('.model-detail-benchmark-toolbar')?.remove();
    specs.querySelector('.model-detail-conditions-host')?.remove();
    const head=specs.querySelector('thead')!;head.replaceChildren();
    const header=document.createElement('tr');header.append(...(zh?['模型配置','输入','下载']:['Configuration','Input','Download']).map(t=>cell(t,true)));head.append(header);
    const row=specs.querySelector('.model-detail-spec-row')!;
    for(const child of [...row.children].slice(2,-1))child.remove();
    specs.querySelector('caption')!.textContent=zh?'配置与下载':'Configuration and downloads';
    const expanded=specs.querySelector<HTMLTableCellElement>('.model-detail-expanded-row > td');if(expanded)expanded.colSpan=3;
    if (id === 'paraformer') for (const link of specs.querySelectorAll<HTMLAnchorElement>('.model-detail-download-list a')) {
      const component = link.title.match(/encoder|predictor|decoder/i)?.[0];
      if (component) link.textContent = (zh ? '下载 ' : 'Download ') + component;
    }
    panel.append(specs,performance(id,observations(variant,'performance'),context),accuracy(observations(variant,'accuracy'),context));
    host.append(panel);
  }
  select.addEventListener('change',()=>{for(const panel of host.querySelectorAll<HTMLElement>('.representative-configuration'))panel.hidden=panel.dataset.variantId!==select.value;});
  return host;
}
