(() => {
  'use strict';

  const esc = value => String(value ?? '').replace(/[&<>"']/g, character => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[character]));
  const reports = Array.isArray(window.OE_REPORTS) ? window.OE_REPORTS : [];
  const models = Array.isArray(window.MODEL_DATA?.models) ? window.MODEL_DATA.models : [];
  const params = new URLSearchParams(location.search);
  const scopedModel = models.find(model => model.id === params.get('model'));
  const origin = scopedModel || models.find(model => model.id === params.get('from'));
  const selected = reports.find(report => report.id === params.get('report'));
  const back = origin ? `index.html#model/${origin.id}` : 'index.html';
  const library = 'reports.html' + (origin ? `?from=${encodeURIComponent(origin.id)}` : '');
  const header = '<header class="header"><a href="index.html" class="brand"><img src="assets/rdk-brand/logo.png" alt="D-Robotics"><span>RDK <b>Model Zoo</b></span></a><nav><a href="https://github.com/D-Robotics/rdk_model_zoo" target="_blank" rel="noopener">GitHub ↗</a></nav><div class="language-switch" role="group" aria-label="Language / 语言"><button type="button" data-language="en" lang="en" aria-label="English">EN</button><button type="button" data-language="zh" lang="zh-CN" aria-label="简体中文">中文</button></div></header>';
  const footer = '<footer><div class="footer-inner">© 2026 D-Robotics / Model Zoo</div></footer>';

  const fmt = (value, digits = 2) => value == null || !Number.isFinite(Number(value)) ? '—' : Number(Number(value).toFixed(digits));
  const giga = value => value == null || !Number.isFinite(Number(value)) ? '—' : `${Number((value / 1e9).toFixed(2))} G`;
  const cosineClass = value => value == null ? 'is-unmeasured' : value >= 0.99 ? 'is-close' : value >= 0.95 ? 'is-mid' : 'is-weak';
  const chip = (label, value) => `<span class="rd-chip"><b>${esc(label)}</b>${esc(value)}</span>`;

  function strip(bars) {
    const peak = Math.max(...bars, 0.001);
    return `<span class="rd-strip" role="img" aria-label="逐区间利用率">${bars.map(value => `<i style="height:${Math.max(4, Math.round((value / peak) * 100))}%"></i>`).join('')}</span>`;
  }

  function overviewPanel(data, report) {
    const aggregate = data.aggregate || {};
    const quantization = data.quantization || {};
    const summary = quantization.summary || {};
    const weakest = summary.weakest_node;
    const profiler = data.board_measurement;
    const boardModel = models.find(model => (report.modelIds || []).includes(model.id));
    const perfRecords = boardModel?.benchmark?.performance || [];
    const catalogLatency = perfRecords
      .filter(record => record.metric === 'latency')
      .sort((a, b) => (a.concurrency || 1) - (b.concurrency || 1))[0];
    const catalogThroughput = perfRecords.find(record =>
      record.metric === 'throughput' && (record.concurrency || 1) === (catalogLatency?.concurrency || 1));
    const profilerBpu = profiler?.bpu_inference_ms;
    const boardCpu = profiler?.cpu_inference_ms;
    const boardStages = profiler?.stages || {};
    const generateTask = boardStages.generate_task_ms;
    const latencyBoard = profilerBpu ? profilerBpu.avg_time : catalogLatency?.value;
    const bpuScopedFps = profiler && profiler.average_latency_ms != null && boardCpu?.avg_time != null
      ? 1000 / (profiler.average_latency_ms - boardCpu.avg_time)
      : profiler?.fps ?? catalogThroughput?.value;
    const fpsBoard = bpuScopedFps ?? catalogThroughput?.value;
    const pairCard = (pairedLabel, fallbackLabel, estimateText, boardText, suffix = '') => boardText != null
      ? `<div class="rd-card"><span class="rd-card-label">${pairedLabel}</span><strong>${estimateText} / ${boardText}${suffix}</strong></div>`
      : `<div class="rd-card"><span class="rd-card-label">${fallbackLabel}</span><strong>${estimateText}${suffix}</strong></div>`;
    const cards = [
      pairCard('估算 / 实测延迟', '估算延迟', aggregate.latency_ms != null ? fmt(aggregate.latency_ms) : '—', latencyBoard != null ? fmt(latencyBoard) : null, ' ms'),
      pairCard('估算 / 实测 FPS', '估算 FPS', fmt(aggregate.fps_derived), fpsBoard != null ? fmt(fpsBoard) : null),
      `<div class="rd-card"><span>估算 DDR</span><strong>${aggregate.ddr_mb_per_frame != null ? `${fmt(aggregate.ddr_mb_per_frame, 1)} MB` : '—'}</strong></div>`,
      `<div class="rd-card"><span>BPU OPs</span><strong>${giga(aggregate.bpu_ops_per_frame)}</strong></div>`,
      `<div class="rd-card"><span>量化节点</span><strong>${summary.total_nodes != null ? `${summary.total_nodes} 层` : '—'}</strong></div>`,
      (() => {
        const weakestList = summary.weakest_nodes || (weakest ? [weakest] : []);
        const percentiles = summary.cosine_percentiles || {};
        const shortName = name => name.split('/').slice(-2).join('/');
        const rows = weakestList.slice(0, 3).map(node =>
          `<span class="rd-weak-row"><b>${fmt(node.cosine, 4)}</b><code title="${esc(node.name)}">${esc(shortName(node.name))}</code></span>`).join('');
        const stats = percentiles.p10 != null ? `<small>P10 ${fmt(percentiles.p10, 4)} · 中位 ${fmt(percentiles.median, 4)}</small>` : '';
        return `<div class="rd-card rd-card-weak"><span>最弱相似度 TOP3</span><div class="rd-weak-rows">${rows || '—'}</div>${stats}</div>`;
      })(),
    ];
    const subgraphs = data.subgraphs || [];
    const estimatePanel = subgraphs.length
      ? `<div class="rd-duo-panel"><h3>编译器估算</h3>${subgraphs.map(item => `<div class="rd-duo-subgraph"><div class="rd-duo-info"><code>${esc(item.name)}</code><div class="rd-duo-stats"><span>${fmt(item.perf?.latency_ms)} ms</span><span>${fmt(item.perf?.fps)} FPS</span><span>${fmt(item.perf?.ddr_mb_per_frame, 1)} MB</span></div></div>${item.intervals?.utilization?.length ? strip(item.intervals.utilization) : ''}</div>`).join('')}${subgraphs.length > 1 ? `<div class="rd-duo-row is-total"><span>合计</span><b>${fmt(aggregate.latency_ms)} ms</b><small>${fmt(aggregate.fps_derived)} FPS</small></div>` : ''}</div>`
      : '';
    const boardRow = (name, value, note, strong = false) => `<div class="rd-duo-row${strong ? ' is-total' : ''}"><span>${name}</span><b>${value}</b>${note ? `<small>${note}</small>` : ''}</div>`;
    const boardPanel = profiler
      ? `<div class="rd-duo-panel"><h3>板端实测</h3>${profilerBpu ? boardRow(generateTask ? 'BPU 核执行' : 'BPU 推理', `${fmt(profilerBpu.avg_time)} ms`, fmt(bpuScopedFps) != null ? `${fmt(bpuScopedFps)} FPS` : '') : ''}${generateTask ? boardRow('任务生成', `${fmt(generateTask.avg_time)} ms`) : ''}${boardCpu && boardCpu.avg_time > 0 ? boardRow('CPU 反量化', `${fmt(boardCpu.avg_time)} ms`, profiler.cpu_op_count ? `${profiler.cpu_op_count} 个算子` : '') : ''}${boardRow('合计', profiler.average_latency_ms != null ? `${fmt(profiler.average_latency_ms)} ms` : '—', profiler.fps != null ? `${fmt(profiler.fps)} FPS` : '', true)}</div>`
      : '';
    const decompSection = estimatePanel || boardPanel
      ? `<div class="rd-duo">${estimatePanel}${boardPanel}</div>`
      : '';
    const inputs = (data.io?.inputs || []).map(item => chip(item.name, `${item.shape.join(' × ')} · ${item.dtype}`)).join('') || '<span class="rd-muted">—</span>';
    const outputs = (data.io?.outputs || []).map(item => chip(item.name, `${item.shape.join(' × ')} · ${item.dtype}`)).join('') || '<span class="rd-muted">—</span>';
    const toolchain = data.toolchain || {};
    const toolchainChips = [
      ['hbdk', toolchain.hbdk], ['hb_mapper', toolchain.hb_mapper],
      ['horizon_nn', toolchain.horizon_nn], ['runtime', toolchain.runtime],
      ['march', data.target?.march],
    ].filter(([, value]) => value).map(([label, value]) => chip(label, value)).join('');
    return `<section class="rd-panel" data-panel="overview" role="tabpanel">
      <div class="rd-cards">${cards.join('')}</div>
      ${decompSection}
      <div class="rd-section"><h2>模型输入</h2><div class="rd-chips">${inputs}</div></div>
      <div class="rd-section"><h2>模型输出</h2><div class="rd-chips">${outputs}</div></div>
      <div class="rd-section"><h2>工具链</h2><div class="rd-chips">${toolchainChips}</div></div>
    </section>`;
  }

  function quantizationPanel(data) {
    const quantization = data.quantization || {};
    const nodes = quantization.nodes || [];
    const outputs = quantization.outputs || [];
    const summary = quantization.summary || {};
    const units = summary.unit_counts || {};
    const types = [...new Set(nodes.map(node => node.type))].sort();
    return `<section class="rd-panel" data-panel="quantization" role="tabpanel" hidden>
      ${nodes.length ? `<div class="rd-filters"><input type="search" id="rd-node-search" aria-label="搜索节点" placeholder="搜索节点"><select id="rd-node-unit" aria-label="执行单元"><option value="">执行单元：全部</option>${Object.keys(units).map(unit => `<option value="${esc(unit)}">${esc(unit)}</option>`).join('')}</select><select id="rd-node-type" aria-label="算子类型"><option value="">算子：全部</option>${types.map(type => `<option value="${esc(type)}">${esc(type)}</option>`).join('')}</select></div>
      <div class="rd-table-wrap"><table class="rd-table"><thead><tr><th>节点</th><th>执行单元</th><th>子图</th><th>算子类型</th><th>余弦相似度</th><th>阈值</th><th>数据类型</th></tr></thead><tbody id="rd-node-body"></tbody></table></div>
      ${outputs.length ? `<div class="rd-section"><h2>输出级对比</h2><div class="rd-table-wrap"><table class="rd-table rd-outputs"><thead><tr><th>输出</th><th>余弦相似度</th><th>L1 距离</th><th>L2 距离</th><th>切比雪夫距离</th></tr></thead><tbody>${outputs.map(item => `<tr><td><code>${esc(item.name)}</code></td><td class="rd-cosine ${cosineClass(item.cosine)}">${fmt(item.cosine, 6)}</td><td>${fmt(item.l1_distance, 4)}</td><td>${fmt(item.l2_distance, 6)}</td><td>${fmt(item.chebyshev_distance, 4)}</td></tr>`).join('')}</tbody></table></div></div>` : ''}`
      : '<div class="report-empty"><h2>暂无逐层数据</h2></div>'}
    </section>`;
  }

  function nodeRows(nodes) {
    return nodes.map(node => `<tr><td><code>${esc(node.name)}</code></td><td>${esc(node.on)}</td><td>${node.subgraph}</td><td>${esc(node.type)}</td><td class="rd-cosine ${cosineClass(node.cosine)}">${node.cosine == null ? '未测量' : fmt(node.cosine, 6)}</td><td>${fmt(node.threshold, 4)}</td><td>${esc(node.dtype)}</td></tr>`).join('');
  }

  function renderNative(root, data, report) {
    root.innerHTML = `
      <div class="rd-tabs" role="tablist"><button type="button" class="rd-tab is-active" data-tab="overview" role="tab" aria-selected="true">概览</button><button type="button" class="rd-tab" data-tab="quantization" role="tab" aria-selected="false">逐层量化</button><button type="button" class="rd-tab" data-tab="raw" role="tab" aria-selected="false">原始数据</button></div>
      ${overviewPanel(data, report)}${quantizationPanel(data)}
      <section class="rd-panel" data-panel="raw" role="tabpanel" hidden><details class="rd-raw"><summary>查看完整 JSON</summary><pre>${esc(JSON.stringify(data, null, 2))}</pre></details></section>`;

    root.querySelectorAll('.rd-tab').forEach(tab => tab.addEventListener('click', () => {
      root.querySelectorAll('.rd-tab').forEach(item => {
        const active = item === tab;
        item.classList.toggle('is-active', active);
        item.setAttribute('aria-selected', String(active));
      });
      root.querySelectorAll('.rd-panel').forEach(panel => { panel.hidden = panel.dataset.panel !== tab.dataset.tab; });
    }));

    const nodes = data.quantization?.nodes || [];
    const search = root.querySelector('#rd-node-search');
    const unit = root.querySelector('#rd-node-unit');
    const type = root.querySelector('#rd-node-type');
    const body = root.querySelector('#rd-node-body');
    if (body) {
      const apply = () => {
        const query = (search?.value || '').trim().toLowerCase();
        body.innerHTML = nodeRows(nodes.filter(node =>
          (!query || node.name.toLowerCase().includes(query) || node.type.toLowerCase().includes(query))
          && (!unit?.value || node.on === unit.value)
          && (!type?.value || node.type === type.value)));
        window.HubI18n.apply();
      };
      search?.addEventListener('input', apply);
      unit?.addEventListener('change', apply);
      type?.addEventListener('change', apply);
      apply();
    }
  }

  if (selected) {
    document.getElementById('report-app').innerHTML = `${header}<main class="report-view report-view-native"><div class="report-view-toolbar"><div class="report-view-toolbar-inner"><div><a href="${back}">← 返回模型详情</a><a href="${library}">报告库</a><h1>${esc(selected.name)}${selected.march ? ` · ${esc(selected.march)}` : ''}</h1></div></div></div><div class="rd-root" id="rd-root"><div class="rd-loading">加载中…</div></div></main>${footer}`;
    const root = document.getElementById('rd-root');
    const mountIframe = () => {
      root.outerHTML = `<iframe class="report-frame" src="${esc(selected.path)}" title="OE 转换报告" sandbox="allow-scripts allow-downloads"></iframe>`;
    };
    if (selected.dataUrl) {
      fetch(selected.dataUrl)
        .then(response => {
          if (!response.ok) throw new Error(`HTTP ${response.status} for ${selected.dataUrl}`);
          return response.json();
        })
        .then(data => renderNative(root, data, selected))
        .catch(error => {
          console.warn(`OE data unavailable, falling back to the original report: ${error.message}`);
          if (selected.path) mountIframe();
          else root.innerHTML = '<div class="report-empty"><h2>报告数据缺失</h2></div>';
        });
    } else if (selected.path) {
      mountIframe();
    } else {
      root.innerHTML = '<div class="report-empty"><h2>报告数据缺失</h2></div>';
    }
  } else {
    document.getElementById('report-app').innerHTML = `${header}<main class="report-library"><a class="report-back" href="${back}">← 返回模型目录</a><div class="report-heading"><div><h1>OE 转换报告</h1>${scopedModel ? `<p>${esc(scopedModel.name)}</p>` : ''}</div><label class="report-search"><input type="search" id="report-search" aria-label="搜索报告" placeholder="搜索报告"></label></div><div id="report-count" aria-live="polite"></div><div id="report-list"></div></main>${footer}`;

    function render() {
      const query = document.getElementById('report-search').value.trim().toLowerCase();
      const available = scopedModel ? reports.filter(report => report.modelIds.includes(scopedModel.id)) : reports;
      const items = available.filter(report => [report.name, report.hardware, report.march, ...(report.sources || [])].join(' ').toLowerCase().includes(query));
      document.getElementById('report-count').textContent = `${items.length} 个报告`;
      if (!reports.length) {
        document.getElementById('report-list').innerHTML = '<div class="report-empty"><h2>暂未发布任何 OE 报告</h2><p>当前尚无可浏览的转换报告。</p></div>';
      } else if (!items.length) {
        document.getElementById('report-list').innerHTML = `<div class="report-empty"><h2>${scopedModel ? '暂无匹配的 OE 报告' : '没有匹配的报告'}</h2><a href="${library}">查看全部报告 →</a></div>`;
      } else {
        document.getElementById('report-list').innerHTML = items.map(report => `<a class="report-row" href="reports.html?report=${encodeURIComponent(report.id)}${origin ? `&from=${encodeURIComponent(origin.id)}` : ''}"><div><h2>${esc(report.name)}</h2><code>${esc(report.sources?.[0] || report.path)}</code></div></a>`).join('');
      }
      window.HubI18n.apply();
    }

    document.getElementById('report-search').addEventListener('input', render);
    render();
  }

  document.querySelector('.language-switch').addEventListener('click', event => {
    const button = event.target.closest('[data-language]');
    if (button) window.HubI18n.setLocale(button.dataset.language);
  });
  window.HubI18n.apply();
})();
