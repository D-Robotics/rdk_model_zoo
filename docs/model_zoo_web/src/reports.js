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

  if (selected) {
    document.getElementById('report-app').innerHTML = `${header}<main class="report-view"><div class="report-view-toolbar"><div><a href="${back}">← 返回模型详情</a><a href="${library}">报告库</a><h1>${esc(selected.name)}</h1><span>${esc(selected.hardware || selected.march || '')}</span></div><a class="report-download" href="${esc(selected.path)}" download>下载原始报告 ↓</a></div><iframe class="report-frame" src="${esc(selected.path)}" title="OE 转换报告" sandbox="allow-scripts allow-downloads"></iframe></main>`;
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
        document.getElementById('report-list').innerHTML = items.map(report => `<a class="report-row" href="reports.html?report=${encodeURIComponent(report.id)}${origin ? `&from=${encodeURIComponent(origin.id)}` : ''}"><div><h2>${esc(report.name)}</h2><code>${esc(report.sources?.[0] || report.path)}</code></div><div class="report-row-meta"><span>${esc(report.hardware || report.march || '—')}</span><span>${report.kind === 'performance' ? '性能报告' : '编译报告'}</span><span>${Number(report.bytes / 1e6).toFixed(1)} MB</span><span aria-hidden="true">↗</span></div></a>`).join('');
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
