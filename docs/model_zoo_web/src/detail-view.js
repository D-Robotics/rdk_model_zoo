(() => {
  'use strict';
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const icon = name => `<span class="mz-icon" style="--icon:url('assets/icons/${name}.svg')" aria-hidden="true"></span>`;
  const external = (url, text, cls = '') => `<a class="${cls}" href="${esc(url)}" target="_blank" rel="noopener">${text}${icon('external-link')}</a>`;
  const amount = (record, accuracy = false) => {
    if (!record) return '—';
    const value = accuracy && record.unit === 'ratio' ? record.value * 100 : record.value;
    return `${record.qualifier === 'lower-bound' ? '≥ ' : record.qualifier === 'approximate' ? '≈ ' : ''}${Number(value.toFixed(2))}`;
  };
  const metricName = metric => ({
    latency: '推理延迟', throughput: '吞吐量', post_process_latency: '后处理延迟',
    'bbox-all-map-50-95': '检测 mAP50–95', 'mask-all-map-50-95': '分割 mAP50–95',
    'keypoints-all-map-50-95': '关键点 AP50–95', 'top-1': 'Top-1',
    'bbox-all-map-50-95-retention': '精度保留率',
  }[metric] || metric);
  const threadLabel = threads => threads === 1 ? '单线程' : `${threads} 线程`;
  const formats = m => [...new Set(m.assets.map(asset => asset.format.toUpperCase()))].join(' + ');
  const hardwareOf = (model, data) => model.benchmark?.environment?.hardware || data.release.compatibility.hardware;
  const dropdown = (id, label, options, selected, disabled = false) => {
    const current = options.find(option => String(option.value) === String(selected)) || options[0];
    return `<div class="mz-select"><button type="button" class="mz-select-trigger" id="${id}" data-value="${esc(current?.value || '')}" role="combobox" aria-haspopup="listbox" aria-expanded="false" aria-controls="${id}-options" aria-label="${label}" ${disabled ? 'disabled' : ''}><span class="mz-select-value">${esc(current?.label || '—')}</span>${icon('chevron-down')}</button><ul id="${id}-options" class="mz-select-options" role="listbox" aria-label="${label}" hidden>${options.map((option, index) => `<li id="${id}-option-${index}" role="option" data-value="${esc(option.value)}" data-label="${esc(option.label)}" aria-selected="${String(option.value) === String(current?.value)}"><span>${esc(option.label)}</span>${icon('check')}</li>`).join('')}</ul></div>`;
  };
  const assetRole = asset => asset.role || '部署模型';
  function propertyRows(model) {
    const p = window.MODEL_PROPERTIES?.[model.id] || {};
    const shape = value => Array.isArray(value) ? value.join(' × ') : String(value).replace(/\s+x\s+/g, ' × ');
    const row = (label, value, title = '') => `<dt>${label}</dt><dd${title ? ` title="${esc(title)}"` : ''}>${esc(value)}</dd>`;
    const rows = [];
    if (p.parameterCount) rows.push(row('参数量', `${Number((p.parameterCount / 1e6).toFixed(3))} M`, String(p.parameterCount)));
    if (p.parameterCountMillions) rows.push(row('参数量', `${p.parameterCountMillions} M`));
    if (p.gflops) rows.push(row('浮点计算量', `${Number(p.gflops.toFixed(2))} GFLOPs @ ${p.inputShape.slice(-2).join(' × ')}`));
    if (p.inputShape) rows.push(row('源模型输入', shape(p.inputShape)));
    else if (p.inputResolution) rows.push(row('输入尺寸', shape(p.inputResolution)));
    if (p.inputLayout) rows.push(row('输入布局', p.inputLayout));
    if (p.predictionShape) rows.push(row('预测输出', shape(p.predictionShape)));
    if (p.keypointShape) rows.push(row('关键点数', `${p.keypointShape[0]}`));
    if (p.maskChannels) rows.push(row('掩码通道数', String(p.maskChannels)));
    if (p.maskPrototypeShape) rows.push(row('掩码原型', shape(p.maskPrototypeShape)));
    if (p.decoderInputShape) rows.push(row('解码器输入', shape(p.decoderInputShape)));
    for (const output of p.componentOutputs || []) rows.push(row(output.name, shape(output.shape)));
    const labels = { image: '图像输入', image_feature: '图像特征', texts: '文本输入', text_features: '文本特征' };
    for (const input of p.interfaces || []) rows.push(row(labels[input.name] || input.name, `${shape(input.shape)} · ${input.dtype}`));
    if (!rows.length) rows.push(row('输入尺寸', model.shape));
    return rows.join('');
  }

  function render({ model: m, models, data }) {
    const b = m.benchmark;
    const hardware = hardwareOf(m, data);
    const variants = models.filter(other => other.sample === m.sample && other.task === m.task);
    const hardwareOptions = [...new Set(variants.map(other => hardwareOf(other, data)))];
    const performance = b?.performance || [];
    const conditions = [...new Set(performance.filter(p => ['latency', 'throughput'].includes(p.metric)).map(p => p.concurrency || 1))].sort((a, b) => a - b);
    const related = models.filter(other => other.id !== m.id && (other.sample === m.sample || other.task === m.task)).slice(0, 3);
    return `<nav class="mz-breadcrumb" aria-label="模型导航"><a class="back-link" href="#">Model Zoo</a><span aria-hidden="true">/</span><span>${m.name}</span></nav>
      <section class="mz-overview"><div class="mz-introduction"><span class="mz-task">${m.task}</span><h1>${m.name}</h1><p>${m.description}</p><div class="mz-primary-actions"><a class="button" href="#model/${m.id}/downloads">${icon('download')}获取模型</a>${external(m.source, '模型仓库', 'mz-text-link')}</div></div><figure class="mz-demo"><img src="${esc(m.coverImage)}" alt="${esc(m.name + ' · ' + m.coverLabel)}"></figure></section>
      <section class="mz-benchmark" aria-label="性能与精度"><div class="mz-benchmark-title"><h2>性能与精度</h2></div><div class="mz-configuration"><div><span>目标硬件</span>${dropdown('benchmark-hardware', '目标硬件', hardwareOptions.map(value => ({ value, label: value })), hardware)}</div><div><span>模型变体</span>${dropdown('benchmark-variant', '模型变体', variants.filter(other => hardwareOf(other, data) === hardware).map(other => ({ value: other.id, label: other.name })), m.id)}</div><div><span>测试条件</span>${dropdown('benchmark-condition', '测试条件', conditions.map(value => ({ value: String(value), label: threadLabel(value) })), String(conditions[0] || ''), !conditions.length)}</div></div>
      ${b ? `<div class="mz-benchmark-body"><div class="mz-performance"><div class="mz-performance-grid" id="detail-performance" aria-live="polite"></div></div></div>` : `<div class="mz-benchmark-empty"><div><h3>暂无性能汇总</h3></div>${external(m.source + '/evaluator', '查看评测说明')}</div>`}<div class="mz-report-actions"><span class="mz-report-placeholder" aria-disabled="true">OE 转换报告</span></div></section>
      <div class="mz-information"><div class="mz-main-column"><section class="mz-section"><h2>模型参数</h2><dl class="mz-specifications">${propertyRows(m)}</dl></section>
      <section class="mz-section" id="downloads"><div class="mz-section-heading"><h2>模型文件</h2><span>${m.assets.length} 个文件</span></div><div class="mz-files">${m.assets.map(a => `<div class="mz-file"><div class="mz-file-copy"><div class="mz-file-title"><h3>${assetRole(a)}</h3><span>${esc(a.format.toUpperCase())}</span></div><code>${esc(a.filename)}</code>${a.sha256 ? `<small>SHA-256: ${esc(a.sha256)}</small>` : ''}</div>${external(a.url, icon('download') + '<span>下载</span>', 'mz-download')}</div>`).join('')}</div></section>
      </div>
      <aside class="mz-side-column"><section><h2>目标平台</h2><div class="mz-supported-platform"><span class="mz-platform-mark">${icon('cpu')}</span><div><strong>${esc(hardware)}</strong></div></div></section><section><h2>开发资源</h2><div class="mz-resource-links">${external(m.source, '模型仓库')}${external(m.source + '/runtime/python', '运行文档')}${external(m.source + '/conversion', '模型转换')}</div></section><section><h2>模型许可</h2>${external(m.source, '查看许可说明')}</section></aside></div>
      ${related.length ? `<section class="mz-related"><div class="mz-section-heading"><h2>相关模型</h2><a href="#">查看全部模型</a></div><div class="mz-related-grid">${related.map(other => `<a href="#model/${other.id}" class="mz-related-model"><img src="${esc(other.coverImage)}" alt="${esc(other.name)}" loading="lazy"><div><h3>${other.name}</h3><span>${other.task}</span></div></a>`).join('')}</div></section>` : ''}
      `;
  }

  let activeController = null;
  function destroy() {
    activeController?.abort();
    activeController = null;
  }

  function bind(root, m, models, data) {
    destroy();
    activeController = new AbortController();
    const signal = activeController.signal;
    const listen = (node, event, handler) => node?.addEventListener(event, handler, { signal });
    const b = m.benchmark;
    const refresh = () => window.HubI18n?.apply();
    let openSelect = null;
    function closeSelect(restoreFocus = false) {
      if (!openSelect) return;
      const { trigger, list } = openSelect;
      list.hidden = true;
      trigger.setAttribute('aria-expanded', 'false');
      trigger.removeAttribute('aria-activedescendant');
      if (restoreFocus) trigger.focus();
      openSelect = null;
    }
    for (const container of root.querySelectorAll('.mz-select')) {
      const trigger = container.querySelector('.mz-select-trigger');
      const list = container.querySelector('.mz-select-options');
      const options = [...list.querySelectorAll('[role="option"]')];
      let active = 0;
      function activate(index) {
        active = Math.max(0, Math.min(index, options.length - 1));
        options.forEach((option, i) => { option.dataset.active = String(i === active); });
        const option = options[active];
        if (!option) return;
        trigger.setAttribute('aria-activedescendant', option.id);
        if (option.offsetTop < list.scrollTop) list.scrollTop = option.offsetTop;
        else if (option.offsetTop + option.offsetHeight > list.scrollTop + list.clientHeight) {
          list.scrollTop = option.offsetTop + option.offsetHeight - list.clientHeight;
        }
      }
      function open() {
        if (trigger.disabled || !options.length) return;
        closeSelect();
        list.hidden = false;
        trigger.setAttribute('aria-expanded', 'true');
        const rect = trigger.getBoundingClientRect();
        const height = Math.min(list.scrollHeight, 240);
        container.classList.toggle('is-above', innerHeight - rect.bottom < height + 12 && rect.top > height + 12);
        openSelect = { trigger, list, container };
        activate(options.findIndex(option => option.dataset.value === trigger.dataset.value));
        trigger.focus();
      }
      function choose(option) {
        if (!option) return;
        const changed = trigger.dataset.value !== option.dataset.value;
        trigger.dataset.value = option.dataset.value;
        trigger.querySelector('.mz-select-value').textContent = option.dataset.label;
        options.forEach(item => item.setAttribute('aria-selected', String(item === option)));
        closeSelect(true);
        if (changed) trigger.dispatchEvent(new Event('change', { bubbles: true }));
        refresh();
      }
      listen(trigger, 'click', () => openSelect?.trigger === trigger ? closeSelect() : open());
      listen(trigger, 'keydown', event => {
        if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
          event.preventDefault();
          const wasOpen = openSelect?.trigger === trigger;
          if (!wasOpen) open();
          else if (event.key === 'ArrowDown') activate((active + 1) % options.length);
          else if (event.key === 'ArrowUp') activate((active - 1 + options.length) % options.length);
          if (event.key === 'Home') activate(0);
          if (event.key === 'End') activate(options.length - 1);
        } else if ((event.key === 'Enter' || event.key === ' ') && openSelect?.trigger === trigger) {
          event.preventDefault();
          choose(options[active]);
        } else if (event.key === 'Escape') {
          event.preventDefault();
          closeSelect(true);
        } else if (event.key === 'Tab') closeSelect();
      });
      listen(list, 'click', event => choose(event.target.closest('[role="option"]')));
    }
    listen(document, 'pointerdown', event => {
      if (openSelect && !openSelect.container.contains(event.target)) closeSelect();
    });
    let openPopover = null;
    const closePopover = focus => { openPopover?.classList.remove('is-open'); openPopover?.style.removeProperty('--clamp'); openPopover?.parentElement?.setAttribute('aria-expanded', 'false'); if (focus) openPopover?.parentElement?.focus?.(); openPopover = null; };
    listen(root, 'click', event => {
      const button = event.target.closest('.mz-accuracy-info');
      if (!button) return;
      const popover = button.querySelector('.mz-accuracy-popover');
      if (openPopover === popover) return closePopover(true);
      closePopover();
      openPopover = popover;
      popover.classList.add('is-open');
      const margin = 4;
      const left = popover.getBoundingClientRect().left;
      const overflowLeft = Math.max(0, margin - left);
      const overflowRight = Math.max(0, left + popover.offsetWidth - (document.documentElement.clientWidth - margin));
      const shift = overflowLeft ? overflowLeft : -overflowRight;
      if (shift) popover.style.setProperty('--clamp', `${shift}px`);
      button.setAttribute('aria-expanded', 'true');
    });
    listen(document, 'pointerdown', event => { if (openPopover && !openPopover.parentElement.contains(event.target)) closePopover(); });
    listen(document, 'keydown', event => { if (event.key === 'Escape' && openPopover) closePopover(); });

    function performance() {
      if (!b) return;
      const records = b.performance || [];
      const threads = Number(root.querySelector('#benchmark-condition')?.dataset.value || 1);
      const pick = metric => records.find(p => p.metric === metric && (p.concurrency || 1) === threads);
      const post = records.find(p => p.metric === 'post_process_latency');
      const delay = pick('latency'), speed = pick('throughput');
      const value = (record, unit) => record ? `${amount(record)}<small>${esc(unit)}</small>` : '—';
      const staged = [...new Set((b.accuracy || []).filter(a => a.model_stage).map(a => a.metric))];
      const accuracyTiles = staged.map(metric => {
        const find = stage => (b.accuracy || []).find(a => a.metric === metric && a.model_stage === stage);
        const reference = find('float'), quantized = find('quantized');
        const tileName = metric === 'top-1' ? 'Top-1' : staged.length > 1 ? metric.replace(/-all-map-50-95$/, ' mAP50-95') : 'mAP50-95';
        const plain = record => record ? `${amount(record, true)}%` : '—';
        const scale = record => record.unit === 'ratio' ? record.value * 100 : record.value;
        const delta = reference && quantized ? Number((scale(quantized) - scale(reference)).toFixed(2)) : null;
        const big = record => record ? `${amount(record, true)}<small>%</small>` : '—';
        return `<div><span class="mz-accuracy-label">${esc(tileName)}<button type="button" class="mz-accuracy-info" aria-label="${esc(tileName)}：查看浮点参考" aria-haspopup="dialog">${icon('circle-help')}<span class="mz-accuracy-popover" role="dialog" aria-label="${esc(tileName)}对照">浮点参考 <b>${plain(reference)}</b> · 差距 ${delta === null ? '—' : `<b>${delta > 0 ? '+' : delta < 0 ? '−' : ''}${Math.abs(delta)}%</b>`}</span></button></span><strong>${big(quantized || reference)}</strong></div>`;
      });
      root.querySelector('#detail-performance').innerHTML = `<div><span>推理延迟</span><strong>${value(delay, delay?.unit || 'ms')}</strong></div>${post ? `<div><span>后处理延迟</span><strong>${value(post, post.unit)}</strong></div>` : ''}<div><span>吞吐量</span><strong>${value(speed, speed?.unit || 'fps')}</strong></div>${accuracyTiles.join('')}`;
      refresh();
    }
    listen(root.querySelector('#benchmark-condition'), 'change', performance);
    listen(root.querySelector('#benchmark-variant'), 'change', event => {
      if (event.target.dataset.value !== m.id) location.hash = '#model/' + event.target.dataset.value;
    });
    listen(root.querySelector('#benchmark-hardware'), 'change', event => {
      const target = models.find(other => other.sample === m.sample && other.task === m.task && hardwareOf(other, data) === event.target.dataset.value);
      if (target && target.id !== m.id) location.hash = '#model/' + target.id;
    });
    performance();
  }
  window.ModelDetail = { render, bind, destroy };
})();
