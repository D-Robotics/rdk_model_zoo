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
    latency: 'Runtime 延迟', throughput: '吞吐量', pre_process_latency: '前处理延迟',
    post_process_latency: '后处理延迟', end_to_end_latency: '端到端延迟',
    'bbox-all-map-50-95': '检测 mAP50–95', 'mask-all-map-50-95': '分割 mAP50–95',
    'keypoints-all-map-50-95': '关键点 AP50–95', 'top-1': 'Top-1',
    'bbox-all-map-50-95-retention': '精度保留率',
  }[metric] || metric);
  const threadLabel = threads => threads === 1 ? 'Runtime 单路' : `Runtime ${threads} 路并发`;
  const formats = m => [...new Set(m.assets.map(asset => asset.format.toUpperCase()))].join(' + ');
  const hardwareOf = (model, data) => model.benchmark?.environment?.hardware || data.release.compatibility.hardware;
  // Preferred hardware display order: strongest S-series first, X5 last.
  const platformDisplayOrder = hardware => ({
    'RDK S600': 0, 'RDK S100P': 1, 'RDK S100': 2, 'RDK X5': 3,
  }[String(hardware)] ?? 99);
  const byPlatformOrder = (a, b) => platformDisplayOrder(a) - platformDisplayOrder(b) || String(a).localeCompare(String(b));
  const sizePreference = ['n', 's', 'm', 'l', 'x'];
  const sizeRank = size => { const index = sizePreference.indexOf(String(size || '').toLowerCase()); return index < 0 ? sizePreference.length : index; };
  const relatedGroupId = model => model.family || model.catalogId || `${model.name}/${model.taskId || model.task}`;
  const byteSize = value => {
    const bytes = Number(value);
    if (!Number.isFinite(bytes) || bytes <= 0) return null;
    const units = ['B', 'KiB', 'MiB', 'GiB'];
    const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
    return `${Number((bytes / (1024 ** exponent)).toFixed(2))} ${units[exponent]}`;
  };
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
    if (p.gflops) {
      const resolution = Array.isArray(p.inputShape) ? ` @ ${p.inputShape.slice(-2).join(' × ')}` : '';
      rows.push(row('浮点计算量', `${Number(Number(p.gflops).toFixed(2))} GFLOPs${resolution}`));
    }
    if (p.sourceInput) {
      const source = p.sourceInput;
      rows.push(row('源模型输入', [source.format?.toUpperCase(), source.dtype?.toUpperCase(), source.layout, shape(source.shape)].filter(Boolean).join(' · ')));
    } else if (p.inputShape) rows.push(row('源模型输入', shape(p.inputShape)));
    else if (p.inputResolution) rows.push(row('输入尺寸', shape(p.inputResolution)));
    if (p.runtimeInput) rows.push(row('Runtime 输入', [p.runtimeInput.format?.toUpperCase(), shape(p.runtimeInput.resolution)].filter(Boolean).join(' · ')));
    else if (p.inputLayout) rows.push(row('输入布局', p.inputLayout));
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
    const groupId = m.catalogId || `${m.name}/${m.taskId || m.task}`;
    const variants = models.filter(other => (other.catalogId || `${other.name}/${other.taskId || other.task}`) === groupId);
    const hardwareOptions = [...new Set(variants.map(other => hardwareOf(other, data)))].sort(byPlatformOrder);
    const performance = b?.performance || [];
    const conditions = [...new Set(performance.filter(p => ['latency', 'throughput'].includes(p.metric)).map(p => p.concurrency || 1))].sort((a, b) => a - b);
    const relatedCandidates = models.filter(other => relatedGroupId(other) !== relatedGroupId(m)
      && other.task === m.task);
    const relatedByFamily = new Map();
    for (const other of relatedCandidates) {
      const key = relatedGroupId(other);
      const current = relatedByFamily.get(key);
      if (!current || sizeRank(other.modelSize) < sizeRank(current.modelSize)
        || (sizeRank(other.modelSize) === sizeRank(current.modelSize)
          && platformDisplayOrder(hardwareOf(other, data)) < platformDisplayOrder(hardwareOf(current, data)))) {
        relatedByFamily.set(key, other);
      }
    }
    const related = [...relatedByFamily.values()].slice(0, 3);
    const downloadModels = variants.filter(other => other.assets?.length);
    const selectedDownloadModel = downloadModels.find(other => other.id === m.id) || downloadModels[0];
    const selectedPlatform = selectedDownloadModel ? hardwareOf(selectedDownloadModel, data) : hardware;
    const downloadPlatforms = [...new Set(downloadModels.map(other => hardwareOf(other, data)))].sort(byPlatformOrder);
    const dialogId = `get-model-${m.id}`;
    const description = window.HubI18n?.locale === 'en' ? (m.descriptionEn || m.description) : m.description;
    const downloadDialog = selectedDownloadModel ? `<dialog class="mz-download-dialog" id="${esc(dialogId)}" aria-labelledby="${esc(dialogId)}-title"><form method="dialog" class="mz-download-dialog-shell"><header class="mz-download-dialog-header"><div><span>获取模型</span><h2 id="${esc(dialogId)}-title">${esc(m.name)}</h2></div><button type="submit" value="cancel" class="mz-dialog-close" aria-label="关闭" title="关闭">${icon('x')}</button></header><div class="mz-download-dialog-body"><fieldset class="mz-download-step"><legend><span>1</span>选择芯片</legend><div class="mz-chip-options">${downloadPlatforms.map((platform, index) => `<label class="mz-download-choice"><input type="radio" name="download-platform" value="${esc(platform)}" ${platform === selectedPlatform ? 'checked' : ''}><span class="mz-choice-content"><strong>${esc(platform)}</strong></span><span class="mz-choice-check">${icon('check')}</span></label>`).join('')}</div></fieldset><fieldset class="mz-download-step"><legend><span>2</span>选择模型类型</legend><div class="mz-model-options">${downloadModels.map((model, index) => { const platform = hardwareOf(model, data); const asset = model.assets[0]; const size = byteSize(asset.sizeBytes) || ''; return `<label class="mz-download-choice mz-model-choice" data-platform="${esc(platform)}" ${platform === selectedPlatform ? '' : 'hidden'}><input type="radio" name="download-model" value="${esc(model.id)}" data-url="${esc(asset.url)}" data-filename="${esc(asset.filename)}" data-size="${esc(size)}" ${model.id === selectedDownloadModel.id ? 'checked' : ''}><span class="mz-choice-content"><strong>${esc(model.variantName || model.name)}</strong></span><span class="mz-choice-check">${icon('check')}</span></label>`; }).join('')}</div></fieldset><div class="mz-selected-file"><span>下载文件</span><code data-download-filename>${esc(selectedDownloadModel.assets[0].filename)}</code><small data-download-size>${esc(byteSize(selectedDownloadModel.assets[0].sizeBytes) || '')}</small></div></div><footer class="mz-download-dialog-footer"><button type="submit" value="cancel" class="mz-dialog-cancel">取消</button><a class="button mz-dialog-download" data-download-link href="${esc(selectedDownloadModel.assets[0].url)}" target="_blank" rel="noopener">${icon('download')}下载</a></footer></form></dialog>` : '';
    return `<nav class="mz-breadcrumb" aria-label="模型导航"><a class="back-link" href="#">Model Zoo</a><span aria-hidden="true">/</span><span>${m.name}</span></nav>
      <section class="mz-overview"><div class="mz-introduction"><span class="mz-task">${m.task}</span><h1>${m.name}</h1><p data-i18n-zh="${esc(m.description)}" data-i18n-en="${esc(m.descriptionEn || m.description)}">${esc(description)}</p><div class="mz-primary-actions"><button class="button" type="button" data-open-download>${icon('download')}获取模型</button>${external(m.source, '模型仓库', 'mz-text-link')}</div></div><figure class="mz-demo"><img src="${esc(m.coverImage)}" alt="${esc(m.name + ' · ' + m.coverLabel)}"></figure></section>
      <section class="mz-benchmark" aria-label="性能与精度"><div class="mz-benchmark-title"><h2>性能与精度</h2></div><div class="mz-configuration"><div><span>目标硬件</span>${dropdown('benchmark-hardware', '目标硬件', hardwareOptions.map(value => ({ value, label: value })), hardware)}</div><div><span>模型变体</span>${dropdown('benchmark-variant', '模型变体', variants.filter(other => hardwareOf(other, data) === hardware).sort((a, b) => sizeRank(a.modelSize) - sizeRank(b.modelSize)).map(other => ({ value: other.id, label: other.variantName || other.name })), m.id)}</div><div><span>Runtime 并发</span>${dropdown('benchmark-condition', 'Runtime 并发', conditions.map(value => ({ value: String(value), label: threadLabel(value) })), String(conditions[0] || ''), !conditions.length)}</div></div>
      ${b ? `<div class="mz-benchmark-body"><div class="mz-performance"><div class="mz-performance-grid" id="detail-performance" aria-live="polite"></div></div></div>` : `<div class="mz-benchmark-empty"><div><h3>暂无性能汇总</h3></div>${external(m.source + '/evaluator', '查看评测说明')}</div>`}<div class="mz-report-actions">${m.reportDataUrl ? `<a class="mz-report-link" href="reports.html?report=${encodeURIComponent(m.id)}&from=${encodeURIComponent(m.id)}">${icon('scan-eye')}查看转换详情</a>` : (m.reportUrl || m.reportSourceUrl) ? external(m.reportUrl || m.reportSourceUrl, 'OE 转换报告') : '<span class="mz-report-placeholder" aria-disabled="true">OE 转换报告</span>'}</div></section>
      <div class="mz-information"><div class="mz-main-column"><section class="mz-section"><h2>模型参数</h2><dl class="mz-specifications">${propertyRows(m)}</dl></section>
      <section class="mz-section" id="downloads"><div class="mz-section-heading"><h2>模型文件</h2><span>${m.assets.length} 个文件</span></div><div class="mz-files">${m.assets.map(a => { const size = byteSize(a.sizeBytes); const downloadLabel = `${icon('download')}<span>下载</span>${size ? `<span class="mz-download-size">${esc(size)}</span>` : ''}`; return `<div class="mz-file"><div class="mz-file-copy"><div class="mz-file-title"><h3>${assetRole(a)}</h3></div><code>${esc(a.filename)}</code></div>${external(a.url, downloadLabel, 'mz-download')}</div>`; }).join('')}</div></section>
      </div>
      <aside class="mz-side-column"><section><h2>目标平台</h2><div class="mz-platform-list">${hardwareOptions.map(platform => { const target = variants.find(other => hardwareOf(other, data) === platform); const active = platform === hardware; return `<button type="button" class="mz-platform-option${active ? ' is-active' : ''}" data-platform-target="${esc(target?.id || '')}" ${active || !target ? 'disabled' : ''} aria-label="切换到 ${esc(platform)}"><span class="mz-platform-mark">${icon('cpu')}</span><strong>${esc(platform)}</strong></button>`; }).join('')}</div></section><section><h2>开发资源</h2><div class="mz-resource-links">${external(m.source, '模型仓库')}${external(m.source + '/runtime/python', '运行文档')}${external(m.source + '/conversion', '模型转换')}</div></section><section><h2>模型许可</h2><div class="mz-resource-links">${m.licenseName && m.licenseUrl ? external(m.licenseUrl, esc(m.licenseName)) : '<span>未声明</span>'}</div></section></aside></div>
      ${related.length ? `<section class="mz-related"><div class="mz-section-heading"><h2>相关模型</h2><a href="#">查看全部模型</a></div><div class="mz-related-grid">${related.map(other => `<a href="#model/${other.id}" class="mz-related-model"><img src="${esc(other.coverImage)}" alt="${esc(other.name)}" loading="lazy"><div><h3>${other.name}</h3><span>${other.task}</span></div></a>`).join('')}</div></section>` : ''}
      ${downloadDialog}
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
    const downloadDialog = root.querySelector('.mz-download-dialog');
    const openDownloadButton = root.querySelector('[data-open-download]');
    let downloadOpener = null;
    function syncDownloadDialog() {
      if (!downloadDialog) return;
      const selectedPlatform = downloadDialog.querySelector('input[name="download-platform"]:checked')?.value;
      const modelChoices = [...downloadDialog.querySelectorAll('.mz-model-choice')];
      for (const choice of modelChoices) choice.hidden = choice.dataset.platform !== selectedPlatform;
      let selectedModel = downloadDialog.querySelector('input[name="download-model"]:checked');
      if (!selectedModel || selectedModel.closest('.mz-model-choice')?.hidden) {
        selectedModel = modelChoices.find(choice => !choice.hidden)?.querySelector('input[name="download-model"]');
        if (selectedModel) selectedModel.checked = true;
      }
      if (!selectedModel) return;
      downloadDialog.querySelector('[data-download-filename]').textContent = selectedModel.dataset.filename;
      const size = downloadDialog.querySelector('[data-download-size]');
      size.textContent = selectedModel.dataset.size || '';
      size.hidden = !selectedModel.dataset.size;
      downloadDialog.querySelector('[data-download-link]').href = selectedModel.dataset.url;
    }
    listen(openDownloadButton, 'click', () => {
      closeSelect();
      downloadOpener = openDownloadButton;
      syncDownloadDialog();
      document.documentElement.classList.add('mz-dialog-open');
      if (typeof downloadDialog?.showModal === 'function') downloadDialog.showModal();
      else downloadDialog?.setAttribute('open', '');
    });
    listen(downloadDialog, 'change', event => {
      if (event.target.matches('input[name="download-platform"], input[name="download-model"]')) {
        syncDownloadDialog();
      }
    });
    listen(downloadDialog, 'click', event => {
      if (event.target === downloadDialog) downloadDialog.close();
    });
    listen(downloadDialog, 'close', () => {
      document.documentElement.classList.remove('mz-dialog-open');
      downloadOpener?.focus();
      downloadOpener = null;
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
      const delay = pick('latency'), speed = pick('throughput');
      const endToEndRecords = Array.isArray(b.endToEnd)
        ? b.endToEnd
        : b.endToEnd ? [b.endToEnd] : [];
      const endToEnd = endToEndRecords.find(record =>
        Number(record?.timing?.runtimeSubmissionThreads || record?.timing?.pipelineStreams) === threads
      ) || (endToEndRecords.length === 1 && conditions.length === 1 ? endToEndRecords[0] : null);
      const preprocess = endToEnd?.metrics?.preprocess || pick('pre_process_latency');
      const postprocess = endToEnd?.metrics?.postprocess || pick('post_process_latency');
      const endToEndLatency = endToEnd?.metrics?.endToEnd || pick('end_to_end_latency');
      const value = (record, unit) => record ? `${amount(record)}<small>${esc(unit)}</small>` : '—';
      const plainLatency = record => record ? `${amount(record)} ${record.unit || 'ms'}` : '未测';
      const e2eThreads = Number(endToEnd?.timing?.opencvThreads);
      const e2eStreams = Number(endToEnd?.timing?.pipelineStreams);
      const e2eSubmissionThreads = Number(endToEnd?.timing?.runtimeSubmissionThreads);
      const e2eThroughput = Number(endToEnd?.throughputFps);
      const e2eImplementation = /cpp/i.test(endToEnd?.timing?.implementation || '') ? 'C++' : endToEnd?.timing?.implementation;
      const e2eThroughputText = Number.isFinite(e2eThroughput) ? `${Number(e2eThroughput.toFixed(2))} FPS` : '未测';
      const infoRow = (label, rowValue) => `<span class="mz-popover-row"><span>${esc(label)}</span><b>${esc(rowValue)}</b></span>`;
      const e2eRows = [
        e2eImplementation ? infoRow('实现', e2eImplementation) : '',
        Number.isFinite(e2eThreads) ? infoRow('CPU 线程', String(e2eThreads)) : '',
        Number.isFinite(e2eStreams) ? infoRow('流水线', e2eStreams === 1 ? '单路' : `${e2eStreams} 路`) : '',
        Number.isFinite(e2eSubmissionThreads) ? infoRow('Runtime 提交', e2eSubmissionThreads === 1 ? '单线程' : `${e2eSubmissionThreads} 线程`) : '',
        infoRow('前处理', plainLatency(preprocess)),
        infoRow('后处理', plainLatency(postprocess)),
        infoRow('端到端吞吐量', e2eThroughputText),
      ].join('');
      const endToEndTile = `<div class="${endToEndLatency ? '' : 'is-unmeasured'}"><span class="mz-accuracy-label">端到端延迟<button type="button" class="mz-accuracy-info" aria-label="端到端延迟：查看完整流水线条件" aria-haspopup="dialog">${icon('circle-help')}<span class="mz-accuracy-popover mz-e2e-popover" role="dialog" aria-label="端到端流水线条件">${e2eRows}</span></button></span><strong>${endToEndLatency ? value(endToEndLatency, endToEndLatency.unit || 'ms') : '未测'}</strong></div>`;
      const staged = [...new Set((b.accuracy || []).filter(a => a.model_stage).map(a => a.metric))];
      const accuracyTiles = staged.map(metric => {
        const find = stage => (b.accuracy || []).find(a => a.metric === metric && a.model_stage === stage);
        const reference = find('float'), quantized = find('quantized');
        const tileName = metric === 'top-1' ? 'Top-1' : staged.length > 1 ? metric.replace(/-all-map-50-95$/, ' mAP50-95') : 'mAP50-95';
        const plain = record => record ? `${amount(record, true)}%` : '—';
        const scale = record => record.unit === 'ratio' ? record.value * 100 : record.value;
        const delta = reference && quantized ? Number((scale(quantized) - scale(reference)).toFixed(2)) : null;
        const big = record => record ? `${amount(record, true)}<small>%</small>` : '—';
        const deltaText = delta === null ? '—' : `${delta > 0 ? '+' : delta < 0 ? '−' : ''}${Math.abs(delta)}%`;
        const comparisonRows = `${infoRow('浮点参考', plain(reference))}${infoRow('差距', deltaText)}`;
        return `<div><span class="mz-accuracy-label">${esc(tileName)}<button type="button" class="mz-accuracy-info" aria-label="${esc(tileName)}：查看浮点参考" aria-haspopup="dialog">${icon('circle-help')}<span class="mz-accuracy-popover" role="dialog" aria-label="${esc(tileName)}对照">${comparisonRows}</span></button></span><strong>${big(quantized || reference)}</strong></div>`;
      });
      root.querySelector('#detail-performance').innerHTML = `<div><span>Runtime 延迟</span><strong>${value(delay, delay?.unit || 'ms')}</strong></div>${endToEndTile}<div><span>吞吐量</span><strong>${value(speed, speed?.unit || 'fps')}</strong></div>${accuracyTiles.join('')}`;
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
    for (const button of root.querySelectorAll('[data-platform-target]')) {
      listen(button, 'click', () => {
        const targetId = button.dataset.platformTarget;
        if (targetId && targetId !== m.id) location.hash = '#model/' + targetId;
      });
    }
    performance();
  }
  window.ModelDetail = { render, bind, destroy };
})();
