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
  const metricName = (metric, data) => data?.accuracyMetricLabels?.[metric] || ({
    latency: 'Runtime 延迟', throughput: '吞吐量', pre_process_latency: '前处理延迟',
    post_process_latency: '后处理延迟', end_to_end_latency: '端到端延迟',
    'bbox-all-map-50-95': '检测 mAP50–95', 'mask-all-map-50-95': '分割 mAP50–95',
    'keypoints-all-map-50-95': '关键点 AP50–95', 'top-1': 'Top-1', 'top-5': 'Top-5',
    'obb-map-50': '旋转框 mAP50',
    'bbox-all-map-50-95-retention': '精度保留率',
  }[metric] || { zh: metric, en: metric });
  const threadLabel = threads => threads === 1 ? 'Runtime 单路' : `Runtime ${threads} 路并发`;
  const formats = m => [...new Set(m.assets.map(asset => asset.format.toUpperCase()))].join(' + ');
  const hardwareOf = (model, data) => model.hardware || model.benchmark?.environment?.hardware || data.release.compatibility.hardware;
  // Preferred hardware display order: strongest S-series first, X5 last.
  const platformDisplayOrder = hardware => ({
    'RDK S600': 0, 'RDK S100P': 1, 'RDK S100': 2, 'RDK X5': 3,
  }[String(hardware)] ?? 99);
  const byPlatformOrder = (a, b) => platformDisplayOrder(a) - platformDisplayOrder(b) || String(a).localeCompare(String(b));
  const sizePreference = ['n', 's', 'm', 'l', 'x', 'atto', 'femto', 'pico', 'nano', 'tiny', 'small', 'base', 'large', 'b0', 'b1', 'b2', 'b3', 'b4', 'b5', 's0', 's1', 's2', 's3', 's4'];
  const sizeRank = size => { const index = sizePreference.indexOf(String(size || '').toLowerCase()); return index < 0 ? sizePreference.length : index; };
  const relatedGroupId = model => model.family || model.catalogId || `${model.name}/${model.taskId || model.task}`;
  const formatShape = value => {
    if (Array.isArray(value)) return value.join(' × ');
    const text = String(value ?? '').trim();
    if (!text) return '';
    const parts = text.replace(/^[\[(]+|[\]\)]+$/g, '').split(/[,x×]/i).map(part => part.trim()).filter(Boolean);
    if (parts.length >= 2 && parts.every(part => /^\d+$/.test(part))) return parts.join(' × ');
    return text.replace(/\s+x\s+/g, ' × ');
  };
  const formatSourceInput = source => {
    if (!source) return '';
    return [
      source.format ? String(source.format).toUpperCase() : '',
      source.dtype ? String(source.dtype).toUpperCase() : '',
      source.layout || '',
      formatShape(source.shape),
    ].filter(Boolean).join(' · ');
  };
  const taskDomain = model => {
    const ids = Array.isArray(model.tasks) && model.tasks.length ? model.tasks : (model.taskId ? [model.taskId] : []);
    const groups = window.CatalogFacets?.groups || [];
    return groups.find(group => group.tasks.some(([id]) => ids.includes(id)))?.id || '';
  };
  // Tasks with a published page use that page's accuracy tiles. Tasks without
  // one still get two named accuracy tiles so the row stays five wide.
  const pendingAccuracyTiles = {
    cls: [
      { metric: 'top-1', zh: 'Top-1 准确率', en: 'Top-1 accuracy' },
      { metric: 'top-5', zh: 'Top-5 准确率', en: 'Top-5 accuracy' },
    ],
    detect: [{ metric: 'bbox-all-map-50-95', zh: '检测 mAP50–95', en: 'Detection mAP50–95' }],
    seg: [
      { metric: 'segmentation-box-all-map-50-95', zh: '检测框 AP50–95', en: 'Box AP50–95' },
      { metric: 'mask-all-map-50-95', zh: '掩码 AP50–95', en: 'Mask AP50–95' },
    ],
    pose: [{ metric: 'keypoints-all-map-50-95', zh: '关键点 AP50–95', en: 'Keypoint AP50–95' }],
    obb: [{ metric: 'obb-map-50', zh: '旋转框 mAP50', en: 'Oriented box mAP50' }],
    semseg: [
      { metric: 'miou', zh: 'mIoU', en: 'mIoU' },
      { metric: 'pixel-accuracy', zh: '像素准确率', en: 'Pixel accuracy' },
    ],
    'prompt-seg': [
      { metric: 'miou', zh: 'mIoU', en: 'mIoU' },
      { metric: 'dice', zh: 'Dice', en: 'Dice' },
    ],
    depth: [
      { metric: 'abs-rel', zh: 'AbsRel', en: 'AbsRel' },
      { metric: 'delta-1', zh: 'δ1', en: 'δ1' },
    ],
    track: [
      { metric: 'mota', zh: 'MOTA', en: 'MOTA' },
      { metric: 'idf1', zh: 'IDF1', en: 'IDF1' },
    ],
    embed: [
      { metric: 'knn-top1', zh: 'k-NN Top-1', en: 'k-NN Top-1' },
      { metric: 'linear-top1', zh: '线性探测 Top-1', en: 'Linear probe Top-1' },
    ],
    'ocr-det': [
      { metric: 'detection-precision', zh: '检测精确率', en: 'Detection precision' },
      { metric: 'detection-recall', zh: '检测召回率', en: 'Detection recall' },
    ],
    'ocr-rec': [
      { metric: 'recognition-accuracy', zh: '识别准确率', en: 'Recognition accuracy' },
      { metric: 'cer', zh: '字符错误率', en: 'Character error rate' },
    ],
    similarity: [
      { metric: 'clip-zeroshot-top1', zh: 'CLIP zero-shot Top-1', en: 'CLIP zero-shot Top-1' },
      { metric: 'retrieval-r-at-1', zh: '图文检索 R@1', en: 'Image-text retrieval R@1' },
    ],
    'ov-detect': [
      { metric: 'ov-map-50-95', zh: '开放词汇 mAP50–95', en: 'Open-vocabulary mAP50–95' },
      { metric: 'ov-map-50', zh: '开放词汇 mAP50', en: 'Open-vocabulary mAP50' },
    ],
    lpr: [
      { metric: 'plate-accuracy', zh: '车牌准确率', en: 'Plate accuracy' },
      { metric: 'character-accuracy', zh: '字符准确率', en: 'Character accuracy' },
    ],
    matting: [
      { metric: 'sad', zh: 'SAD', en: 'SAD' },
      { metric: 'mse', zh: 'MSE', en: 'MSE' },
    ],
    'video-cls': [
      { metric: 'top-1', zh: 'Top-1 准确率', en: 'Top-1 accuracy' },
      { metric: 'top-5', zh: 'Top-5 准确率', en: 'Top-5 accuracy' },
    ],
    avatar: [
      { metric: 'lse-d', zh: 'LSE-D', en: 'LSE-D' },
      { metric: 'fid', zh: 'FID', en: 'FID' },
    ],
    'text-gen': [
      { metric: 'perplexity', zh: '困惑度', en: 'Perplexity' },
      { metric: 'generation-throughput', zh: '生成吞吐', en: 'Generation throughput' },
    ],
    vlm: [
      { metric: 'vqa-accuracy', zh: 'VQA 准确率', en: 'VQA accuracy' },
      { metric: 'retrieval-r-at-1', zh: '图文检索 R@1', en: 'Image-text retrieval R@1' },
    ],
    manip: [
      { metric: 'success-rate', zh: '任务成功率', en: 'Task success rate' },
      { metric: 'trajectory-error', zh: '轨迹误差', en: 'Trajectory error' },
    ],
    loco: [
      { metric: 'velocity-error', zh: '速度跟踪误差', en: 'Velocity tracking error' },
      { metric: 'survival-time', zh: '存活时间', en: 'Survival time' },
    ],
    drive: [
      { metric: 'planning-error', zh: '规划误差', en: 'Planning error' },
      { metric: 'collision-rate', zh: '碰撞率', en: 'Collision rate' },
    ],
    lane: [
      { metric: 'lane-f1', zh: '车道线 F1', en: 'Lane F1' },
      { metric: 'lane-accuracy', zh: '车道线准确率', en: 'Lane accuracy' },
    ],
    'point-seg': [
      { metric: 'point-miou', zh: '点云 mIoU', en: 'Point mIoU' },
      { metric: 'point-accuracy', zh: '点云准确率', en: 'Point accuracy' },
    ],
    asr: [
      { metric: 'wer', zh: '词错误率 WER', en: 'Word error rate' },
      { metric: 'cer', zh: '字错误率 CER', en: 'Character error rate' },
    ],
    kws: [
      { metric: 'kws-accuracy', zh: '唤醒准确率', en: 'Spotting accuracy' },
      { metric: 'false-alarm-rate', zh: '误唤醒率', en: 'False alarm rate' },
    ],
  };
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
  const singleStream = (model, metric) => (model.benchmark?.performance || []).find(record => record.metric === metric && (record.concurrency || 1) === 1);
  const stagedAccuracy = (model, metric) => ['quantized', 'board', 'float']
    .map(stage => (model.benchmark?.accuracy || []).find(record => record.metric === metric && record.model_stage === stage))
    .find(Boolean);
  // Each size choice carries three aligned figures — single-stream latency,
  // the headline accuracy and the file size — as value-over-label cells.
  function choiceStats(model, data) {
    const latency = singleStream(model, 'latency');
    // Segmentation is judged by its masks; other tasks have one headline metric.
    const staged = (model.benchmark?.accuracy || []).filter(record => record.model_stage).map(record => record.metric);
    const metric = staged.find(id => id.startsWith('mask-')) || staged[0];
    const accuracy = metric ? stagedAccuracy(model, metric) : null;
    const label = metric ? metricName(metric, data) : null;
    const name = typeof label === 'string' ? { zh: label, en: label } : label;
    // The column shows the metric itself ("mAP50–95", "Top-1"), which reads
    // the same in both languages; the task is in the dialog title. Mask AP
    // keeps its qualifier so it is not mistaken for box AP.
    const metricToken = name?.zh.match(/m?AP\d+(?:–\d+)?|Top-\d+/)?.[0];
    const caption = !name ? '' : metricToken && !metric.startsWith('mask-')
      ? `<span data-no-i18n>${esc(metricToken)}</span>`
      : `<span data-i18n-zh="${esc(name.zh)}" data-i18n-en="${esc(name.en || name.zh)}">${esc(name.zh)}</span>`;
    const size = byteSize(model.assets?.[0]?.sizeBytes);
    const cell = (value, unit, caption) => `<span class="mz-store-stat"><b>${esc(value)}<small>${esc(unit)}</small></b>${caption}</span>`;
    const cells = [
      latency ? cell(amount(latency), latency.unit || 'ms', '<span>延迟</span>') : '',
      accuracy && caption ? cell(amount(accuracy, true), '%', caption) : '',
      size ? cell(...size.split(' '), '<span>文件大小</span>') : '',
    ].join('');
    return cells ? `<span class="mz-store-stats">${cells}</span>` : '';
  }
  // Label files the Ultralytics YOLO sample ships per task. An explicit
  // --model-path is treated as a custom model, so labels must be passed;
  // pose models are single-class and draw without one.
  const yoloLabelFiles = { detect: 'coco_classes.names', seg: 'coco_classes.names', cls: 'imagenet_classes.names', obb: 'ultralytics_dota_classes.names' };
  // Native board commands for this exact artifact: fetch the file the page
  // publishes (its benchmark and checksum), then run the sample against it.
  function quickstartSection(source, steps) {
    if (!steps.length) return '';
    const step = (index, title, command) => `<li class="mz-quickstart-step"><div class="mz-quickstart-label"><span>${index}</span>${title}</div><div class="mz-code"><pre><code data-no-i18n>${esc(command)}</code></pre><button type="button" class="mz-copy" data-copy-code aria-label="复制命令" title="复制命令">${icon('copy')}</button></div></li>`;
    const link = source ? `<a href="${esc(source)}#quickstart" target="_blank" rel="noopener">完整说明</a>` : '';
    return `<section class="mz-section mz-quickstart"><div class="mz-section-heading"><h2>快速上手</h2>${link}</div><ol class="mz-quickstart-steps">${steps.map((item, index) => step(index + 1, item.title, item.command)).join('')}</ol></section>`;
  }
  function quickStart(model) {
    const asset = model.assets?.[0];
    const [sample, family, task] = String(model.catalogId || '').split('/');
    if (sample !== 'ultralytics_yolo' || !asset?.url || !model.releasePlatform || !family || !task) return '';
    const root = 'samples/vision/ultralytics_yolo';
    const modelPath = `${root}/model/${asset.filename}`;
    const download = `wget -O ${modelPath} \\\n  ${asset.url}`;
    const run = [
      `python3 ${root}/runtime/python/main.py \\`,
      `  --platform ${String(model.releasePlatform).toLowerCase()} --family ${family} --task ${task} \\`,
      `  --model-path ${modelPath}${yoloLabelFiles[task] ? ' \\' : ''}`,
      ...(yoloLabelFiles[task] ? [`  --label-file ${root}/test_data/${yoloLabelFiles[task]}`] : []),
    ].join('\n');
    return quickstartSection(model.source, [
      { title: '下载模型', command: download },
      { title: '运行推理', command: run },
    ]);
  }
  function pendingQuickStart(model) {
    if (!model.dataPending) return '';
    const steps = [];
    if (model.quickstart?.download) steps.push({ title: '下载模型', command: model.quickstart.download });
    if (model.quickstart?.run) steps.push({ title: '运行推理', command: model.quickstart.run });
    return quickstartSection(model.source, steps);
  }
  const isRdkSampleSource = model => Boolean(model.source && (
    model.sourceType === 'repository'
    || /^https?:\/\/github\.com\/D-Robotics\/rdk_model_zoo\/tree\/[^/]+\/samples\/vision\/ultralytics_yolo(?:\/|$)/i.test(model.source)
  ));
  const repositoryLinks = (model, cls = '') => {
    if (!model.source || (model.releaseStatus === 'candidate' && !isRdkSampleSource(model))) return '';
    return external(model.source, '模型仓库', cls);
  };
  const developerResourceLinks = model => {
    if (!model.source || (model.releaseStatus === 'candidate' && !isRdkSampleSource(model))) return '';
    if (model.dataPending) {
      const links = [external(model.source, '模型仓库')];
      links.push(external(model.usageSource || model.source, '运行文档'));
      const conversionDirectories = new Set();
      for (const config of model.conversions || []) {
        const directory = String(config.path || '').includes('/')
          ? config.path.slice(0, config.path.lastIndexOf('/'))
          : '';
        const url = config.url && directory
          ? config.url.replace(/\/blob\/([^/]+)\/.*$/, `/tree/$1/${directory}`)
          : config.url;
        if (url && !conversionDirectories.has(url)) {
          conversionDirectories.add(url);
          links.push(external(url, '模型转换'));
        }
      }
      for (const item of model.upstream || []) {
        if (!item?.url) continue;
        links.push(external(item.url, item.kind === 'paper' ? '论文' : (item.name || '上游仓库')));
      }
      return links.join('');
    }
    return `${external(model.source, '模型仓库')}${external(model.source + '/runtime/python', '运行文档')}${external(model.source + '/conversion', '模型转换')}`;
  };
  function propertyRows(model) {
    if (model.dataPending) return pendingPropertyRows(model);
    const p = window.MODEL_PROPERTIES?.[model.id] || {};
    const shape = formatShape;
    const row = (label, value, title = '') => `<dt>${label}</dt><dd${title ? ` title="${esc(title)}"` : ''}>${esc(value)}</dd>`;
    const rows = [];
    if (p.parameterCount) rows.push(row('参数量', `${Number((p.parameterCount / 1e6).toFixed(3))} M`, String(p.parameterCount)));
    if (p.parameterCountMillions) rows.push(row('参数量', `${p.parameterCountMillions} M`));
    if (p.gflops) {
      const resolution = Array.isArray(p.inputShape) ? ` @ ${p.inputShape.slice(-2).join(' × ')}` : '';
      rows.push(row('浮点计算量', `${Number(Number(p.gflops).toFixed(2))} GFLOPs${resolution}`));
    }
    if (p.sourceInput) {
      rows.push(row('源模型输入', formatSourceInput(p.sourceInput)));
    } else if (p.inputShape) rows.push(row('源模型输入', shape(p.inputShape)));
    else if (p.inputResolution) rows.push(row('输入尺寸', shape(p.inputResolution)));
    if (p.runtimeInput) rows.push(row('Runtime 输入', [p.runtimeInput.format ? String(p.runtimeInput.format).toUpperCase() : '', shape(p.runtimeInput.resolution)].filter(Boolean).join(' · ')));
    else if (p.inputLayout) rows.push(row('输入布局', p.inputLayout));
    const precisionPolicy = model.benchmark?.precisionPolicy;
    if (typeof precisionPolicy === 'string' && precisionPolicy.trim()) {
      const policy = precisionPolicy.trim().toLowerCase();
      const precisionLabel = policy === 'int8-int16'
        ? 'INT8 + INT16 混合精度'
        : policy === 'int8-fp16'
          ? 'INT8 + FP16 混合精度'
          : policy.toUpperCase();
      rows.push(row('量化精度', precisionLabel, policy));
    }
    if (p.predictionShape) rows.push(row('预测输出', shape(p.predictionShape)));
    if (p.keypointShape) rows.push(row('关键点数', `${p.keypointShape[0]}`));
    if (p.maskChannels) rows.push(row('掩码通道数', String(p.maskChannels)));
    if (p.maskPrototypeShape) rows.push(row('掩码原型', shape(p.maskPrototypeShape)));
    if (p.decoderInputShape) rows.push(row('解码器输入', shape(p.decoderInputShape)));
    for (const output of p.componentOutputs || []) rows.push(row(output.name, shape(output.shape)));
    const labels = { image: '图像输入', image_feature: '图像特征', texts: '文本输入', text_features: '文本特征' };
    for (const input of p.interfaces || []) rows.push(row(labels[input.name] || input.name, `${shape(input.shape)} · ${input.dtype}`));
    if (!rows.length) rows.push(row('输入尺寸', model.shape || '—'));
    return rows.join('');
  }
  function pendingPropertyRows(model) {
    const row = (label, value) => `<dt>${label}</dt><dd>${esc(value || '数据待补充')}</dd>`;
    const config = (model.conversions || []).find(item => item.trainInput || item.trainDtype || item.trainLayout || item.trainShape || item.resolution || item.runtimeInput) || {};
    const input = (model.inputs || []).find(item => item.format || item.layout || item.shape) || {};
    const explicit = model.sourceInput;
    const source = explicit && (explicit.format || explicit.dtype || explicit.layout || explicit.shape)
      ? explicit
      : {
          format: config.trainInput,
          dtype: config.trainDtype,
          layout: config.trainLayout,
          shape: config.trainShape || config.resolution,
        };
    const runtimeInput = [
      (input.format || config.runtimeInput) ? String(input.format || config.runtimeInput).toUpperCase() : '',
      formatShape(input.shape || config.resolution),
    ].filter(Boolean).join(' · ');
    return [
      row('参数量', '数据待补充'),
      row('浮点计算量', '数据待补充'),
      row('源模型输入', formatSourceInput(source)),
      row('Runtime 输入', runtimeInput),
      row('量化精度', '数据待补充'),
    ].join('');
  }

  function render({ model: m, models, data }) {
    const b = m.benchmark;
    const hardware = hardwareOf(m, data);
    const groupId = m.catalogId || `${m.name}/${m.taskId || m.task}`;
    const variants = models.filter(other => (other.catalogId || `${other.name}/${other.taskId || other.task}`) === groupId);
    const hardwareOptions = [...new Set(variants.map(other => hardwareOf(other, data)))].sort(byPlatformOrder);
    const performance = b?.performance || [];
    const conditionChoices = m.dataPending
      ? (b?.conditionOptions?.length ? b.conditionOptions.map(option => ({
          value: option.key,
          label: Number.isInteger(option.concurrency) ? threadLabel(option.concurrency) : (option.label || 'Runtime 单路'),
        })) : [{ value: '1', label: 'Runtime 单路' }])
      : [...new Set(performance.filter(p => ['latency', 'throughput'].includes(p.metric)).map(p => p.concurrency || 1))].sort((a, b) => a - b)
          .map(value => ({ value: String(value), label: threadLabel(value) }));
    const sameTask = models.filter(other => relatedGroupId(other) !== relatedGroupId(m) && other.task === m.task);
    const domainPeers = !sameTask.length && m.dataPending
      ? models.filter(other => relatedGroupId(other) !== relatedGroupId(m) && taskDomain(other) && taskDomain(other) === taskDomain(m))
      : [];
    const relatedCandidates = sameTask.length ? sameTask : domainPeers;
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
    const coverSrc = m.coverImage || (m.dataPending ? 'assets/pending-cover.svg' : '');
    const cover = coverSrc
      ? `<figure class="mz-demo"><img src="${esc(coverSrc)}" alt="${esc(m.name)}"></figure>`
      : '';
    const benchmarkEmpty = `<div class="mz-benchmark-empty"><div><h3>暂无性能汇总</h3></div>${m.source ? external(m.source + '/evaluator', '查看评测说明') : ''}</div>`;
    const benchmarkBody = b
      ? `<div class="mz-benchmark-body"><div class="mz-performance"><div class="mz-performance-grid" id="detail-performance" aria-live="polite"></div></div></div>`
      : benchmarkEmpty;
    const upstreamLine = !m.dataPending && (m.upstream || []).length
      ? `<section class="mz-section"><h2>上游</h2><div class="mz-resource-links">${m.upstream.map(item => external(item.url, esc(item.name))).join('')}</div></section>`
      : '';
    const conversionLine = !m.dataPending && (m.conversions || []).length
      ? `<section class="mz-section"><h2>转换配置</h2><div class="mz-resource-links">${m.conversions.map(item => external(item.url, esc(item.path.split('/').pop()))).join('')}</div></section>`
      : '';
    const downloadLine = !m.dataPending && (m.documentedDownloads || []).length
      ? `<section class="mz-section"><h2>文档中的下载</h2><div class="mz-resource-links">${m.documentedDownloads.map(item => external(item.url, esc(item.filename))).join('')}</div></section>`
      : '';
    const usageLine = !m.dataPending && (m.usage || []).length
      ? `<section class="mz-section mz-quickstart"><div class="mz-section-heading"><h2>运行命令</h2>${m.usageSource ? external(m.usageSource, '命令来源') : ''}</div><div class="mz-code"><pre><code data-no-i18n>${esc(m.usage.join('\n'))}</code></pre></div></section>`
      : '';
    const documentedDownloads = (m.documentedDownloads || []).filter(item => /^https:\/\//.test(item.url || ''));
    const acquireButton = m.assets?.length
      ? `<button class="button" type="button" data-open-download>${icon('download')}获取模型</button>`
      : (m.dataPending && documentedDownloads.length > 1
        ? `<button class="button" type="button" data-open-download>${icon('download')}获取模型</button>`
        : (m.dataPending && documentedDownloads.length === 1
          ? `<a class="button" href="${esc(documentedDownloads[0].url)}" target="_blank" rel="noopener">${icon('download')}获取模型</a>`
          : (m.dataPending
            ? `<span title="模型产物待发布"><button class="button" type="button" disabled>${icon('download')}获取模型</button></span>`
            : '')));
    const kicker = m.dataPending
      ? `<div class="mz-kicker"><span class="mz-task">${m.task}</span><span class="pending-badge">数据待补充</span></div>`
      : `<span class="mz-task">${m.task}</span>`;
    const reportAction = m.reportDataUrl
      ? `<a class="mz-report-link" href="reports.html?report=${encodeURIComponent(m.id)}&from=${encodeURIComponent(m.id)}">${icon('scan-eye')}查看转换详情</a>`
      : (m.dataPending
        ? ((m.reportUrl || m.reportSourceUrl)
          ? `<a class="mz-report-link" href="${esc(m.reportUrl || m.reportSourceUrl)}" target="_blank" rel="noopener">${icon('scan-eye')}查看转换详情</a>`
          : `<span class="mz-report-link is-disabled" aria-disabled="true">${icon('scan-eye')}查看转换详情</span>`)
        : ((m.reportUrl || m.reportSourceUrl)
          ? external(m.reportUrl || m.reportSourceUrl, 'OE 转换报告')
          : '<span class="mz-report-placeholder" aria-disabled="true">OE 转换报告</span>'));
    const platformSizes = platform => downloadModels.filter(other => hardwareOf(other, data) === platform).length;
    const selectedAsset = selectedDownloadModel?.assets[0];
    const documentedDialog = !selectedDownloadModel && documentedDownloads.length > 1
      ? `<dialog class="mz-download-dialog mz-store" id="${esc(dialogId)}" aria-labelledby="${esc(dialogId)}-title"><form method="dialog" class="mz-store-shell"><header class="mz-store-header"><div><span>获取模型</span><h2 id="${esc(dialogId)}-title">${esc(m.name)}</h2></div><button type="submit" value="cancel" class="mz-dialog-close" aria-label="关闭" title="关闭">${icon('x')}</button></header><div class="mz-store-body"><div class="mz-resource-links">${documentedDownloads.map(item => external(item.url, esc(item.filename))).join('')}</div></div></form></dialog>`
      : '';
    const downloadDialog = selectedDownloadModel ? `<dialog class="mz-download-dialog mz-store" id="${esc(dialogId)}" aria-labelledby="${esc(dialogId)}-title"><form method="dialog" class="mz-store-shell"><header class="mz-store-header"><div><span>获取模型</span><h2 id="${esc(dialogId)}-title">${esc(m.name)}</h2></div><button type="submit" value="cancel" class="mz-dialog-close" aria-label="关闭" title="关闭">${icon('x')}</button></header><div class="mz-store-body"><fieldset class="mz-store-step"><legend>选择芯片</legend><div class="mz-store-grid">${downloadPlatforms.map(platform => `<label class="mz-store-choice"><input type="radio" name="download-platform" value="${esc(platform)}" ${platform === selectedPlatform ? 'checked' : ''}><span class="mz-store-choice-main"><strong>${esc(platform)}</strong><small>${platformSizes(platform)} 个尺寸</small></span></label>`).join('')}</div></fieldset><fieldset class="mz-store-step"><legend>选择尺寸</legend><div class="mz-store-list">${downloadModels.map(model => { const platform = hardwareOf(model, data); const asset = model.assets[0]; return `<label class="mz-store-choice mz-model-choice" data-platform="${esc(platform)}" data-model-size="${esc(model.modelSize || '')}" ${platform === selectedPlatform ? '' : 'hidden'}><input type="radio" name="download-model" value="${esc(model.id)}" data-url="${esc(asset.url)}" ${model.id === selectedDownloadModel.id ? 'checked' : ''}><span class="mz-store-choice-main"><strong>${esc(model.variantName || model.name)}</strong></span>${choiceStats(model, data)}</label>`; }).join('')}</div></fieldset></div><footer class="mz-store-summary"><a class="button mz-dialog-download" data-download-link href="${esc(selectedAsset.url)}" target="_blank" rel="noopener">${icon('download')}下载</a></footer></form></dialog>` : documentedDialog;
    return `<nav class="mz-breadcrumb" aria-label="模型导航"><a class="back-link" href="#">Model Zoo</a><span aria-hidden="true">/</span><span>${m.name}</span></nav>
      <section class="mz-overview"><div class="mz-introduction">${kicker}<h1>${m.name}</h1><p data-i18n-zh="${esc(m.description)}" data-i18n-en="${esc(m.descriptionEn || m.description)}">${esc(description)}</p><div class="mz-primary-actions">${acquireButton}<button class="button secondary mz-ask-model" type="button" data-ask-model><span class="mz-icon" style="--icon:url('assets/icons/sparkles.svg')" aria-hidden="true"></span>询问此模型</button>${repositoryLinks(m, 'mz-text-link')}</div></div>${cover}</section>
      <section class="mz-benchmark" aria-label="性能与精度"><div class="mz-benchmark-title"><h2>性能与精度</h2></div><div class="mz-configuration"><div><span>目标硬件</span>${dropdown('benchmark-hardware', '目标硬件', hardwareOptions.map(value => ({ value, label: value })), hardware)}</div><div><span>模型变体</span>${dropdown('benchmark-variant', '模型变体', variants.filter(other => hardwareOf(other, data) === hardware).sort((a, b) => sizeRank(a.modelSize) - sizeRank(b.modelSize) || String(a.variantName || a.name).localeCompare(String(b.variantName || b.name))).map(other => ({ value: other.id, label: other.variantName || other.name })), m.id)}</div><div><span>Runtime 并发</span>${dropdown('benchmark-condition', 'Runtime 并发', conditionChoices, conditionChoices[0]?.value || '', m.dataPending ? false : !conditionChoices.length)}</div></div>
      ${benchmarkBody}<div class="mz-report-actions">${reportAction}</div></section>
      <div class="mz-information"><div class="mz-main-column"><section class="mz-section"><h2>模型参数</h2><dl class="mz-specifications">${propertyRows(m)}</dl></section>${upstreamLine}${conversionLine}${downloadLine}${usageLine}${quickStart(m)}${pendingQuickStart(m)}
      </div>
      <aside class="mz-side-column"><section><h2>目标平台</h2><div class="mz-platform-list">${hardwareOptions.map(platform => { const target = variants.find(other => hardwareOf(other, data) === platform && other.modelSize === m.modelSize) || variants.find(other => hardwareOf(other, data) === platform); const active = platform === hardware; return `<button type="button" class="mz-platform-option${active ? ' is-active' : ''}" data-platform-target="${esc(target?.id || '')}" ${active || !target ? 'disabled' : ''} aria-label="切换到 ${esc(platform)}"><span class="mz-platform-mark">${icon('cpu')}</span><strong>${esc(platform)}</strong></button>`; }).join('')}</div></section><section><h2>开发资源</h2><div class="mz-resource-links">${developerResourceLinks(m)}</div></section><section><h2>模型许可</h2><div class="mz-resource-links">${m.licenseName && m.licenseUrl ? external(m.licenseUrl, esc(m.licenseName)) : '<span>未声明</span>'}</div></section></aside></div>
      ${related.length ? `<section class="mz-related"><div class="mz-section-heading"><h2>相关模型</h2><a href="#">查看全部模型</a></div><div class="mz-related-grid">${related.map(other => `<a href="#model/${other.id}" class="mz-related-model"><img src="${esc(other.coverImage || (other.dataPending ? 'assets/pending-cover.svg' : ''))}" alt="${esc(other.name)}" loading="lazy"><div><h3>${other.name}</h3><span>${other.task}</span></div></a>`).join('')}</div></section>` : ''}
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
        const size = selectedModel?.closest('.mz-model-choice')?.dataset.modelSize;
        const visible = modelChoices.filter(choice => !choice.hidden);
        selectedModel = (visible.find(choice => choice.dataset.modelSize === size) || visible[0])?.querySelector('input[name="download-model"]');
        if (selectedModel) selectedModel.checked = true;
      }
      if (!selectedModel) return;
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
    // The sheet sinks back out before it closes, whichever way it is dismissed.
    function closeDownloadDialog() {
      if (!downloadDialog?.open || downloadDialog.classList.contains('is-closing')) return;
      if (typeof downloadDialog.close !== 'function') return downloadDialog.removeAttribute('open');
      if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return downloadDialog.close();
      const finish = () => {
        clearTimeout(timer);
        downloadDialog.removeEventListener('transitionend', onEnd);
        downloadDialog.classList.remove('is-closing');
        downloadDialog.close();
      };
      const onEnd = event => { if (event.target === downloadDialog && event.propertyName === 'opacity') finish(); };
      const timer = setTimeout(finish, 320);
      downloadDialog.addEventListener('transitionend', onEnd);
      downloadDialog.classList.add('is-closing');
    }
    listen(downloadDialog, 'click', event => {
      if (event.target === downloadDialog) closeDownloadDialog();
    });
    listen(downloadDialog, 'submit', event => { event.preventDefault(); closeDownloadDialog(); });
    listen(downloadDialog, 'cancel', event => { event.preventDefault(); closeDownloadDialog(); });
    listen(downloadDialog, 'close', () => {
      document.documentElement.classList.remove('mz-dialog-open');
      downloadOpener?.focus();
      downloadOpener = null;
    });
    listen(root, 'click', async event => {
      const button = event.target.closest('[data-copy-code]');
      if (!button) return;
      const code = button.parentElement.querySelector('code');
      const label = button.getAttribute('aria-label');
      try {
        await navigator.clipboard.writeText(code.textContent);
        button.dataset.state = 'copied';
        button.setAttribute('aria-label', '已复制');
      } catch {
        // Clipboard access can be denied; select the command for a manual copy.
        getSelection()?.selectAllChildren(code);
        button.setAttribute('aria-label', '请手动复制');
      }
      button.title = button.getAttribute('aria-label');
      refresh();
      setTimeout(() => { delete button.dataset.state; button.setAttribute('aria-label', label); button.title = label; refresh(); }, 1600);
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

    function pendingPerformance() {
      const grid = root.querySelector('#detail-performance');
      if (!grid) return;
      const selected = root.querySelector('#benchmark-condition')?.dataset.value || '';
      const inCondition = row => Boolean(selected) && row.conditionKey === selected;
      const records = (b?.performance || []).filter(inCondition);
      const headline = metric => {
        const matches = records.filter(row => row.metric === metric);
        return matches.find(row => row.statistic === 'mean') || matches.find(row => !row.statistic) || null;
      };
      const delay = headline('latency');
      const speed = headline('throughput');
      const stages = (b?.stages || []).filter(inCondition);
      const stageReading = label => stages.find(row => row.stageLabel === label && (row.statistic === 'mean' || !row.statistic)) || stages.find(row => row.stageLabel === label);
      const pendingText = record => record ? `${amount(record)} ${record.unit || ''}`.trim() : '数据待补充';
      const tileValue = record => record ? `${amount(record)}<small>${esc(record.unit || '')}</small>` : '数据待补充';
      const infoRow = (label, rowValue) => `<span class="mz-popover-row"><span>${esc(label)}</span><b>${esc(rowValue)}</b></span>`;
      const sourceRow = record => record?.source ? `<span class="mz-popover-row"><span>来源</span><b>${external(record.source, '记录')}</b></span>` : '';
      const provenance = (matches, includeValues = true) => {
        if (!matches.length) return '';
        const condition = matches.find(row => row.condition)?.condition;
        const source = matches.find(row => row.source);
        return [
          ...(includeValues ? matches.map(row => infoRow(row.statistic ? `统计量 ${row.statistic}` : '记录值', pendingText(row))) : []),
          condition ? infoRow('条件', condition) : '',
          sourceRow(source),
          infoRow('文档说明', '摘自仓库文档，不是本站正式发布的测量。'),
        ].join('');
      };
      const help = (label, rows) => rows
        ? `<button type="button" class="mz-accuracy-info" aria-label="${esc(label)}：查看来源和条件" aria-haspopup="dialog">${icon('circle-help')}<span class="mz-accuracy-popover mz-e2e-popover" role="dialog" aria-label="${esc(label)}">${rows}</span></button>`
        : '';
      const preprocess = stageReading('前处理');
      const postprocess = stageReading('后处理');
      const endToEndLatency = stageReading('端到端');
      const e2eRows = [
        infoRow('实现', '数据待补充'),
        infoRow('CPU 线程', '数据待补充'),
        infoRow('流水线', '数据待补充'),
        infoRow('Runtime 提交', '数据待补充'),
        infoRow('前处理', pendingText(preprocess)),
        infoRow('后处理', pendingText(postprocess)),
        infoRow('端到端吞吐量', '数据待补充'),
      ].join('');
      const endToEndTile = `<div class="${endToEndLatency ? '' : 'is-unmeasured'}"><span class="mz-accuracy-label">端到端延迟<button type="button" class="mz-accuracy-info" aria-label="端到端延迟：查看完整流水线条件" aria-haspopup="dialog">${icon('circle-help')}<span class="mz-accuracy-popover mz-e2e-popover" role="dialog" aria-label="端到端流水线条件">${e2eRows}</span></button></span><strong>${endToEndLatency ? tileValue(endToEndLatency) : '数据待补充'}</strong></div>`;
      const metricTile = (label, record, matches) => `<div class="${record ? '' : 'is-unmeasured'}"><span class="mz-accuracy-label">${label}${help(label, provenance(matches))}</span><strong>${tileValue(record)}</strong></div>`;
      const accuracyRows = b?.accuracy || [];
      const taskCode = String(m.catalogId || '').split('/').at(-1);
      const tileSpecs = pendingAccuracyTiles[taskCode] || [
        { metric: 'task-metric-1', zh: '任务指标 1', en: 'Task metric 1' },
        { metric: 'task-metric-2', zh: '任务指标 2', en: 'Task metric 2' },
      ];
      const percentText = record => {
        if (!record) return '数据待补充';
        if (record.unit === 'ratio' || record.unit === 'percent') return `${amount(record, true)}%`;
        return pendingText(record);
      };
      const accuracyTiles = tileSpecs.map(spec => {
        const rows = accuracyRows.filter(row => row.metric === spec.metric);
        const find = stage => rows.find(row => row.model_stage === stage);
        const reference = find('float');
        const quantized = find('quantized');
        const board = find('board');
        const unstaged = rows.filter(row => !row.model_stage);
        const shown = quantized || board || (unstaged.length === 1 ? unstaged[0] : null) || reference;
        const tileNameZh = esc(spec.zh);
        const tileNameEn = esc(spec.en || spec.zh);
        const scale = record => record.unit === 'ratio' ? record.value * 100 : record.value;
        const delta = reference && quantized && ['ratio', 'percent'].includes(reference.unit) && reference.unit === quantized.unit
          ? Number((scale(quantized) - scale(reference)).toFixed(2))
          : null;
        const deltaText = delta === null ? '数据待补充' : `${delta > 0 ? '+' : delta < 0 ? '−' : ''}${Math.abs(delta)}%`;
        const sourced = [reference, quantized, board, ...unstaged].filter(Boolean);
        const comparisonRows = `${infoRow('浮点参考', percentText(reference))}${infoRow('端到端差值', deltaText)}${sourced.length ? provenance(sourced, false) : ''}`;
        const comparisonControl = `<button type="button" class="mz-accuracy-info" aria-label="${tileNameZh}：查看浮点参考" aria-haspopup="dialog">${icon('circle-help')}<span class="mz-accuracy-popover mz-e2e-popover" role="dialog" aria-label="${tileNameZh}对照">${comparisonRows}</span></button>`;
        return `<div class="${shown ? '' : 'is-unmeasured'}"><span class="mz-accuracy-label"><span data-i18n-zh="${tileNameZh}" data-i18n-en="${tileNameEn}">${tileNameZh}</span>${comparisonControl}</span><strong>${shown ? (shown.unit === 'ratio' || shown.unit === 'percent' ? `${amount(shown, true)}<small>%</small>` : tileValue(shown)) : '数据待补充'}</strong></div>`;
      });
      const latencyMatches = records.filter(row => row.metric === 'latency');
      const throughputMatches = records.filter(row => row.metric === 'throughput');
      grid.innerHTML = `${metricTile('Runtime 延迟', delay, delay ? latencyMatches : [])}${endToEndTile}${metricTile('吞吐量', speed, speed ? throughputMatches : [])}${accuracyTiles.join('')}`;
      refresh();
    }

    function performance() {
      if (m.dataPending) return pendingPerformance();
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
        const reference = find('float'), quantized = find('quantized'), board = find('board');
        const metricLabel = metricName(metric, data);
        const tileName = typeof metricLabel === 'string' ? { zh: metricLabel, en: metricLabel } : metricLabel;
        const tileNameZh = esc(tileName.zh || metric);
        const tileNameEn = esc(tileName.en || tileName.zh || metric);
        const scope = metric === 'obb-map-50' ? null : board?.scope || quantized?.scope || reference?.scope;
        const paired = Boolean(reference && quantized);
        const sourceScope = paired ? null : board?.source_scope;
        const scopeNote = [sourceScope, scope].filter(Boolean).map(note =>
          `<small class="mz-accuracy-scope" data-i18n-zh="${esc(note.zh)}" data-i18n-en="${esc(note.en)}">${esc(note.zh)}</small>`,
        ).join('');
        const plain = record => record ? `${amount(record, true)}%` : '—';
        const scale = record => record.unit === 'ratio' ? record.value * 100 : record.value;
        const delta = reference && quantized ? Number((scale(quantized) - scale(reference)).toFixed(2)) : null;
        const big = record => record ? `${amount(record, true)}<small>%</small>` : '—';
        const modelTask = String(m.catalogId || '').split('/').at(-1).toLowerCase();
        const usesEndToEndDelta = ['detect', 'cls', 'seg', 'pose', 'obb'].includes(modelTask);
        const deltaText = delta === null ? '—' : `${delta > 0 ? '+' : delta < 0 ? '−' : ''}${Math.abs(delta)}%`;
        const comparisonRows = `${infoRow('浮点参考', plain(reference))}${infoRow(usesEndToEndDelta ? '端到端差值' : '差距', deltaText)}`;
        const comparisonControl = reference && quantized
          ? `<button type="button" class="mz-accuracy-info" aria-label="${tileNameZh}：查看浮点参考" aria-haspopup="dialog">${icon('circle-help')}<span class="mz-accuracy-popover" role="dialog" aria-label="${tileNameZh}对照">${comparisonRows}</span></button>`
          : '';
        return `<div><span class="mz-accuracy-label"><span data-i18n-zh="${tileNameZh}" data-i18n-en="${tileNameEn}">${tileNameZh}</span>${comparisonControl}</span><strong>${big(quantized || board || reference)}</strong>${scopeNote}</div>`;
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
