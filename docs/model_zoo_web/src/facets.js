(() => {
  'use strict';
  // Task IDs follow rdk_model_zoo/docs/catalog/src/catalog/task-groups.ts.
  const groups = [
    { id: 'vision', label: '视觉', tasks: [
      ['object-detection', '目标检测'], ['image-classification', '图像分类'],
      ['instance-segmentation', '实例分割'], ['semantic-segmentation', '语义分割'],
      ['pose-estimation', '姿态估计'], ['promptable-image-segmentation', '提示式分割'],
      ['monocular-depth-estimation', '深度估计'], ['multi-object-tracking', '目标跟踪'],
      ['image-embedding', '视觉特征提取'], ['ocr-text-detection', '文字检测'],
      ['ocr-text-recognition', '文字识别'],
    ] },
    { id: 'llm', label: '语言模型', tasks: [['text-generation', '文本生成']] },
    { id: 'robotics', label: '机器人', tasks: [
      ['robot-manipulation-policy', '机械臂策略'], ['legged-locomotion-control', '足式运动控制'],
      ['autonomous-driving', '自动驾驶'], ['lane-detection', '车道线检测'],
      ['point-cloud-segmentation', '点云分割'],
    ] },
    { id: 'audio', label: '音频', tasks: [['speech-recognition', '语音识别'], ['keyword-spotting', '关键词唤醒']] },
    { id: 'multimodal', label: '多模态', tasks: [
      ['image-text-similarity', '图文匹配'], ['vision-language-model', '视觉语言模型'],
      ['open-vocabulary-object-detection', '开放词汇检测'],
    ] },
  ];
  function taskIds(model) {
    if (Array.isArray(model.tasks)) return model.tasks;
    if (model.taskId) return [model.taskId];
    return [];
  }
  function normalizePrecision(value) {
    const normalized = value.trim().toLowerCase().replace(/[ _-]/g, '');
    if (['int8', 'qint8', 's8'].includes(normalized)) return 'int8';
    if (['int16', 'qint16', 's16'].includes(normalized)) return 'int16';
    if (['float', 'float16', 'float32', 'float64', 'fp16', 'fp32', 'fp64', 'bf16', 'bfloat16'].includes(normalized)) return 'float';
    return value.trim().toLowerCase();
  }
  function quantizations(model) {
    // Precision is independent of file format, I/O dtype and accuracy.model_stage.
    const records = [model.benchmark, ...(model.benchmarks || [])].filter(Boolean);
    const values = records.map(record => record.precision).filter(value => typeof value === 'string' && value.trim());
    return values.length ? [...new Set(values.map(normalizePrecision))] : ['unspecified'];
  }
  function searchTerms(model) {
    const ids = taskIds(model);
    return groups.filter(group => group.tasks.some(([id]) => ids.includes(id)))
      .map(group => [group.id, group.label, ...group.tasks.filter(([id]) => ids.includes(id)).flat()].join(' ')).join(' ');
  }
  window.CatalogFacets = { groups, taskIds, quantizations, searchTerms };
})();
