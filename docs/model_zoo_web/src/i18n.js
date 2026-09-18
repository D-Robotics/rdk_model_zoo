(() => {
  'use strict';

  const storageKey = 'rdk-model-zoo-language';
  let locale = 'zh';
  try {
    const stored = localStorage.getItem(storageKey);
    if (stored === 'en' || stored === 'zh') locale = stored;
  } catch {
    // Language switching remains available when storage is blocked.
  }

  const messages = {
    '当前目录暂未发布任何模型。': 'No models have been published to this catalog yet.',
    '没有匹配的模型': 'No matching models',
    '试试其他关键词，或清除筛选条件。': 'Try another search term or clear the filters.',
    '主导航': 'Main navigation',
    '跳转到模型': 'Skip to models',
    'OE 转换报告': 'OE conversion reports',
    '暂未发布任何 OE 报告': 'No OE reports have been published',
    '当前尚无可浏览的转换报告。': 'There are no conversion reports to browse yet.',
    '返回模型目录': 'Back to models',
    '返回模型详情': 'Back to model details',
    '报告库': 'Report library',
    '搜索报告': 'Search reports',
    '没有匹配的报告': 'No matching reports',
    '暂无匹配的 OE 报告': 'No matching OE reports',
    '查看全部报告': 'View all reports',
    '下载原始报告': 'Download original report',
    '性能报告': 'Performance report',
    '编译报告': 'Compiler report',
    '筛选模型': 'Filter models',
    '筛选条件': 'Filters',
    '清除筛选': 'Clear filters',
    '搜索模型': 'Search models',
    '任务类型': 'Task type',
    '硬件平台': 'Hardware',
    '尚未收录': 'Not catalogued',
    '模型列表': 'Model list',
    '加载更多': 'Load more',
    '语言模型': 'LLM',
    '机器人': 'Robotics',
    '音频': 'Audio',
    '视觉': 'Vision',
    '多模态': 'Multimodal',
    '目标检测': 'Object detection',
    '图像分类': 'Image classification',
    '实例分割': 'Instance segmentation',
    '语义分割': 'Semantic segmentation',
    '姿态估计': 'Pose estimation',
    '提示式分割': 'Promptable segmentation',
    '深度估计': 'Depth estimation',
    '目标跟踪': 'Object tracking',
    '视觉特征提取': 'Visual embeddings',
    '文字检测': 'Text detection',
    '文字识别': 'Text recognition',
    '文本生成': 'Text generation',
    '机械臂策略': 'Manipulation policies',
    '足式运动控制': 'Legged locomotion',
    '自动驾驶': 'Autonomous driving',
    '车道线检测': 'Lane detection',
    '点云分割': 'Point cloud segmentation',
    '语音识别': 'Speech recognition',
    '关键词唤醒': 'Keyword spotting',
    '图文匹配': 'Image-text similarity',
    '视觉语言模型': 'Vision-language models',
    '开放词汇检测': 'Open-vocabulary detection',
    '子类': ' subtasks',
    '模型导航': 'Model navigation',
    '获取模型': 'Get model',
    '模型仓库': 'Model repository',
    '性能与精度': 'Performance and accuracy',
    '目标硬件': 'Hardware',
    '模型变体': 'Model variant',
    '测试条件': 'Test conditions',
    '暂无性能汇总': 'No performance summary',
    '查看评测说明': 'View evaluation notes',
    '模型参数': 'Model properties',
    '模型文件': 'Model files',
    '部署模型': 'Deployment model',
    '下载': 'Download',
    '目标平台': 'Target platform',
    '开发资源': 'Developer resources',
    '运行文档': 'Runtime documentation',
    '模型转换': 'Model conversion',
    '发布清单': 'Release manifest',
    '模型许可': 'Model license',
    '查看许可说明': 'View license details',
    '相关模型': 'Related models',
    '查看全部模型': 'View all models',
    '参数量': 'Parameters',
    '浮点计算量': 'Float compute',
    '源模型输入': 'Source input',
    '输入尺寸': 'Input size',
    '输入布局': 'Input layout',
    '预测输出': 'Prediction output',
    '关键点数': 'Keypoints',
    '掩码通道数': 'Mask channels',
    '掩码原型': 'Mask prototype',
    '解码器输入': 'Decoder input',
    '图像输入': 'Image input',
    '图像特征': 'Image features',
    '文本输入': 'Text input',
    '文本特征': 'Text features',
    '推理延迟': 'Inference latency',
    '吞吐量': 'Throughput',
    '后处理延迟': 'Post-process latency',
    '单线程': 'Single thread',
  };

  const pattern = new RegExp(
    Object.keys(messages)
      .sort((a, b) => b.length - a.length)
      .map(value => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
      .join('|'),
    'g',
  );

  function english(value) {
    return String(value)
      .replace(/^共 (\d+) 个模型$/, (_, count) => `${count} ${count === '1' ? 'model' : 'models'}`)
      .replace(/^已选 (\d+) 项$/, (_, count) => `${count} selected`)
      .replace(/^已显示 (\d+) \/ (\d+) 个模型$/, (_, shown, total) => `${shown} of ${total} models shown`)
      .replace(/^(\d+) 个文件$/, (_, count) => `${count} ${count === '1' ? 'file' : 'files'}`)
      .replace(/^(\d+) 个报告$/, (_, count) => `${count} ${count === '1' ? 'report' : 'reports'}`)
      .replace(/(\d+) 线程/g, (_, count) => `${count} threads`)
      .replace(pattern, match => messages[match]);
  }

  const originals = new WeakMap();
  function localize(node, key, read, write) {
    let entries = originals.get(node);
    if (!entries) {
      entries = new Map();
      originals.set(node, entries);
    }
    const current = read();
    const previous = entries.get(key);
    const source = previous && current === previous.rendered ? previous.source : current;
    const rendered = locale === 'en' ? english(source) : source;
    if (rendered !== current) write(rendered);
    entries.set(key, { source, rendered });
  }

  function apply() {
    document.documentElement.lang = locale === 'en' ? 'en' : 'zh-CN';
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    let node;
    while ((node = walker.nextNode())) {
      if (node.parentElement?.closest('script, style, [data-language]')) continue;
      localize(node, 'text', () => node.nodeValue, value => { node.nodeValue = value; });
    }
    document.querySelectorAll('[aria-label], [placeholder], [title], img[alt]').forEach(element => {
      if (element.closest('[data-language]')) return;
      for (const attribute of ['aria-label', 'placeholder', 'title', 'alt']) {
        if (element.hasAttribute(attribute)) {
          localize(element, attribute, () => element.getAttribute(attribute), value => element.setAttribute(attribute, value));
        }
      }
    });
    document.querySelectorAll('[data-language]').forEach(button => {
      button.setAttribute('aria-pressed', String(button.dataset.language === locale));
    });
  }

  function setLocale(next) {
    if (next !== 'en' && next !== 'zh') return;
    locale = next;
    try {
      localStorage.setItem(storageKey, next);
    } catch {
      // The active page still switches when storage is blocked.
    }
    apply();
  }

  window.HubI18n = { apply, setLocale, english, get locale() { return locale; } };
})();
