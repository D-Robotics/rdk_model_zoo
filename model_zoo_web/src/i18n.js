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
    'Language / 语言': 'Language',
    '获取模型': 'Get model',
    '选择芯片': 'Select chip',
    '选择模型类型': 'Select model type',
    '下载文件': 'Download file',
    '关闭': 'Close',
    '取消': 'Cancel',
    '模型仓库': 'Model repository',
    '性能与精度': 'Performance and accuracy',
    '目标硬件': 'Hardware',
    '模型变体': 'Model variant',
    '测试条件': 'Test conditions',
    'Runtime 并发': 'Runtime concurrency',
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
    '未声明': 'Not declared',
    '模型参考推理结果': 'Model reference inference result',
    '相关模型': 'Related models',
    '查看全部模型': 'View all models',
    '参数量': 'Parameters',
    '部署模型大小': 'Model size',
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
    'Runtime 输入': 'Runtime input',
    'Runtime 延迟': 'Runtime latency',
    '端到端延迟': 'End-to-end latency',
    '端到端吞吐量': 'E2E throughput',
    '端到端流水线条件': 'End-to-end pipeline conditions',
    '查看完整流水线条件': 'View full pipeline conditions',
    '查看浮点参考': 'View float reference',
    '前处理': 'Pre-processing',
    '后处理': 'Post-processing',
    '浮点参考': 'Float reference',
    '差距': 'Delta',
    'CPU 线程': 'CPU threads',
    '流水线': 'Pipeline',
    'Runtime 提交': 'Runtime submissions',
    '实现': 'Implementation',
    '单路': 'Single stream',
    '未测': 'Not measured',
    '未测量': 'Not measured',
    '推理延迟': 'Inference latency',
    '吞吐量': 'Throughput',
    '后处理延迟': 'Post-process latency',
    '单线程': 'Single thread',
    '查看转换详情': 'View conversion details',
    '加载中…': 'Loading…',
    '概览': 'Overview',
    '逐层量化': 'Layer-wise quantization',
    '原始数据': 'Raw data',
    '总延迟': 'Total latency',
    '推导 FPS': 'Derived FPS',
    '每帧 DDR': 'DDR per frame',
    '估算延迟': 'Estimated latency',
    '估算 FPS': 'Estimated FPS',
    '估算 DDR': 'Estimated DDR',
    '延迟': 'Latency',
    '编译器估算': 'Compiler estimate',
    '板端实测': 'On-device measurement',
    '逐区间利用率': 'Per-interval utilization',
    '估算 / 实测延迟': 'Est. / Measured latency',
    '估算 / 实测 FPS': 'Est. / Measured FPS',
    'BPU 推理': 'BPU inference',
    'CPU 反量化': 'CPU dequantization',
    '合计': 'Total',
    'BPU OPs': 'BPU OPs',
    '量化节点': 'Quantized nodes',
    '最弱相似度': 'Lowest similarity',
    '层': 'layers',
    '模型输入': 'Model inputs',
    '模型输出': 'Model outputs',
    '工具链': 'Toolchain',
    '节点': 'Node',
    '子图': 'Subgraph',
    '执行单元：全部': 'Execution unit: all',
    '执行单元': 'Execution unit',
    '算子：全部': 'Op type: all',
    '算子类型': 'Op type',
    '搜索节点': 'Search nodes',
    '余弦相似度': 'Cosine similarity',
    '阈值': 'Threshold',
    '数据类型': 'Data type',
    '输出级对比': 'Output-level comparison',
    'L1 距离': 'L1 distance',
    'L2 距离': 'L2 distance',
    '切比雪夫距离': 'Chebyshev distance',
    '输出': 'Output',
    '查看完整 JSON': 'View full JSON',
    '报告数据缺失': 'Report data missing',
    '暂无逐层数据': 'No layer-wise data',
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
      .replace(/^COCO 目标检测模型，面向 RDK (.+) 的 INT8 (.+) 部署版本。$/, (_, platform, input) => `COCO object detection model with an INT8 ${input} deployment for RDK ${platform}.`)
      .replace(/RDK ([^·]+) 板端推理结果$/, (_, platform) => `RDK ${platform} on-device inference result`)
      .replace(/^Runtime 单路$/, 'Single Runtime stream')
      .replace(/^Runtime (\d+) 路并发$/, (_, count) => `${count} concurrent Runtime streams`)
      .replace(/^(\d+) CPU 线程$/, (_, count) => `${count} CPU threads`)
      .replace(/^(\d+) 路$/, (_, count) => `${count} streams`)
      .replace(/^(.+)：查看浮点参考$/, (_, name) => `${name}: view float reference`)
      .replace(/^(.+)对照$/, (_, name) => `${name} comparison`)
      .replace(/^查看 (.+)$/, (_, name) => `View ${name}`)
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
    document.querySelectorAll('[data-i18n-zh][data-i18n-en]').forEach(element => {
      const value = element.getAttribute(locale === 'en' ? 'data-i18n-en' : 'data-i18n-zh');
      if (element.textContent !== value) element.textContent = value;
      if (element.hasAttribute('title')) element.setAttribute('title', value);
    });
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    let node;
    while ((node = walker.nextNode())) {
      if (node.parentElement?.closest('script, style, [data-language], [data-i18n-zh][data-i18n-en]')) continue;
      localize(node, 'text', () => node.nodeValue, value => { node.nodeValue = value; });
    }
    document.querySelectorAll('[aria-label], [placeholder], [title], img[alt]').forEach(element => {
      if (element.closest('[data-language], [data-i18n-zh][data-i18n-en]')) return;
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
