import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { buildPendingModels } from '../pending-preview.mjs';

const taskRegistry = JSON.parse(readFileSync(fileURLToPath(new URL('../../task-registry.json', import.meta.url)), 'utf8'));
const safeId = (...parts) => parts.join('-').toLowerCase().replace(/[^a-z0-9-]+/g, '-').replace(/-+/g, '-');

function page(record) {
  return buildPendingModels([{
    registration: 'pending',
    sample_revision: '7a97c1780b0cfa131ff0007d662e255e5e41c12e',
    sample_path: 'samples/vision/example',
    description: { zh: '待发布', en: 'pending' },
    usage: [],
    ...record,
  }], {
    officialRepositoryUrl: 'https://github.com/D-Robotics/rdk_model_zoo',
    taskRegistry,
    safeId,
  });
}

const file = (platform, filename) => ({
  platform,
  filename,
  url: `https://archive.d-robotics.cc/downloads/rdk_model_zoo/${filename}`,
});

test('resolution and size tokens keep each published file on its own variant', () => {
  const models = page({
    id: 'siglip/siglip/embed',
    name: 'SigLIP',
    source: 'siglip',
    family: 'siglip',
    task: 'embed',
    supported_platforms: [{ platform: 's100' }, { platform: 's100p' }],
    published_variants: [
      { name: 'base-patch16-224' },
      { name: 'base-patch16-384' },
      { name: 'base-patch16-512' },
      { name: 'large-patch16-256' },
    ],
    sourced_benchmarks: [
      { id: 'a', platform: 's100', variant: 'SigLIP base patch16 224 on RDK S100', metric: 'latency', value: 26.8, unit: 'ms', source: 'docs/release/s/benchmarks.yaml' },
      { id: 'b', platform: 's100', variant: 'SigLIP base patch16 384 on RDK S100', metric: 'latency', value: 46.7, unit: 'ms', source: 'docs/release/s/benchmarks.yaml' },
      { id: 'c', platform: 's100', variant: 'SigLIP base 512 on RDK S100', metric: 'latency', value: 81.7, unit: 'ms', source: 'docs/release/s/benchmarks.yaml' },
      { id: 'd', platform: 's100', variant: 'SigLIP large 256 on RDK S100', metric: 'latency', value: 68.8, unit: 'ms', source: 'docs/release/s/benchmarks.yaml' },
      { id: 'e', platform: 's100p', variant: 'SigLIP base patch16 224 on RDK S100P', metric: 'latency', value: 18.8, unit: 'ms', source: 'docs/release/s/benchmarks.yaml' },
    ],
    documented_downloads: [
      file('s100', 's100/bpu-siglip-base-patch16-224.hbm'),
      file('s100', 's100/bpu-siglip-base-patch16-384.hbm'),
      file('s100', 's100/bpu-siglip-base-patch16-512.hbm'),
      file('s100', 's100/bpu-siglip-large-patch16-256.hbm'),
    ],
  });
  const urls = variant => models.find(model => model.releasePlatform === 'S100' && model.variantName === variant)
    .documentedDownloads.map(item => item.filename);
  assert.deepEqual(urls('SigLIP base patch16 224 on RDK S100'), ['s100/bpu-siglip-base-patch16-224.hbm']);
  assert.deepEqual(urls('SigLIP base patch16 384 on RDK S100'), ['s100/bpu-siglip-base-patch16-384.hbm']);
  assert.deepEqual(urls('SigLIP base 512 on RDK S100'), ['s100/bpu-siglip-base-patch16-512.hbm']);
  assert.deepEqual(urls('SigLIP large 256 on RDK S100'), ['s100/bpu-siglip-large-patch16-256.hbm']);
  assert.equal(models.filter(model => model.releasePlatform === 'S100' && model.variantName.startsWith('base-')).length, 0);
  assert.equal(models.find(model => model.variantName.includes('S100P')).documentedDownloads.length, 0);

  const dino = page({
    id: 'dinov2/dinov2/embed',
    name: 'DINOv2',
    source: 'dinov2',
    family: 'dinov2',
    task: 'embed',
    supported_platforms: [{ platform: 's100' }],
    published_variants: [{ name: 'vits14-224-int16' }],
    sourced_benchmarks: [
      { id: 'd', platform: 's100', variant: 'DINOv2 ViT-S/14 on RDK S100', metric: 'latency', value: 3.73, unit: 'ms', source: 'docs/release/s/benchmarks.yaml' },
    ],
    documented_downloads: [file('s100', 'nash-e/dinov2_vits14_224_int16_nashe.hbm')],
  });
  assert.equal(dino.length, 1);
  assert.equal(dino[0].variantName, 'DINOv2 ViT-S/14 on RDK S100');
  assert.deepEqual(dino[0].documentedDownloads.map(item => item.filename), ['nash-e/dinov2_vits14_224_int16_nashe.hbm']);
  assert.equal(dino[0].benchmark.performance[0].value, 3.73);
});

test('YOLOE sizes, MobileSAM decoder, and UNet VOC bind one file each', () => {
  const yolo = page({
    id: 'yoloe/yoloe26/seg',
    name: 'YOLOE-26',
    source: 'yoloe',
    family: 'yoloe26',
    task: 'seg',
    supported_platforms: [{ platform: 's100' }],
    published_variants: ['26n', '26s', '26m', '26l', '26x'].map(name => ({ name })),
    documented_downloads: ['n', 's', 'm', 'l', 'x'].map(size => file('s100', `nash-e/yoloe_26${size}_seg_pf_nashe_640x640_nv12.hbm`)),
  });
  assert.deepEqual(yolo.map(model => [model.variantName, model.documentedDownloads.map(item => item.filename)]), [
    ['26n', ['nash-e/yoloe_26n_seg_pf_nashe_640x640_nv12.hbm']],
    ['26s', ['nash-e/yoloe_26s_seg_pf_nashe_640x640_nv12.hbm']],
    ['26m', ['nash-e/yoloe_26m_seg_pf_nashe_640x640_nv12.hbm']],
    ['26l', ['nash-e/yoloe_26l_seg_pf_nashe_640x640_nv12.hbm']],
    ['26x', ['nash-e/yoloe_26x_seg_pf_nashe_640x640_nv12.hbm']],
  ]);

  const sam = page({
    id: 'mobile_sam/mobile_sam/prompt-seg',
    name: 'MobileSAM',
    source: 'mobile_sam',
    family: 'mobile_sam',
    task: 'prompt-seg',
    supported_platforms: [{ platform: 'x5' }],
    sourced_benchmarks: [
      { id: 'enc', platform: 'x5', variant: 'MobileSAM Encoder', metric: 'latency', value: 1, unit: 'ms', source: 'docs/release/x5/benchmarks.yaml' },
      { id: 'dec', platform: 'x5', variant: 'MobileSAM Box-prompt Decoder', metric: 'latency', value: 2, unit: 'ms', source: 'docs/release/x5/benchmarks.yaml' },
    ],
    documented_downloads: [
      file('x5', 'mobile_sam_image_encoder_norm_512x512_allint16.bin'),
      file('x5', 'mobile_sam_decoder_512_box_default.bin'),
    ],
  });
  assert.deepEqual(sam.find(model => model.variantName.includes('Encoder')).documentedDownloads.map(item => item.filename), ['mobile_sam_image_encoder_norm_512x512_allint16.bin']);
  assert.deepEqual(sam.find(model => model.variantName.includes('Decoder')).documentedDownloads.map(item => item.filename), ['mobile_sam_decoder_512_box_default.bin']);

  const unet = page({
    id: 'unet/unet/semseg',
    name: 'UNet',
    source: 'unet',
    family: 'unet',
    task: 'semseg',
    supported_platforms: [{ platform: 'x5' }],
    sourced_benchmarks: [
      { id: 'voc', platform: 'x5', variant: 'UNet ResNet18 Pascal VOC 512x512', metric: 'latency', value: 52.72, unit: 'ms', source: 'docs/release/x5/benchmarks.yaml' },
      { id: 'r34', platform: 'x5', variant: 'UNet ResNet34', metric: 'miou', value: 0.2, unit: 'ratio', source: 'docs/release/x5/benchmarks.yaml' },
      { id: 'hist', platform: 'x5', variant: 'UNet ResNet18 Historical PyTorch FP32', metric: 'miou', value: 0.1, unit: 'ratio', source: 'docs/release/x5/benchmarks.yaml' },
    ],
    documented_downloads: [file('x5', 'unet_resnet18_voc_512x512_nv12.bin'), file('x5', 'unet_resnet34_voc_512x512_nv12.bin')],
  });
  assert.deepEqual(unet.find(model => model.variantName.includes('Pascal')).documentedDownloads.map(item => item.filename), ['unet_resnet18_voc_512x512_nv12.bin']);
  assert.deepEqual(unet.find(model => model.variantName.includes('ResNet34')).documentedDownloads.map(item => item.filename), ['unet_resnet34_voc_512x512_nv12.bin']);
  assert.equal(unet.find(model => model.variantName.includes('Historical')).documentedDownloads.length, 0);
});

test('multi-thread rows join the sibling and keep both latency conditions', () => {
  const models = page({
    id: 'repvgg/repvgg/cls',
    name: 'RepVGG',
    source: 'repvgg',
    family: 'repvgg',
    task: 'cls',
    supported_platforms: [{ platform: 'x5' }],
    sourced_benchmarks: [
      { id: 'base', platform: 'x5', variant: 'RepVGG B1g2 224x224', metric: 'latency', value: 9.77, unit: 'ms', condition: 'single-frame, single-thread, single-BPU-core', source: 'docs/release/x5/benchmarks.yaml' },
      { id: 'fps', platform: 'x5', variant: 'RepVGG B1g2 224x224', metric: 'throughput', value: 109.61, unit: 'fps', condition: 'multi-thread', source: 'docs/release/x5/benchmarks.yaml' },
      { id: 'mt', platform: 'x5', variant: 'RepVGG B1g2 224x224 Multi-thread', metric: 'latency', value: 36.19, unit: 'ms', condition: 'multi-thread', source: 'docs/release/x5/benchmarks.yaml' },
      { id: 'a0', platform: 'x5', variant: 'RepVGG A0 224x224', metric: 'latency', value: 1.85, unit: 'ms', condition: 'single-frame, single-thread, single-BPU-core', source: 'docs/release/x5/benchmarks.yaml' },
    ],
    documented_downloads: [
      file('x5', 'RepVGG_B1g2_224x224_nv12.bin'),
      file('x5', 'RepVGG_A0_224x224_nv12.bin'),
    ],
  });
  assert.equal(models.some(model => /multi-thread/i.test(model.variantName)), false);
  const b1 = models.find(model => model.variantName === 'RepVGG B1g2 224x224');
  assert.deepEqual(b1.documentedDownloads.map(item => item.filename), ['RepVGG_B1g2_224x224_nv12.bin']);
  assert.deepEqual(b1.benchmark.performance.map(row => [row.metric, row.value, row.condition]), [
    ['latency', 9.77, 'single-frame, single-thread, single-BPU-core'],
    ['throughput', 109.61, 'multi-thread'],
    ['latency', 36.19, 'multi-thread'],
  ]);
  assert.deepEqual(models.find(model => model.variantName.includes('A0')).documentedDownloads.map(item => item.filename), ['RepVGG_A0_224x224_nv12.bin']);
});

test('EdgeNeXt sizes do not share bins, and Gemma companions stay on both dialogs', () => {
  const edge = page({
    id: 'edgenext/edgenext/cls',
    name: 'EdgeNeXt',
    source: 'edgenext',
    family: 'edgenext',
    task: 'cls',
    supported_platforms: [{ platform: 'x5' }],
    sourced_benchmarks: ['Small', 'X-Small', 'XX-Small'].map(size => ({
      id: size, platform: 'x5', variant: `EdgeNeXt ${size} 224x224`, metric: 'latency', value: 1, unit: 'ms', source: 'docs/release/x5/benchmarks.yaml',
    })),
    documented_downloads: [
      file('x5', 'EdgeNeXt_small_224x224_nv12.bin'),
      file('x5', 'EdgeNeXt_x_small_224x224_nv12.bin'),
      file('x5', 'EdgeNeXt_xx_small_224x224_nv12.bin'),
    ],
  });
  const names = model => model.documentedDownloads.map(item => item.filename);
  assert.deepEqual(names(edge.find(model => model.variantName.includes('XX-Small'))), ['EdgeNeXt_xx_small_224x224_nv12.bin']);
  assert.deepEqual(names(edge.find(model => model.variantName.includes('X-Small') && !model.variantName.includes('XX-'))), ['EdgeNeXt_x_small_224x224_nv12.bin']);
  assert.deepEqual(names(edge.find(model => model.variantName.includes('Small') && !model.variantName.includes('X-'))), ['EdgeNeXt_small_224x224_nv12.bin']);

  const gemma = page({
    id: 'gemma4-e2b/gemma4-e2b/vlm',
    name: 'Gemma4-E2B',
    source: 'gemma4-e2b',
    family: 'gemma4-e2b',
    task: 'vlm',
    supported_platforms: [{ platform: 's100p' }, { platform: 's600' }],
    documented_downloads: [
      file('s100p', 's100p/gemma4-e2b_vit_ptq.hbm'),
      file('s600', 's600/gemma4-e2b_vit_ptq.hbm'),
      file('s100p', 'common/tok_embeddings.bin'),
      file('s100p', 'common/tokenizer/tokenizer.json'),
      file('s600', 'common/tok_embeddings.bin'),
      file('s600', 'common/tokenizer/tokenizer.json'),
    ],
  });
  for (const model of gemma) {
    assert.deepEqual(model.documentedDownloads.map(item => item.filename).sort(), [
      'common/tok_embeddings.bin',
      'common/tokenizer/tokenizer.json',
      model.releasePlatform === 'S100P' ? 's100p/gemma4-e2b_vit_ptq.hbm' : 's600/gemma4-e2b_vit_ptq.hbm',
    ].sort());
  }
});

test('FCOS thread rows stay without a sibling bin', () => {
  const models = page({
    id: 'fcos/fcos/detect',
    name: 'FCOS',
    source: 'fcos',
    family: 'fcos',
    task: 'detect',
    supported_platforms: [{ platform: 'x5' }],
    published_variants: [{ name: 'efficientnetb0 / 512' }],
    sourced_benchmarks: [
      { id: 'base', platform: 'x5', variant: 'FCOS EfficientNet B0 512x512', metric: 'latency', value: 1, unit: 'ms', source: 'docs/release/x5/benchmarks.yaml' },
      { id: 'one', platform: 'x5', variant: 'FCOS EfficientNet B0 512x512 Single Thread', metric: 'latency', value: 3.3, unit: 'ms', source: 'docs/release/x5/benchmarks.yaml' },
      { id: 'two', platform: 'x5', variant: 'FCOS EfficientNet B0 512x512 Two Threads', metric: 'latency', value: 6.2, unit: 'ms', source: 'docs/release/x5/benchmarks.yaml' },
    ],
    documented_downloads: [file('x5', 'fcos_efficientnetb0_detect_512x512_bayese_nv12.bin')],
  });
  assert.equal(models.some(model => model.variantName === 'efficientnetb0 / 512'), false);
  assert.deepEqual(models.find(model => model.variantName === 'FCOS EfficientNet B0 512x512').documentedDownloads.map(item => item.filename), ['fcos_efficientnetb0_detect_512x512_bayese_nv12.bin']);
  assert.equal(models.find(model => model.variantName.includes('Single Thread')).documentedDownloads.length, 0);
  assert.equal(models.find(model => model.variantName.includes('Two Threads')).documentedDownloads.length, 0);
  assert.equal(models.find(model => model.variantName.includes('Single Thread')).benchmark.performance[0].value, 3.3);
});
