import assert from 'node:assert/strict';
import { test } from 'node:test';
import { isPublishedDownloadSource } from '../model-download-policy.mjs';

test('published model downloads accept model-zoo-web OSS and official LLM OSS only', () => {
  for (const url of [
    'https://model-zoo-web.oss-cn-beijing.aliyuncs.com/rdk_s100/pointnet/model.hbm',
    'https://model-zoo-web.oss-cn-beijing.aliyuncs.com/rdk_x5/unet/model.bin',
    'https://model-zoo-web.oss-cn-beijing.aliyuncs.com/rdk_s600/clip/model_bundle.tar.gz',
    'https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm/2.0.0/s600_models/Gemma4-E2B_w4.zip',
    'https://model-zoo-web.oss-cn-beijing.aliyuncs.com/rdk_s600/gemma4/tokenizer/tokenizer.json',
  ]) assert.equal(isPublishedDownloadSource(url), true, url);
  for (const url of [
    'https://archive.d-robotics.cc/downloads/rdk_model_zoo_web/rdk_s100/pointnet/model.hbm',
    'https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/pointnet/model.hbm',
    'https://model-zoo-web.oss-cn-beijing.aliyuncs.com/assets/rdk_s100/pointnet/model/release.json',
    'https://codeload.github.com/D-Robotics/mot/tar.gz/0120205146d88c3474bfc2936522e349e169fa8d',
    'https://model-zoo-web.oss-cn-beijing.aliyuncs.com.example.org/rdk_x5/unet/model.bin',
    'https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/other/model.zip',
    'http://model-zoo-web.oss-cn-beijing.aliyuncs.com/rdk_x5/unet/model.bin',
    'https://model-zoo-web.oss-cn-beijing.aliyuncs.com/rdk_x5/unet/model.bin?signature=temporary',
    'not a URL', '', null,
  ]) assert.equal(isPublishedDownloadSource(url), false, String(url));
});
