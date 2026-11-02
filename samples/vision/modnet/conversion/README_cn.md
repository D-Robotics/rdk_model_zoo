[English](README.md) | 简体中文

# MODNet 转换

<a id="source-model"></a>
## 源模型

公开摄影人像模型使用 [MODNet 官方源码](https://github.com/ZHKKKe/MODNet/tree/28165a451e4610c9d77cfdf925a94610bb2810fb)，固定提交 `28165a451e4610c9d77cfdf925a94610bb2810fb`。[官方权重目录](https://drive.google.com/drive/folders/1umYmlCulvIFNaqPjwod1SayFmSRHziyR)中的 `modnet_photographic_portrait_matting.ckpt` 文件 ID 为 `1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz`。本次下载观察到 SHA-256 `7c22235f0925deba15d4d63e53afcb654c47055bbcd98f56e393ab2584007ed8`，大小 26,255,603 字节。作者未提供发布方校验值；观察摘要用于固定本次转换的权重字节。

[官方许可声明](https://github.com/ZHKKKe/MODNet/blob/28165a451e4610c9d77cfdf925a94610bb2810fb/README.md#license)明确代码、模型和演示使用 Apache-2.0，`doc/gif` 除外。保留上游 LICENSE 和版权说明。

<a id="directory"></a>
## 目录结构

`prepare.py` 导出官方单 matte 推理模型，与原始 PyTorch 推理对照，验证校准图并在外部目录生成 X5 配置。模型二进制、权重及校准张量不入库。

<a id="toolchain-targets"></a>
## 工具链与目标

仅 X5：OpenExplorer 镜像 `registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_20_x5_gpu:v1.2.8`，观察镜像 ID `sha256:bb706c129bafb4d36d2c338c957369cc5fdb51e0eb65ad5c927b3fda4023ad3f`，Mapper 1.24.3。导出使用 PyTorch 2.8.0、ONNX 1.17.0、ONNX Runtime 1.19.2。本流程为 PTQ 转换，板端性能与抠像精度另行补测。

<a id="export"></a>
## 导出与校准准备

在外部工作目录获取官方仓库并切换到固定提交，下载上述官方权重，保持源仓库干净。准备固定的 COCO person train2017 200 张校准图（seed 42），`manifest.json` 包含 `count`、`seed`、`images`，每张包含 `name`、`sha256`。

在已具备依赖的环境中，从仓库根目录运行：

```bash
python3 samples/vision/modnet/conversion/prepare.py \
  --source-dir /work/MODNet \
  --checkpoint /work/modnet_photographic_portrait_matting.ckpt \
  --checkpoint-sha256 7c22235f0925deba15d4d63e53afcb654c47055bbcd98f56e393ab2584007ed8 \
  --calibration-dir /work/coco-person-keypoints-train2017-seed42-200 \
  --output /work/modnet-run/prepared
```

输出目录必须全新。ONNX 固定 opset 11、batch 1，float32 输入 `input` `[1,3,512,512]`，matte 输出 `matte` `[1,1,512,512]`。官方 ONNX 模块省略训练时使用的 semantic/detail 输出。原始 `MODNet.forward(x, inference=True)[2]` 为对照；三张真实输入与固定随机输入必须有限，并通过 PyTorch/ORT `atol=1e-4`、`rtol=1e-3` 检查。

<a id="calibration"></a>
## 校准契约

复用 Runtime 的 `preprocess_image`：BGR→RGB，`(pixel-127.5)/127.5`，按长边用 `INTER_AREA` 缩放，居中补零到 512，输出 float32 NCHW。逐张核验源图摘要，记录张量摘要和统计，回读三张张量做图对照。YAML 使用 `featuremap`、NCHW、`no_preprocess`，归一化只执行一次。人物图用于首轮 PTQ，正式抠像精度需独立标注集。

<a id="compile"></a>
## 编译

在匹配的 X5 环境中挂载所有外部路径后执行：

```bash
hb_mapper checker --model-type onnx --march bayes-e \
  --model /work/modnet-run/prepared/modnet_photographic_portrait_512x512_rgb_bayese_20261102.onnx
hb_mapper makertbin --model-type onnx --config /work/modnet-run/prepared/config.yaml
hb_model_info /work/modnet-run/output/modnet_photographic_portrait_512x512_rgb_bayese_20261102.bin
```

配置固定 `bayes-e`、default 校准、O3 latency、单 BPU 核和两个编译进程。float32 `featuremap` 输入必须使用 `input_source: {input: ddr}`，不能套用 `pyramid`。每次尝试使用全新 working 目录，保留配置和日志。

<a id="validation"></a>
## 转换后验证

核查 checker/compiler 成功、唯一 BIN、`BPU march: bayes-e` 及实际 float32 NCHW 输入和 matte 输出元数据。保留原始 OE、节点 cosine、静态编译报告；这些不是板端速度或抠像精度。Runtime 以精确 asset ID 绑定新制品，并在加载时校验元数据。

<a id="artifacts"></a>
## 产物

新文件为 `modnet_photographic_portrait_512x512_rgb_bayese_20261102.bin`，与旧手动制品 `modnet_512x512_rgb.bin` 保持独立。保留 `prepare-report.json`、`calibration-manifest.json`、`preprocess-report.md`、YAML、ONNX、权重身份、编译日志、model-info 与模型摘要。

<a id="known-gaps"></a>
## 历史模型与待测项

旧手动制品 `modnet_512x512_rgb.bin` 的权重/导出/PTQ 身份仍未知，其历史性能表继续保留，不用于新摄影人像模型。新制品的板端性能、板端 Runtime 执行和标注集抠像精度待补测。
