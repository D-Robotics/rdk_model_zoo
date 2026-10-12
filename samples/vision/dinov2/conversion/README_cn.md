[English](README.md) | 简体中文

# 模型转换 — DINOv2 ViT-S/14

按下文命令将 DINOv2 导出为 ONNX、准备校准图像，并编译所选 Nash 目标的 HBM。

<a id="source-model"></a>
## 源模型

- 框架：OE 3.7.0 环境中的 PyTorch（Torch 2.6）；导出器支持 Torch >=2.4，使用 `dynamo=False` 的 legacy exporter。
- 源仓库：`https://github.com/facebookresearch/dinov2`，revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8`。
- Checkpoint：`dinov2_vits14_pretrain.pth`，SHA-256 为 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`。
- 导出器在 `torch.load` 前校验完整 checkpoint digest；HTTPS checkpoint 会流式写入临时文件，校验/加载后删除。

<a id="directory"></a>
## 目录结构

```text
conversion/
├── onnx_export/  # onnx_export 相关文件
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
└── mapper.py  # Python 脚本
```

<a id="toolchain-targets"></a>
## 工具链与目标

转换使用 OpenExplore 3.7.0、hmct 2.6.5 / hbdk 4.7.5 和三个 Nash march。OE 镜像提供 Torch 2.6；只安装固定版本的 ONNX 增量包。转换在 x86 Linux 上进行，不在板端运行。

| Target | march | OE 版本 | 配置 |
| --- | --- | --- | --- |
| s100 | `nash-e` | 3.7.0 | `conversion/mapper.py` 生成 |
| s100p | `nash-m` | 3.7.0 | `conversion/mapper.py` 生成 |
| s600 | `nash-p` | 3.7.0 | `conversion/mapper.py` 生成 |

OE 入口是下面的 x86 Linux 容器命令：

```bash
# cwd：仓库根目录；源树挂载到 /workspace
sudo docker run -it --rm --network host --shm-size=15g \
  -v "$(pwd)":/workspace -w /workspace \
  registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0 \
  /bin/bash
```

### 准备指定版本的源码与权重

以下在 OE 容器内执行，工作目录为 `samples/vision/dinov2/conversion`。需要网络获取上游材料。新建的 `dinov2/` 为上游源码检出目录。保留 OE 已有 Torch，仅安装固定 ONNX 包：

```bash
# cwd: samples/vision/dinov2/conversion, inside OE 3.7.0
# 1. The OE 3.7 image provides Torch 2.6; preserve it. The exporter supports
# Torch >=2.4 and this setup was validated with these ONNX packages.
python3 -c "import torch; print(torch.__version__)"
python3 -m pip install --upgrade onnx==1.19.0 onnxruntime==1.23.2
python3 -c "import onnx, onnxruntime; print(onnx.__version__, onnxruntime.__version__)"

# 2. Fetch the pinned source and official Apache-2.0 checkpoint.
git clone https://github.com/facebookresearch/dinov2.git
cd dinov2
git checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8
cd ..
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
printf '%s  %s\n' \
  b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9 \
  dinov2_vits14_pretrain.pth | sha256sum -c -

```

<a id="export"></a>
## 导出（ONNX）

`onnx_export/export_dinov2.py` 校验固定版本的源代码和权重文件，构建 ViT-S/14，固定 224 输入尺寸的位置编码，将 attention 重写为显式 MatMul + Softmax + MatMul，并导出输入 `input` `(1,3,224,224)` float32 和两个输出 `cls_feat`、`patch_feat`。导出后还会运行 ONNXRuntime 输出比较。执行：

```bash
# cwd：samples/vision/dinov2/conversion；前置：OE 环境、本地源 clone、已校验 checkpoint
python3 onnx_export/export_dinov2.py \
  --weights ./dinov2_vits14_pretrain.pth \
  --weights-sha256 b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9 \
  --repo ./dinov2 \
  --repo-revision 7764ea0f912e53c92e82eb78a2a1631e92725fc8 \
  --out ./dinov2_vits14_224.onnx
# 预期：ONNX input=(1,3,224,224) F32；输出 cls_feat=(1,384)、patch_feat=(1,256,384)，并打印对拍日志
```

<a id="calibration"></a>
## 校准

`mapper.py::prepare_calibration` 从 `--cal-images` 读取 JPG/JPEG/PNG/BMP，将 BGR 转 RGB，bicubic 按比例把短边 resize 到 256，中心 crop 224，执行 `/255` 和 ImageNet mean/std，写出 contiguous float32 NCHW `.npy` tensor，并在临时 workspace 写 `calibration_manifest.json`。源推荐 50 张多样真实照片；随机或合成图片不适合此模型。

使用 featuremap float32 输入、全节点 int16 和 int16 模型输出，`calibration_type` 保持未设置，以使用 hmct 默认的 modelwise KL 校准。下表给出这一配置与 int8、max/percentile 配置的参考测量。

```bash
# cwd：samples/vision/dinov2/conversion；前置：./cal_images 有真实照片
# 完整 mapper 命令会生成校准数据；不需要单独下载命令。
python3 mapper.py --cal-images ./cal_images --march nash-e --output-dir ./output
# 预期：hb_compile 前出现临时 calibration_data_norm/*.npy 和 calibration_manifest.json
```


### 量化配置对照（Nash-E 参考数据）

下表比较五种量化配置，`(sim)` 表示工具链模拟值。在这些测量中，int8 激活的余弦相似度最高为 0.91，max+percentile 校准约为 0.18；attention logits 减去最大值后的张量最小达到 -345。请使用上述全 int16 featuremap/KL 配置。

| Config | cls cosine | patch cosine | Verdict |
|---|---|---|---|
| int8 + softmax-int32, featuremap, max | 0.081 | 0.803 | FAIL |
| int8 + softmax-int32, featuremap, KL | 0.892 | 0.894 | FAIL |
| int16, featuremap, max + 0.9999 | 0.184 | 0.840 | FAIL |
| int16, nv12, KL | 0.999 (sim) | 0.999 (sim) | FAIL (executed 0.01 / 0.12) |
| int16, featuremap, KL (this recipe) | **0.9989** | **0.9983** | **PASS** |

`_reg4` register-token 变体在逐张量量化下的余弦相似度约为 0.80，普通变体约为 0.999。本配方使用普通 ViT-S/14 权重。

<a id="compile"></a>
## 编译

`mapper.py` 按顺序执行导出、校准准备、YAML 生成、`hb_compile` 和制品收集。它先检查 `hb_compile --help`，生成目标配置，使用 `featuremap` 输入和全 int16 量化，最后把 HBM 复制到 `--output-dir`。

```bash
# cwd：samples/vision/dinov2/conversion；在 OE 3.7.0 镜像中运行
python3 mapper.py \
  --weights ./dinov2_vits14_pretrain.pth \
  --weights-sha256 b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9 \
  --repo ./dinov2 \
  --repo-revision 7764ea0f912e53c92e82eb78a2a1631e92725fc8 \
  --cal-images ./cal_images \
  --march nash-e \
  --output-dir ./output \
  --jobs 16
# 预期：output/dinov2_vits14_224_int16_nashe.hbm，以及生成时的 hb_compile_nashe.log
```

S100P 或 S600 使用 `--march nash-m` 或 `nash-p`。mapper 其他默认值为 `--weights./dinov2_vits14_pretrain.pth`、`--repo./dinov2`、`--cal-images./cal_images`、`--output-dir.`、`--jobs 16`，默认不启用 `--save-cache`；启用后保留临时 workspace。不要安装上游 `requirements.txt`，因为它会将 OE Torch 降级到 2.0。

<a id="validation"></a>
## 转换后验证

导出器执行 ONNXRuntime 浮点输出比较并打印 cosine/max-absolute error；`mapper.py` 确认 `hb_compile` 生成预期 HBM。板端验证还应在相同预处理 tensor 上，将两路输出与 float ONNX 比较。

<a id="artifacts"></a>
## 产物

| 产物 | Target | 落盘路径 |
| --- | --- | --- |
| `dinov2_vits14_224_int16_nashe.hbm` | s100 / Nash-E | 下载后 `samples/vision/dinov2/model/nash-e/`，或 mapper 的 `--output-dir` |
| `dinov2_vits14_224_int16_nashm.hbm` | s100p / Nash-M | 下载后 `samples/vision/dinov2/model/nash-m/`，或 mapper 的 `--output-dir` |
| `dinov2_vits14_224_int16_nashp.hbm` | s600 / Nash-P | 下载后 `samples/vision/dinov2/model/nash-p/`，或 mapper 的 `--output-dir` |

<a id="known-gaps"></a>
## 补充准备

- 在指定的 OE 容器中执行转换，并准备可下载固定 checkpoint 的网络。使用对应板卡运行模型检查。
- manifest HBM 哈希为 `sha256: null (unknown)`；下载器打印下载文件的 SHA-256。

## 许可

源 checkpoint 是 Meta AI 的 Apache-2.0 DINOv2。转换代码遵循仓库 [LICENSE](../../../../LICENSE) 的 Apache-2.0。
