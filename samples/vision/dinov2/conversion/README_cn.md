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
sudo docker run -it --rm --gpus all --network host --shm-size=15g \
  -e HBTL_THREAD_NUM=2 -e OMP_NUM_THREADS=2 \
  -e MKL_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 \
  -e OMP_WAIT_POLICY=PASSIVE -e GOMP_SPINCOUNT=0 \
  -e NVIDIA_TF32_OVERRIDE=0 \
  --entrypoint /bin/bash \
  -v "$(pwd)":/workspace -w /workspace \
  registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0 \
  -l
```

请显式设置 `HBTL_THREAD_NUM`：已安装的 HBTL Context 有独立的线程数设置，与 `OMP_NUM_THREADS` 和 mapper 的编译参数 `--jobs` 分别生效。上述命令同时启用 OpenMP 被动等待，并关闭 CUDA 浮点对照中的 TF32。


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

`onnx_export/export_dinov2.py` 校验固定版本的源代码和权重文件，构建 ViT-S/14，固定 224 输入尺寸的位置编码，将 attention 重写为显式 MatMul + Softmax + MatMul，并导出输入 `input` `(1,3,224,224)` float32 和两个输出 `cls_feat`、`patch_feat`。导出器固定随机种子为 0，分别比较未改写的上游模型、注意力/位置编码改写后的模型与 ONNX Runtime 输出。两个输出都必须形状正确、数值有限，满足 `atol=1e-3, rtol=1e-3` 且余弦相似度不低于 `0.99999`，否则停止。可重复传入 `--parity-input normalized.npy` 检查真实预处理张量，并以 `--parity-report host_parity.json` 保存结果。执行：

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

使用 featuremap float32 输入、全节点 int16 和 int16 模型输出，`calibration_type` 保持未设置，以使用 hmct 默认的 modelwise 校准搜索。工具会在日志中记录选中的方法；2026-11-02 的三次校准均选中 `max`。下表是固定源版本记录的历史测量，与本次新转换分开记录。

```bash
# cwd：samples/vision/dinov2/conversion；前置：./cal_images 有真实照片
# 完整 mapper 命令会生成校准数据；不需要单独下载命令。
python3 mapper.py --cal-images ./cal_images --march nash-e --output-dir ./output
# 预期：hb_compile 前出现临时 calibration_data_norm/*.npy 和 calibration_manifest.json
```


### 量化配置对照（Nash-E 参考数据）

下表比较五种历史量化配置，`(sim)` 表示工具链模拟值。在这些测量中，int8 激活的余弦相似度最高为 0.91，max+percentile 校准约为 0.18；attention logits 减去最大值后的张量最小达到 -345。这些历史 KL 结果不代表新转换制品的精度。

| Config | cls cosine | patch cosine | Verdict |
|---|---|---|---|
| int8 + softmax-int32, featuremap, max | 0.081 | 0.803 | FAIL |
| int8 + softmax-int32, featuremap, KL | 0.892 | 0.894 | FAIL |
| int16, featuremap, max + 0.9999 | 0.184 | 0.840 | FAIL |
| int16, nv12, KL | 0.999 (sim) | 0.999 (sim) | FAIL (executed 0.01 / 0.12) |
| int16, featuremap, KL (source-recorded) | **0.9989** | **0.9983** | **PASS** |

`_reg4` register-token 变体在逐张量量化下的余弦相似度约为 0.80，普通变体约为 0.999。本配方使用普通 ViT-S/14 权重。

<a id="compile"></a>
## 编译

`mapper.py` 先准备校准数据，将前三个真实校准张量交给导出器检查，再生成 YAML、运行 `hb_compile` 和收集制品。它先检查 `hb_compile --help`，生成目标配置，使用 `featuremap` 输入和全 int16 量化，最后把 HBM 复制到 `--output-dir`。

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

### 从源码重建独立命名的 O1 版本

mapper 新增 `--optimize-level O1` 和 `--output-prefix`；默认仍为 O2 和历史目标文件名。O1 路径使用新启动的 OE 容器及其内置 ONNX/ONNX Runtime 包；执行上文源码与权重获取命令，跳过包升级命令。用 `python3 -c "import torch, onnx, onnxruntime; print(torch.__version__, onnx.__version__, onnxruntime.__version__)"` 记录实际版本。在该容器环境中，以下命令从固定上游源码、已校验权重和 50 张真实校准图像开始，完成 ONNX 导出与对照、校准及 `hb_compile` 编译，无需私有缓存 BC。

```bash
# cwd：samples/vision/dinov2/conversion；使用上文同一固定源码和权重
python3 mapper.py \
  --weights ./dinov2_vits14_pretrain.pth \
  --weights-sha256 b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9 \
  --repo ./dinov2 \
  --repo-revision 7764ea0f912e53c92e82eb78a2a1631e92725fc8 \
  --cal-images ./cal_images \
  --march nash-e --jobs 2 --optimize-level O1 \
  --output-prefix dinov2_vits14_224_int16_nashe_20261102_o1 \
  --output-dir ./output-o1 --save-cache
```

S100P 改用 `--march nash-m` 与前缀 `dinov2_vits14_224_int16_nashm_20261102_o1`；S600 改用 `--march nash-p` 与前缀 `dinov2_vits14_224_int16_nashp_20261102_o1`。HBTL 线程数保持 2，沿用上文完整环境设置。O1 是独立的编译优化选择，不声称与 O2 位相同或板端运行更快。复现时保留生成的 YAML、校准清单、浮点对照和完整工具链日志。

<a id="validation"></a>
## 转换后验证

导出器检查 ONNX 图，执行上游模型到改写模型、上游模型到 ONNX 的浮点数值门槛检查。`--save-cache` 可保留工作目录中的 `host_parity.json`。这是主机浮点图对照，不代表 HBM 板端精度或性能；`mapper.py` 确认 `hb_compile` 生成预期 HBM。板端验证还应在相同预处理 tensor 上，将两路输出与 float ONNX 比较。

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

## 新转换制品（2026-11-02）

显式版本 `vits14_224_int16_20261102_o1` 当前提供 2 个分别编译的 O1 HBM，目标为 `s100`, `s100p`。不传 `--variant` 或 `--asset-id` 时仍选择历史默认模型。新文件已通过浮点图对照、原装转换及 HBM 输入输出/架构静态检查；板端功能、精度、Python/C++ 性能均为 **not-run**，历史测量数据不适用于这些文件。

| Target | HBM | SHA-256 |
| --- | --- | --- |
| s100 | `dinov2_vits14_224_int16_nashe_20261102_o1.hbm` | `40d9654edb2b83b12928a4b23e8f482ec30c415d2389af5e64957d9921a9df1a` |
| s100p | `dinov2_vits14_224_int16_nashm_20261102_o1.hbm` | `05a94aae2cb7da14a6a0810c7c7e5e5bff604012b943dd66708e4cd9cc224cbe` |

```bash
python3 samples/vision/dinov2/model/download.py --target s100 --variant vits14_224_int16_20261102_o1
python3 samples/vision/dinov2/runtime/python/main.py --target s100 --variant vits14_224_int16_20261102_o1 --dry-run
```

此版本仅开放表中已完成的目标；尚未发布的目标会报错，不会回退。也可用完整的 `s:dinov2:nash-*/filename.hbm` 资产 ID 选择。输入仍是已归一化 RGB float32 FEATUREMAP `(1,3,224,224)`；实际公开输出为 float32，沿用元数据绑定的 `raw_f32` 路径；内部量化仍为 int16 配方，不添加 L2 归一化或 softmax。来源/权重版本、完整哈希、50 张真实图像校准配方、实际环境和验证记录见[转换记录](converted.json)。

## 许可

源 checkpoint 是 Meta AI 的 Apache-2.0 DINOv2。转换代码遵循仓库 [LICENSE](../../../../LICENSE) 的 Apache-2.0。
