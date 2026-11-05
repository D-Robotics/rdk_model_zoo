[English](README.md) | 简体中文

# OSNet X5 转换

<a id="source-model"></a>
## 源模型

使用固定的 PyTorch [源码](../third_party/README.md) 和 [官方 x1.0 Market1501 权重](https://drive.google.com/file/d/1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA/view)，下载到 /inputs/osnet_x1_0_market1501.pth。权重 SHA-256 为 2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154；源码 SHA-256 为 c7c1c29187d6330f859c91da229271531920464c7011aec13842a086b2263cae，原上游 Git commit 未知。

<a id="directory"></a>
## 目录

prepare.py 导出空间头、准备校准和基础配置，并派生后段 INT8 配置。生成的 weights/data/runs 全部写入显式指定的仓库外工作目录。

<a id="toolchain-targets"></a>
## 工具链与目标

在已有 openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310 环境执行：hb_mapper 1.24.3、HBDK 3.49.15、PyTorch 1.13.0+cpu、ONNX 1.12.0、ONNX Runtime 1.16.2，及 NumPy/Pillow/PyYAML。目标为 X5 bayes-e。/repo 挂载完整仓库、/inputs 挂载准备好的文件、/work 为新的可写目录。脚本不创建容器，也未提供 nash-e/nash-p 配方或 S 系列制品。

<a id="export"></a>
## 导出

```bash
python3 /repo/samples/vision/osnet/conversion/prepare.py prepare \
  --source-file /repo/samples/vision/osnet/third_party/osnet.py \
  --weights /inputs/osnet_x1_0_market1501.pth \
  --dataset /inputs/Market-1501-v15.09.15 --workspace /work
```
严格加载 751 类训练权重，将推理 Linear/BN1d 等价改写为 1×1 Conv/BN2d，导出 opset 11，并用 16 张图与原浮点模型对照。ONNX 输入为 [1,3,256,128] float32，输出 [1,512,1,1]。

<a id="calibration"></a>
## 校准

按评测说明准备 Market-1501-v15.09.15 的 bounding_box_train/query/bounding_box_test。上述 prepare 命令以 seed=20261029 选择 128 张训练图，训练与 query 身份不得重叠，并检查 12936/3368/15913 的划分规模。校准张量为平面 RGB float32 [0,255]；融合均值为 123.675,116.28,103.53，缩放系数为 0.0171247538316637,0.0175070028011204,0.0174291938997821。Runtime 使用不同的有符号 RGB-minus-128 字节。

<a id="compile"></a>
## 编译

```bash
hb_mapper checker --model-type onnx \
  --model /work/weights/osnet_x1_0_spatial.onnx --march bayes-e
hb_mapper makertbin --config /work/runs/spatial-rgb-int16/config.yaml --model-type onnx
python3 /repo/samples/vision/osnet/conversion/prepare.py late8 --workspace /work
hb_mapper makertbin --config /work/runs/spatial-rgb-late8/config.yaml --model-type onnx
# Optional fast INT8 comparison
hb_mapper makertbin --config /work/runs/spatial-rgb/config.yaml --model-type onnx
```
配置使用 latency/O3/jobs=8。精度优先配置请求 set_all_nodes_int16，但保留编译器回退；late8 从优化后的浮点图选择 conv4/conv5/projection 节点，设置 INT8 输入/输出。以实际编译分配为准，高精度模型不是全 INT16。

<a id="validation"></a>
## 检查产物

检查编译日志和 quant-info。在 X5 用下方原生命令查看模型信息；登记新制品身份后，再用 evaluator/main.py 采集完整 Market1501 特征。新编译不能继承原 BIN 摘要或指标。matching.OSNet 库接口接受符合合同的自定义模型，普通 CLI 刻意只选择已登记制品。

```bash
hrt_model_exec model_info --model_file=/work/runs/spatial-rgb-late8/compiled/osnet_x1_0.bin
```

<a id="artifacts"></a>
## 产物

/work/weights/osnet_x1_0_spatial.onnx 为源图；/work/runs/<variant>/config.yaml 为编译配置；compiled/osnet_x1_0.bin 为 X5 部署模型。为每个源码/权重配对保留 export.json、calibration_manifest.json、完整编译报告和后续板测结果。

<a id="known-gaps"></a>
## 补充准备

自行准备精确权重、按数据集权限取得的完整 Market1501 和匹配 OE 环境。回读 TAR 包含部署模型、固定输入和结果，不包含原转换工作区。准备脚本拒绝已有校准/输出目录，请使用新工作目录。小规格、S100、S600 需要分别适配和评估。

### 复用现有 BIN 并建立新的 FP32 基线

`export` 使用固定源码和 checkpoint，核验空间头等价性并导出 FP32 ONNX，不创建校准数据或重编译 BIN。要求完整 query/gallery；收据明确标注 float_baseline_only，不代表恢复了原转换目录。`prepare` 仍要求独立的完整训练集，并生成 128 张校准数据及 PTQ 配置。

```bash
python3 samples/vision/osnet/conversion/prepare.py export --weights /absolute/market1501.pth --dataset /absolute/Market1501 --workspace /absolute/new/export-run
```

S 系列使用各平台私有候选包中的 HBM，按 SHA256SUMS 和 artifacts.json 校验后以 --model-path 指定；X5 回读 ZIP/TAR 不含这些 HBM。命令、校准和固定输入要求见 [S 系列 Workbench 流程](../evaluator/S_SERIES_cn.md)。
