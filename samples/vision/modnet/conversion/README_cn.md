[English](README.md) | 简体中文

# MODNet 转换

<a id="source-model"></a>
## 源模型

从官方 [MODNet 工程](https://github.com/ZHKKKe/MODNet) 获取训练 checkpoint 并导出 ONNX。论文：[Is a Green Screen Really Necessary for Real-Time Portrait Matting?](https://arxiv.org/abs/2011.11961)。准备 checkpoint、导出代码、PTQ 配置和校准数据后，使用 X5 工具链编译。

<a id="directory"></a>
## 目录结构

```text
conversion/
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="toolchain-targets"></a>
## 工具链与目标

源文档描述 RDK X5 OpenExplorer 工具 `hb_mapper`、`hb_perf` 和 `hrt_model_exec`。manifest 只有 X5 `bayes-e` 部署制品，没有 S 目标或 C++ 转换路径。

<a id="export"></a>
## 导出

没有 `onnx_export` 脚本，也没有固定 checkpoint/export 配置。用户必须自行提供输入为 float32 RGB NCHW `(1,3,512,512)`、输出为 float32 matte `(1,1,512,512)` 的 ONNX 图。

<a id="calibration"></a>
## 校准

没有 PTQ YAML 或校准生成器。`test_data/person.jpg` 是推理 fixture，不是代表性校准集。

<a id="compile"></a>
## 编译

在 X5 OpenExplorer 环境中准备 `modnet.yaml`，将 `onnx_model` 字段指向导出的 ONNX，并配置代表性校准数据与输出前缀。仓库未附带此 YAML，以下命令使用准备好的配置：

```bash
hb_mapper checker --config modnet.yaml
hb_mapper makertbin --config modnet.yaml
```

<a id="validation"></a>
## 转换后验证

在 OE 环境中使用 `hb_perf` 检查编译模型性能：

```bash
hb_perf model_perf \
    --model ./modnet_512x512_rgb.bin \
    --input-shape input 1x3x512x512
```

将模型复制到 X5，在板端进行单线程运行测量：

```bash
hrt_model_exec perf \
    --model_file ./modnet_512x512_rgb.bin \
    --thread_num 1
```

输入为 float32 RGB NCHW `(1,3,512,512)`，使用 `(pixel - 127.5) / 127.5` 归一化至 [-1,1]；输出为 float32 `(1,1,512,512)` alpha matte，范围 [0,1]。对照运行时元数据并使用评估器比较 matte。

<a id="artifacts"></a>
## 产物

唯一 manifest 行是手工制品 `x5:modnet:modnet_512x512_rgb.bin`，runtime 期望路径为 `../model/modnet_512x512_rgb.bin`。ONNX、checkpoint、校准数据、YAML、日志和编译模型都是外部输入。

<a id="known-gaps"></a>
## 补充准备

- 源 README 提到的 `onnx_export/`、`ptq_yamls/` 未随附。
- checkpoint 版本、导出参数、校准集、PTQ 设置和编译输出命名未知。
- manifest 没有 URL 或发布者 SHA-256。
