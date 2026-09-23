# MODNet 转换

<a id="source-model"></a>
## 源模型

源文档指向官方 [MODNet 工程](https://github.com/ZHKKKe/MODNet) 和论文，但固定源没有锁定 checkpoint revision，也没有仓库内 ONNX exporter。源 README 提到 `onnx_export/` 和 `ptq_yamls/`，但审计文件中不存在这些路径；本迁移不复制也不虚构它们。

<a id="toolchain-targets"></a>
## 工具链与目标

源文档描述 RDK X5 OpenExplorer 工具 `hb_mapper`、`hb_perf` 和 `hrt_model_exec`。manifest 只有 X5 `bayes-e` 部署制品，没有 S 目标或 C++ 转换路径。

<a id="export"></a>
## 导出

源没有 `onnx_export` 脚本，也没有固定 checkpoint/export 配置。用户必须自行提供输入为 float32 RGB NCHW `(1,3,512,512)`、输出为 float32 matte `(1,1,512,512)` 的 ONNX 图。本迁移没有执行或重建缺失流程。

<a id="calibration"></a>
## 校准

审计源树没有 PTQ YAML 或校准生成器。`test_data/person.jpg` 是推理 fixture，不是代表性校准集；本轮没有生成校准数据。

<a id="compile"></a>
## 编译

由于源 ONNX 和 YAML 缺失，无法给出可复现的完整编译命令。用户准备好两者并进入工具链环境后，源文档的概念步骤是先执行 `hb_mapper checker`，再执行 `hb_mapper makertbin`；具体选项、校准、输出前缀和 checkpoint 均未知。生成制品仍须先通过 model binding 才能使用。

<a id="validation"></a>
## 转换后验证

外部模型和配置准备完成后，才可在目标工具链中使用 `hb_perf`、`hrt_model_exec` 并将输入/输出 metadata 与 runtime README 对照。本轮没有运行导出、PTQ、编译或板测。

<a id="artifacts"></a>
## 产物

唯一 manifest 行是手工制品 `x5:modnet:modnet_512x512_rgb.bin`，runtime 期望路径为 `../model/modnet_512x512_rgb.bin`。ONNX、checkpoint、校准数据、YAML、日志和编译模型都是外部输入。

<a id="known-gaps"></a>
## 缺失项

- 源 README 提到的 `onnx_export/`、`ptq_yamls/` 不在固定源 inventory 中。
- checkpoint 版本、导出参数、校准集、PTQ 设置和编译输出命名未知。
- manifest 没有 URL 或发布者 SHA-256；转换和板端验证为 `not-run`。
