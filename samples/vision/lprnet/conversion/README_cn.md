# LPRNet 转换

<a id="source-model"></a>
## 源模型

固定源提供 X5 部署制品和协议说明，但没有仓库内 ONNX 导出脚本、checkpoint、校准集或 PTQ YAML。源 conversion README 使用外部 OE 包和用户提供的 `your_lprnet_config.yaml`；该占位文件不在本仓库，不能视为可复现配方。

<a id="toolchain-targets"></a>
## 工具链与目标

源文档指向带有 `hb_mapper` 和 `hrt_model_exec` 的 RDK X5 OpenExplorer 环境。仅支持 X5；LPRNet 没有 S march 或 S 制品。

<a id="export"></a>
## 导出

固定源没有导出配方。用户必须自行提供 ONNX/checkpoint 流程，并产出 `input float32 NCHW (1,3,24,94)`、`output float32 (1,68,18,1)` 的部署图——即发布版 `lpr.bin` 的 native logits；CTC 解码在移除单元素轴后消费 `(68,18)` 载荷。本迁移没有虚构或执行该流程。

<a id="calibration"></a>
## 校准

固定源没有校准数据集或校准生成器。已提交的 `test_input.dat` 是 runtime 输入 fixture，不是校准集，不能称为图像数据或代表性校准数据。

<a id="compile"></a>
## 编译

用户在本 conversion 目录补齐外部 ONNX 和 YAML 后，源文档给出以下 OE 命令；它们是条件模板，不是本仓可直接运行的完整配方：

```bash
# cwd: samples/vision/lprnet/conversion；输入：外部 your_lprnet_config.yaml
hb_mapper checker --model-type onnx --config ./your_lprnet_config.yaml
hb_mapper makertbin --model-type onnx --config ./your_lprnet_config.yaml
# 成功需 OE 工具报告成功并生成 X5 .bin 制品
```

确切输出文件名、校准选项、量化设置和源 checkpoint 均未知。

<a id="validation"></a>
## 转换后验证

可用 `hrt_model_exec model_info --model_file ./lpr.bin` 检查生成模型，再与 runtime binding 对照。本迁移没有执行转换或 `hrt_model_exec` metadata 检查；已发布制品本身在一块 X5 8GB 和一块 X5 4GB 上通过同板 source/unified 对照（2026-09-24，见 evaluator README），该结果验证 runtime 一致性，不代表转换可复现。

<a id="artifacts"></a>
## 产物

唯一的 manifest 制品是 `x5:lprnet:lpr.bin`；runtime 期望路径为 `../model/lpr.bin`。ONNX、checkpoint、校准数据和编译日志属于外部输入，不纳入仓库。

<a id="known-gaps"></a>
## 缺失项

- 源没有导出脚本、checkpoint 版本、校准生成器、PTQ YAML 或可复现 OE 包。
- `your_lprnet_config.yaml` 是源占位符，不是仓库文件。
- 发布者 SHA-256 未知；转换为 `not-run`。已发布制品的板端一致性记录在 evaluator README 中，不延伸到重建。
