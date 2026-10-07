<!-- 模板：conversion/ README（中文）。契约：readme-contract.md §4.5。
     保持锚点；替换 ⟪…⟫；完成后删除引导。缺失的配方环节必须列入 known-gaps——
     禁止用通用命令骨架伪装已验证的转换流程。 -->

# 模型转换 — ⟪模型名⟫

<a id="source-model"></a>
## 源模型

> **必须回答：** 源框架、权重版本/发布、权重获取方式、与官方发布的对应关系。

- 框架：⟪PyTorch ⟪版本⟫ / PaddlePaddle ⟪版本⟫ / …⟫
- 权重：⟪发布 tag 或 commit⟫，来源 ⟪source⟫
- 对应关系：⟪如官方 yolov8n.pt ⟪ver⟫⟫

<a id="toolchain-targets"></a>
## 工具链与目标

> **必须回答：** OpenExplorer 工具链版本、每个 target 的 march、每个 target 编译的
> 配置入口。

| Target | march | OE 版本 | 配置 |
| --- | --- | --- | --- |
| ⟪target⟫ | ⟪bayes-e / nash-e / nash-m / nash-p⟫ | ⟪版本⟫ | ⟪yaml/脚本路径⟫ |

<a id="export"></a>
## 导出（ONNX）

> **必须回答：** 环境、实际脚本/命令（cwd）、产物 ONNX 及其预期 shape/布局。
> 外部导出工具的获取与准备放在“补充准备”。

```bash
# cwd：⟪dir⟫
⟪准备好所需导出工具后的实际导出命令⟫
# 预期：⟪onnx 路径 + 输入 shape/dtype/布局⟫
```

<a id="calibration"></a>
## 校准

> **必须回答：** 校准数据来源与数量、量化配置、实际校准命令与外部工具的准备步骤。

- 数据集：⟪名称/版本、样本数、来源⟫
- 配置：⟪路径⟫
- 命令：⟪命令与 cwd⟫

<a id="compile"></a>
## 编译

> **必须回答：** 生成 .bin/.hbm 制品的完整命令与制品命名约定
> （`<model>_<resolution>_<chip>.⟪bin|hbm⟫`）。

```bash
# cwd：⟪dir⟫
⟪编译命令⟫
# 预期：⟪制品路径⟫
```

<a id="validation"></a>
## 转换后验证

> **必须回答：** 如何检查制品（板端运行命令、参照对比）、预期输出和成功判断。

- 冒烟：⟪命令——与 sample 快速体验一致，使用新制品⟫
- 预期结果：⟪退出码、输出张量或结果文件⟫

<a id="artifacts"></a>
## 产物

> **必须回答：** 产物清单、target 对应、落盘路径——必须与 model/README.md 的制品表
> 一致。

| 产物 | Target | 落盘路径 |
| --- | --- | --- |
| ⟪file⟫ | ⟪target⟫ | ⟪path⟫ |

<a id="known-gaps"></a>
## 补充准备

> **必须回答：** 所需外部校准数据、配置或工具链示例，逐项给出获取和准备步骤。

- ⟪所需输入/配置及获取、准备步骤⟫
