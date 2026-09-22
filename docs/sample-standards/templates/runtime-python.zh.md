<!-- 模板：runtime/python README（中文）。契约：readme-contract.md §4.3。
     保持锚点；替换 ⟪…⟫；完成后删除引导。参数默认值与 build_parser 机器核对；
     集成示例必须原样可运行（由 sample tests 按 inference-contract 验证）。 -->

# Python 运行 — ⟪模型名⟫

<a id="environment"></a>
## 环境

> **必须回答：** 板卡/系统要求、Python 版本、依赖（或“板端镜像自带”）；显式说明
> `hbm_runtime` 仅存在于板端镜像（host 导入失败是设计行为）。

- 板卡：⟪targets⟫，系统镜像 ≥ ⟪版本⟫
- Python：⟪版本⟫；依赖：⟪清单或无⟫
- `hbm_runtime` 由板端镜像提供——本 runtime 不在开发机上运行。

<a id="usage"></a>
## 使用

> **必须回答：** cwd；默认命令（零额外参数）与自定义命令各一条；成功判据
> （退出码/打印输出/结果文件）。

```bash
# cwd：仓库根目录
python3 samples/⟪domain⟫/⟪name⟫/runtime/python/main.py --target ⟪target⟫ ⟪input⟫
# 成功判据：⟪criterion⟫
```

<a id="parameters"></a>
## 参数

> **必须回答：** **全部** CLI 参数及 parser 实际默认值（机器核对）。kebab-case
> 命名。不得出现 parser 未定义的参数，也不得遗漏。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--target` | str | `auto` | ⟪…⟫ |
| ⟪arg⟫ | ⟪type⟫ | ⟪default⟫ | ⟪…⟫ |

<a id="results"></a>
## 结果

> **必须回答：** 输出字段/文件的位置、格式与含义（坐标约定、类别 id、置信度语义）
> ——字段名与代码返回值一致。

⟪如打印 Top-5 (class_id, label, score)；结果图写 ⟪path⟫，框坐标 [x1,y1,x2,y2] 像素⟫

<a id="integration-example"></a>
## 集成示例

> **必须回答：** **完整可运行**的 Python 片段。输入/配置变量全部在示例内定义或指向
> 已声明前置步骤准备的具体本地文件；无未定义引用。该片段由本 sample 的 tests 针对
> fixture 执行验证。

前置：⟪如已按 model/README.md 准备制品；测试图位于
samples/⟪domain⟫/⟪name⟫/test_data/⟪image⟫⟫

```python
import ⟪module⟫

⟪binding = …（用真实路径具体构造）⟫
model = ⟪Model⟫(⟪args⟫)
result = model.predict(⟪已定义输入⟫)
print(⟪result 字段⟫)
```

<a id="stage-io"></a>
## 三阶段 I/O

> **必须回答：** 本 sample 的 pre_process / forward / post_process 契约摘要，
> 与 docstring 一致（见 inference-contract）。多阶段 pipeline（如 OCR det→rec）
> 每阶段一小节，另加 pipeline.predict 编排说明。

- `pre_process`：⟪Input⟫ → ⟪Tensors + Context⟫（⟪shape/dtype/布局⟫）
- `forward`：⟪tensor dict⟫ → ⟪RawOutputs⟫（⟪输出名/shape/量化语义——raw logit
  还是反量化后值⟫）
- `post_process`：⟪RawOutputs + Context⟫ → ⟪Result⟫（⟪结果类型/字段⟫）
- ⟪多阶段 sample 的 pipeline.predict 编排⟫

<a id="troubleshooting"></a>
## 故障排查

> **必须回答：** 只列真实失败模式（制品缺失/target 不匹配/输入尺寸约束），给出实际
> 报错文本与处置。

| 现象 | 原因 | 处置 |
| --- | --- | --- |
| ⟪报错文本⟫ | ⟪原因⟫ | ⟪处置⟫ |
