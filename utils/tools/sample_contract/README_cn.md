[English](README.md) | 简体中文

# sample-contract 检查器

统一样例布局的静态契约检查器。它执行 `docs/sample-standards/readme-contract.md` 和 `docs/sample-standards/inference-contract.md` 中可由机器判定的规则；无法静态判定的内容交由语义评审。跳过的检查会报告为 skip，不计作通过。

## 检查内容

| 规则 | 范围 |
| --- | --- |
| `R-README-PAIR` | 每个现有层级均提供 `README.md` 与 `README_cn.md` |
| `R-README-SECTIONS` | 固定锚点 ID（实时从模板获取）存在、唯一，且符合模板顺序 |
| `R-README-LINKS` | 相对链接/图片指向现有文件；文档内部片段指向显式锚点；外部 URL 不在检查范围（不访问网络） |
| `R-CLI-DEFAULTS` | 每种语言的 runtime/python 参数表与实际 `build_parser` 默认值一致 |
| `R-I18N-PARAMS` | 英中参数表的选项集合和默认值一致 |
| `R-STAGE-PURITY` | AST 扫描：阶段函数——标准 `preprocess`/`infer`/`postprocess`/`predict` 以及兼容名称 `pre_process`/`forward`/`post_process`，连同对应的 `preprocess_*`/`infer_*`/`postprocess_*`/`forward_*`/`pre_process_*`/`post_process_*`/`run_*` 前缀——不得包含下载、文件写入/保存、子进程或破坏性调用。`main.py` 与 `legacy.py` 按策略跳过并记录原因；`cli.py`/`yolo_cli.py` 中的模块级辅助函数（如响应明确请求进行下载的 `run_prepare`）属于 CLI 应用边界，记录为跳过，但这些文件中使用阶段名称的类方法仍接受检查 |

## 使用方法

```bash
# one sample
python3 utils/tools/sample_contract/check.py --sample samples/vision/resnet

# CI scope: every sample row listed in the progress map
python3 utils/tools/sample_contract/check.py --scope migration --report out.json

# never execute sample code (CLI-default checks then record skips)
python3 utils/tools/sample_contract/check.py --sample ... --parser-mode static

# checker's own tests (fixtures first)
python3 -m unittest discover -s utils/tools/sample_contract/tests -v
```

退出码：`0` 无违规；`1` 发现违规；`2` 用法/配置错误。跳过项（缺少 `main.py`、导入失败、静态模式、按策略跳过的文件，以及 `cli.py`/`yolo_cli.py` 中 CLI 边界的模块级辅助函数）始终打印并纳入 JSON 报告。

### 阶段名称范围

样例架构以 `preprocess`/`infer`/`postprocess` 为主要阶段名称，并保留 `pre_process`/`forward`/`post_process` 作为相同函数体的薄兼容别名。因此纯度扫描覆盖两种名称，下载或写文件不应出现在任何一种阶段函数中。CLI 边界豁免仅覆盖样例本地 `cli.py`/`yolo_cli.py` 中具有阶段形态名称的模块级函数，每次都记录为具名 skip；类方法、其他文件与整个目录仍接受检查。

## 默认值的标准形式

README 的 `Default` 列与解析器经过标准化后进行比较：`None` → `null`（README 可写 `null`/`none`）、布尔值 → `true`/`false`、列表 → JSON 形式（`[0]`、`[0, 1]`）、标量 → 字面量、仓库内绝对路径 → 仓库相对 POSIX 形式。README 单元格中的反引号及包围引号会被去除。

## 检查范围

`--scope migration` 读取 `docs/releases/unified-migration/x5-s-migration-map.md` 的进度区域，检查 **Refactor** 列为 `in-progress` 或 `done` 的全部样例（忽略括号注释）。选中行若没有可解析的 `samples/<domain>/<name>` 目录，会产生 `R-SCOPE` 违规，使进度表与样例树保持同步。

## 豁免

`--exemptions file.json` 接受经过评审的例外，形式为 `{"rule", "path", "line", "reason"}`，其中 `reason` 必填。可选 `"message"` 字段将条目固定到一条确切问题消息；某些规则会在同一行报告多个问题时应使用它（`R-README-SECTIONS` 在第 0 行报告每个缺失锚点）。每条豁免只指定一个问题位置；不接受整个目录的豁免。没有匹配问题的豁免会产生 `R-EXEMPTION`，因此修复问题时，应在同一变更中删除对应豁免。模式与匹配行为由 `tests/` 下的 fixture 测试覆盖。

## 边界

检查器读取文件；导入模式下，导入受信任的仓库 `main.py` 模块并调用 `build_parser`。它不下载、不加载板端 SDK，也不执行 README 代码块。文案质量、重复 `predict` 逻辑、板端行为和转换正确性由语义评审与板端冒烟测试覆盖。
