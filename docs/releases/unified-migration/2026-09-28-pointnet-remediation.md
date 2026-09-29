# PointNet R1 整改（作者记录）

本记录覆盖 2026-09-28 独立评审
[2026-09-28-pointnet-independent-review.md](2026-09-28-pointnet-independent-review.md)
中唯一的阻塞项 **POINTNET-R1**：`samples/vision/pointnet/runtime/python/visualization.py`
残留源码迁移来的 `parse_args()` CLI parser，其返回注解引用了未导入的 `argparse`，
在支持的 Python 3.13 上导入即 `NameError`；默认（不带 `--no-plot`）路径经
`runtime/python/main.py:65` 导入该模块，README 声明的 3.10+ 快速上手路径直接失败，
而 Python 3.14 的延迟注解（PEP 649/749）掩盖了这一缺陷，使既有 21 项主机测试全绿
但不构成充分证据。任务由 GLM（Claude Code 会话）实施，等 Codex 独立复审、提交与
GitHub 同步。

**状态口径：**

- 全程**未 commit / 未 push / 未 merge**；作者改动仅限
  `samples/vision/pointnet/runtime/python/visualization.py` 与
  `samples/vision/pointnet/tests/test_pointnet.py` 两个文件，外加本记录与
  [evidence](../../../evidence/2026-09-28-pointnet-remediation/verification.json)
  新目录。
- **Board = not-run**：本任务不连板卡、不下载模型、不装 OE；板端与渲染效果均不在
  主机整改证据范围内。
- 主机测试通过是作者自检，不等于独立验收；独立 Review 保持 pending。

## 1. 修复方式：删除，而非补 import

按评审验收要求，`visualization.py` 整体删除 `parse_args()`（-38 行），**不是**
补 `import argparse` 保留第二套参数表——源 parser 的 `--img-save-path` 等默认值与
真实 CLI（`build_parser()` 的 `--output-dir`/`--no-plot` 等）互相矛盾，保留只会
制造两套口径。模块 docstring 同步改为声明 `main.py is the sole CLI owner`。

行为保持不变，未做无关重构：

- 绘图函数 `create_point_cloud_axes` / `save_original_view` /
  `save_segmentation_view` 与 `summarize_parts` 原样保留；
- Agg 后端、源 X/Z/Y 坐标轴映射（源列 0/2/1 → 显示 X/Y/Z）、输出文件名
  `result_orig.png` / `result.png`、`--no-plot` 跳过绘图仅写 labels/JSON 的行为
  全部不变；
- `main.py` 与两份 README **零改动**：`main.py` 的 SHA-256 与评审记录
  （`43eba30b…`）一致，README 既有描述（参数表、结果、matplotlib 排障）在删除
  死代码后仍然逐句成立；`run.sh` 经 `python3 -m …main` 启动，从未触达被删
  parser。

## 2. 新增回归（tests/test_pointnet.py，+2 项）

`VisualizationDefaultPathTests`，全部导入**真实** visualization 模块（断言
`__file__` 指向交付路径），只对 matplotlib/numpy 做显式标注的 import double，
从不 mock 模块本身：

- **默认绘图路径端到端**：注入 SDK fixture（复用既有 `metadata(n)` Fake runtime，
  logits 使 argmax 呈 0/1/2/3 循环）+ 绘图 fixture double，跑 `main([])` 默认路径。
  断言：rc=0；`result.json` 的 `point_count`/`counts` 与预期标签一致；
  `labels.npy` 为 int32 且逐点正确；`savefig` 恰好以 `result_orig.png`、
  `result.png` 顺序各调用一次；原图 `scatter3D(label='chair')` 传入的三列逐一等于
  椅子点云归一化后的源列 0/2/1（钉死 X/Z/Y 映射）；分割图按
  back/seat/leg/arm 四次 `scatter`；轴标签恰为 X/Y/Z。当前 venv 无 matplotlib，
  故这是"调用级"证据，**不**声称验证了实际渲染像素或板端效果。
- **导入与注解健全性**（三层，专钉 R1 这一类缺陷）：
  1. 真实模块 fresh import（3.13 上导入即暴露 NameError）；
  2. 对模块与全部可调用对象强制求值 `__annotations__`——在 3.14+ 延迟注解下
     仍会触发未导入名的 NameError（已在 3.14.7 实测）；
  3. 对源码做 AST 静态扫描：所有函数参数/返回/AnnAssign 注解的根名字必须落在
     模块绑定名或 builtins 内——与解释器版本无关。另断言 visualization 不再含
     `parse_args`、`main.build_parser` 仍可调用（唯一 CLI owner）。

## 3. 验证结果（解释器与日志见 evidence/verification.json）

| 检查 | 结果 |
| --- | --- |
| 评审方导入复现脚本（修复后） | Python 3.13.15 rc=0（修复前 rc=1 NameError）；3.14.7 rc=0 |
| RED：临时复活缺陷跑两项新测试（3.14.7） | 2 tests, errors=2（注解强制求值与默认路径双失败）后恢复修复版 |
| AST 静态扫描 RED（scratch 副本） | 正确标出未导入名 `argparse` |
| `samples/vision/pointnet/tests`（3.14.7，live） | **Ran 23 tests — OK**（21 既有 + 2 新增） |
| `samples/_shared/tests`（3.14.7，live） | **Ran 158 tests — OK** |
| resnet / ultralytics_yolo / paddle_ocr（3.14.7，live） | 52 / 143 / 44 — 全部 OK |
| `tools/sample_contract/check.py --sample samples/vision/pointnet` | 0 violations，1 个记录在案的 CLI policy skip，0 exemptions（快照与 live 各一次） |

**环境说明（如实记录，不影响结论）：** 验证期间并行的 MiniCPM 任务正在改共享
manifest `docs/release/s/models.yaml`，中途该文件短暂处于 YAML 不完整状态，
使依赖 manifest 的套件在 live 树上一度报错（43 个错误全部位于
test_assets/test_manifest_coverage 等清单读取类测试，与本整改无关；当时在
/tmp 隔离副本中以 base manifest 完成快照验证，双份日志均保留并加注说明）。
该任务完成后，live 树全量重跑即为上表结果。

## 4. 边界与移交

- Python 3.10–3.12 解释器本机不存在，3.13 缺 numpy/PyYAML：这些版本的**套件**
  为 not-run；3.13 以评审方导入复现（rc=0）加解释器无关的 AST 静态扫描覆盖 R1
  所在层。未安装任何包。
- 绘图 fixture 是调用级记录，不证明 PNG 渲染内容；板端推理、模型下载、导出/量化
  均 not-run。
- 未触碰并行任务的 MiniCPM 与 README 改动、评审报告、历史证据与计划台账。
- 本包到此为止，等 Codex 独立复审、提交与 GitHub 同步；B8 整体与其余七个样例
  家族另行推进。
