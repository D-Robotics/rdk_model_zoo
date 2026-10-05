# 全部本仓 Sample 可读 Runtime — integration_docs_checker 批次报告

日期：2026-10-05。分支：`codex/readable-model-examples-20261001`（共享本地
worktree；`develop` 与远端未动）。执行器：本地 Claude Code 2.1.276 + GLM
`glm-5.3[1m]`；主机 Python `rdk_model_zoo/.venv/bin/python`。
设计依据：`docs/superpowers/specs/2026-10-05-all-sample-readable-runtime.md`
（spec 与 plan）。本批不改任何 Sample 源码；其余批次并行进行中，全仓行为验收
由 Codex 统一安排，本报告不宣称全仓完成。外部证据目录：
`local-execution/20261005-all-sample-readable-runtime/integration_docs_checker/`
（`logs/`、`survey_stages.py`、`stage_surface.json`、`gen_coverage.py`、
`progress.json`）。

> **批次快照声明**：本报告（含 §3 coverage.json 生成时点的入口形态记录）是
> 批次快照；其后 docs_closeout 批次已按更新的源码刷新 coverage.json 的行事实
> 并修正个别归类措辞，`implementation_evaluation` 全行保持
> `pending_final_review`。**最终状态以 Codex 终审报告为准**。

## 1. 范围与产出

| 文件 | 变更 |
| --- | --- |
| `tools/sample_contract/check.py` | R-STAGE-PURITY 增加 canonical 阶段名（`preprocess`/`infer`/`postprocess` 及 `preprocess_*`/`infer_*`/`postprocess_*` 前缀）；新增 `cli.py`/`yolo_cli.py` 模块级应用函数的 CLI 边界（记 skip，类方法与其余文件照查）；模块 docstring 同步 |
| `tools/sample_contract/tests/test_check.py` | 新增 `CanonicalStagePurityTests`（5 项） |
| `tools/sample_contract/tests/fixtures/bad_stage_purity_canonical/` | 新负向 fixture：canonical 阶段内下载/写文件/`infer_` 前缀 + cli.py 边界与类方法对照 |
| `tools/sample_contract/README.md` | R-STAGE-PURITY 行更新 + “Stage-name scope (2026-10-05)” 节 |
| `docs/sample-standards/inference-contract.md` | §1 canonical 四方法为主名、旧名为同实现兼容委托、本地具名类/薄 main/本地 cli；§3 职责表更名并加 cli 行、多阶段不强制三方法、C++ 原生 Generate/stream/reset；§4/§5/§6 用语与参照（`classify.py:ResNetClassifier`）更新 |
| `CLAUDE.md` | 接口段重写（canonical 为主、参考实现 `classify.py`、`samples/_shared/runtime.py:RuntimeSession` 仅薄 SDK 加载/身份且本轮未改后端语义、51 样例范围 + pending review）；C++ 段补原生等价接口 |
| `AGENTS.md` | 可读范例条目扩展（canonical 名、51 样例范围、ACT/Pi0 与 llm 例外、coverage.json `pending_final_review`、检查器双拼写） |
| `README.md` / `README_cn.md` | “Read and extend the code” 段与目录注释扩展到 51 样例范围、ACT/Pi0 排除、待评审措辞（双语同步） |
| `docs/architecture/model-examples.md` | 标题/导语扩展；新增 §2 “全仓形态”（六类别形态表 + 通用不变量）；§4/§5 编号顺延；§5 边界更新（C++ 原生接口、后端语义未改）；既有 ResNet/YOLO 事实表（§4.1/§4.2）未动 |
| `docs/migration/2026-10-05-all-sample-readable-runtime.md` | 新增：51 样例旧新映射（阶段名/单模型/多阶段/原生四张通用表 + 分类 21 类名表 + 逐样例行表）与验证边界 |
| `docs/releases/unified-migration/2026-10-05-all-sample-coverage.json` | 新增：51 行结构化状态表 + 文件系统对账（见 §3） |

## 2. 检查器变更与 TDD 证据

新规则行为先写失败测试（fixture `bad_stage_purity_canonical`：canonical 阶段内
`urlretrieve`/`cv2.imwrite`/`np.save`/`open(wb)`、`infer_` 前缀、cli.py 模块级
`run_prepare` 下载、cli.py 内类方法写文件）：

1. **失败先行**：`$VENV -m unittest discover -s tools/sample_contract/tests -p test_check.py -v`
   → `Ran 32 tests, FAILED (failures=4)`（旧 checker 不识别 canonical 名；
   `run_prepare` 被误报为 stage；cli.py 内类方法未查）。
   日志：`logs/failfirst-canonical-stage-tests.log`。
2. **实现后**：同命令 → `Ran 32 tests, OK`，exit 0。
   日志：`logs/after-checker-tests.log`。（其间一次运行失败 1 项，系新测试自身
   的消息前缀解析笔误，修正断言后通过；该中间运行未单独保留日志，特此说明。）
3. **既有行为回归**：32 项中 27 项为原有 fixture 测试（pair/sections/links/
   CLI-defaults/i18n/purity/exemptions/scope），全部保持通过；未改动
   `main.py`/`legacy.py` 政策 skip，未引入任何目录级豁免。
4. **对已完成样例的契约检查**（仅 ResNet 与 ultralytics_yolo，两者无并行编辑）：
   - `$VENV tools/sample_contract/check.py --sample samples/vision/resnet` →
     exit 0，`0 violations, 3 skips`（legacy.py/main.py 政策 skip + 新增
     cli.py 模块级边界 skip：`run_dry_run, run_list_models`）。
     日志：`logs/contract-resnet.log`。
   - `$VENV tools/sample_contract/check.py --sample samples/vision/ultralytics_yolo`
     → exit 0，`0 violations, 2 skips`。日志：`logs/contract-ultralytics_yolo.log`。
5. **只读探针**（非 checker 运行、非验收）：`survey_stages.py` +
   `stage_surface.json` 对 49 个 Python runtime 样例做方法名清点；另以与新规则
   相同的 AST 判定对全部样例 `runtime/python/*.py` 预扫，快照时刻新规则增量为
   0 findings（用于确认规则收紧未在本批制造已知红灯；并行编辑仍在进行，不作为
   最终结论）。

CLI 边界语义（按任务要求收窄）：仅 `cli.py`/`yolo_cli.py` 的**模块级**、且名字
为 stage 形态的函数豁免并逐名记 skip（reason 含函数名）；类方法、嵌套函数与其余
一切文件照常检查；`infer`/canonical 名不存在全仓忽略。

## 3. coverage.json 生成与文件系统对账

生成器 `gen_coverage.py`（外部目录）：

- 逐行数据：sample/domain/batch/entry/model_class/model_file/cli_helper/
  actual_backend/special_semantics/languages；
- **文件系统对账**：枚举 `samples/{vision,speech,robotics,llm}/*`（排除 `vla`、
  `_shared`）得 51 目录，与 51 行互查 missing/duplicates/extra，非空即退出码 1；
  实际结果 `rows=51 fs_dirs=51 check=ok`（`logs/gen-coverage.log` 保存最终通过
  运行；此前两次运行分别暴露 `_shared` 基础设施目录误计入 extra 与 LLM 行
  model_file 前缀错误，修正后重新生成）；
- `stage_surface_inventory` 为 AST 方法名清点，**明确不是行为验收**：每行固定
  `implementation_evaluation: pending_final_review`；
- `entry_predict_shape_snapshot` 区分三种合法入口形态（main 直接 predict /
  委托 application.py / 流式分块组合阶段）与原生 C++ 入口，快照时刻
  paraformer、himloco 为 application.py 委托，asr 为分块组合（其源码注释明示
  `predict` 为等价单次 API）。

## 4. 未执行（not-run / 待集成）

- 全仓逐样例行为回归、全仓契约检查（其他批次并行编辑中，本批只跑了上述两样例）、
  干净 checkout 入口复现、Catalog build/check：not-run，待 Codex 统一安排。
- 板端 SDK 推理、真实模型下载/ONNX 导出/OE/Mapper/HMCT：按任务约束 not-run。
- `docs/migration/` 与 coverage 中对 entrypoint_polish_vision 批次（pointnet、
  yolo26_depth 的 `return_details`、3dresnet/dinov2/yoloe 入口重写收尾）仅按
  快照与该批 progress 记录描述为“进行中”，未预支完成结论。

## 5. 代码与测试不提交

按分工，本批全部改动留在工作树，由 Codex 评估后按路径提交。
