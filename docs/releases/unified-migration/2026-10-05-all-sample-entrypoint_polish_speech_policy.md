# 全部本仓 Sample 可读 Runtime — entrypoint_polish_speech_policy 批次报告

日期：2026-10-05
执行分支：`codex/readable-model-examples-20261001`（工作树，未提交）。
执行器：本地 Claude Code 2.1.276 + GLM `glm-5.3[1m]`；主机 Python 为
`rdk_model_zoo/.venv/bin/python`（3.14.7）。
输入评审：`local-execution/20261005-all-sample-readable-runtime/entrypoint-polish-review.md`
（Codex 入口评审：main 必须可见地构造模型并调用 `predict()`；需要中间数据/
计时处采用批准的本地 typed details + `predict(..., return_details=True)` 方案；
每请求仍恰好一次生产推理；不得为保留产物而叠加第二条推理链）。
范围：4 个样例独占所有权（asr、kws、paraformer、himloco）；speech_policy_native
与 multistage 两批执行器已结束，本批次在其规范阶段命名成果之上完成入口可读性。
代码与测试未提交，由 Codex 按路径评估提交。外部证据日志目录：
`local-execution/20261005-all-sample-readable-runtime/entrypoint_polish_speech_policy/`
（baseline-/failingfirst-/final-/sdkfree-/contract- 日志、cli_equivalence.json、
report_equivalence.json、progress.json）。

## 本轮改动模式（对应评审要求）

1. **ASR**：`asr.py` 新增本地 typed `ChunkPrediction(text, prepared)` 并给
   `predict` 增加 `return_details=False` 关键字（默认返回旧文本字符串，不改
   默认行为）；`main.py` 的逐 chunk 循环从手动三阶段改为每块一次
   `task.predict(chunk.waveform, chunk.sample_rate, return_details=True)`，
   报告的 `valid_target_samples`/`text` 来自同一次执行的 details——每块仍恰好
   一次 runner 调用。参数声明、list/dry-run 渲染与报告记录移入新
   `cli.py`（`build_parser`/`run_list_models`/`run_dry_run`/`build_report`/
   `record_chunk`/`complete_report`），`main.build_parser` 再导出。
2. **Paraformer**：`application.py` 拆出逐语句助手 `note_runtime`、
   `prepare_utterance`（`Utterance`：manifest entry + 证据 record + features +
   frontend_ms）、`mark_attempted`、`record_prediction`、`save_features`、
   `complete`（digest 复验、prepared-manifest/result 写出）；
   `run`/`execute` 保留为同一助手的兼容组合。`main.py` 在入口内可见地逐语句
   循环：每语句一次
   `bundle.pipeline.predict(utterance.features.tensor, utterance.features.valid_frames)`
   （`load_runtime` 仍是唯一模型工厂，未建第二套模型，也未造含 IO 的
   Model.predict façade）。`inference_attempted`/`inference_executed` 置位顺序、
   preprocess-only feats 流程、CIF 零 token 跳过 decoder 与 `StageError` 归属
   未动（`pipeline.py`/`stages.py` 本批零改动）。
3. **HimLoco**：`application.py` 拆为 `prepare`（板卡/制品门禁、输入发现、
   新目录与报告文件 `open("x")` 独占预留、骨架持久化；`PreparedRun` 持有
   `persist`/`mark_failed`/`close`）、`load_task`（`RuntimeModelRunner` 构造 +
   调度参数 + runtime 元数据证据，模块级 patch 面保留）、`record_sample`
   （action dump + 证据记录 + 报告路径冲突检查）、`complete`（model/manifest
   digest 复验、延迟汇总、状态收尾）；`execute` 保留为兼容组合。`main.py`
   可见地运行 warmup 与逐输入 `task.predict(values)` 循环。报告字段顺序、
   `timing_scope` 语义、warmup 请求与全部错误串不变。
4. **KWS**：复核保留，零改动——`main.py` 已显式构造
   `KWS(runner, binding, config).predict(audio, rate)`（main.py:114），parser
   与报告块不过长，按派单"已有简单入口不无谓改动"处理。
5. **兼容面**：四处样例的 CLI 旗标/默认值/返回码/错误串、公开构造参数、
   旧阶段名别名、量化/预处理/输出契约与 source provenance 不变；SDK 懒导入
   与主机 model-free 模式保持。

## 等价性证据（不只是测试自述）

* **CLI 事实等价**（`cli_equiv_check.py`，从 `git show HEAD:<main.py>` 加载
  旧入口对比工作树）：4 样例 × `--help` 全文 / `parse_args([])` 全部默认值 /
  `--list-models` / `--dry-run` stdout+stderr+退出码 = 16/16 逐字节一致
  （`logs/cli_equivalence.json`）。
* **报告等价**（`report_equiv_check.py`，同一注入 runner/音频 fixture 下分别
  运行 HEAD 入口与工作树入口）：asr 双 chunk `result.json`、paraformer
  preprocess-only `result.json`+`prepared-manifest.json`+feats npy SHA、
  himloco 完整 `report.json`（含 runner 调用数 21+warmup）三者在仅规整
  UTC 时间戳/计时字段与临时根路径后完全一致
  （`logs/report_equivalence.json`）。
* **SDK-free 入口**：每样例 `--help`/`--list-models`/显式 target `--dry-run`
  均主机 exit 0（`logs/sdkfree_*.log`）；ASR auto dry-run、himloco auto
  dry-run 的拒绝路径返回 2 保持。
* **contract checker**：4 样例 0 violations（asr 为 main+cli 两处 CLI-layer
  policy skip，checker 自动识别 `cli.py` 模块级 CLI 边界）。

## 新增测试（先红后绿，日志见外部目录）

* **asr `tests/test_entry.py`（5 项）**：默认 `predict` 返回文本且恰一次
  runner 调用；`return_details=True` 与显式三阶段逐张量相等（text/tensor/
  valid_samples/source_rate，两次请求两次 runner 调用）；不同几何连续调用
  details 逐次独立且模型无残留字段（`vars(task)` 断言）；details 记录本地
  owned；main 循环每 chunk 恰一次 `predict(..., return_details=True)`
  （wrapper 计数 + FakeRuntime 调用数 + report 字段）。红：
  `TypeError: predict() got an unexpected keyword argument 'return_details'`、
  入口未调用 predict（`logs/failingfirst_samples_speech_asr.log`）。
* **paraformer `tests/test_entry.py`（9 项）**：main 不经 `application.run`
  直接驱动 `pipeline.predict(features.tensor, valid_frames)`（run 被 patch 成
  MagicMock 且断言未被调用）；pipeline 抛错时 `failed.json` 保留
  `inference_attempted=true`/`inference_executed=null`（该项红阶段即绿，作为
  行为锁回归如实记录）；`note_runtime`/`prepare_utterance`/`mark_attempted`
  三态迁移/`record_prediction`/`save_features`（npy+双 manifest）/`complete`
  （digest 漂移抛错、completed 收尾）/`run` 兼容组合全流程。红：
  `AttributeError: ... has no attribute 'prepare_utterance'` 等 + main 退出 2
  （`logs/failingfirst_samples_speech_paraformer.log`）。
* **himloco `tests/test_entry.py`（7 项）**：main 可见循环（`execute` 被
  patch 且断言未调用；21 输入 + 2 warmup = 23 次 runner 调用、21 个 dump、
  报告 completed/warmup_completed=2）；`prepare`（报告骨架+复用拒绝）、
  `load_task`（HimLocoTask + runtime 证据）、`record_sample`（dump+digest
  证据+路径冲突拒绝）、`complete`（延迟汇总键集/极值）、`mark_failed`+
  `close`（failed 报告保留 current_source_index、无 latency_ms）。红：
  `AttributeError: module ... has no attribute 'prepare'` + main 经 mocked
  execute 返回非 0（`logs/failingfirst_samples_robotics_himloco.log`）。
* 既有测试全部原样通过（含 himloco `test_cli` 继续经由 `execute` 兼容面、
  paraformer `test_cli`/`test_readable_stages` 继续经由新 main 路径与
  pipeline Mock）。

## 逐样例状态

### samples/speech/asr（30→35 测试）

* 入口：`main.py`（薄）+ 新 `cli.py`；逐 chunk `predict(return_details=True)`。
* 模型类：`asr.py::ASR` + 新 `ChunkPrediction`；规范四方法主线与兼容别名
  （上一批成果）不变，`predict` 增加 details 关键字后默认返回值不变。
* 实际后端：S100/S600 `asr.hbm`（`RuntimeModelRunner`/共享 `SingleArrayRunner`）
  或注入 runner；3503 宽 f32/int SCALE 解码与 CTC/legacy 不变。
* 已运行测试：baseline 30 OK → failing-first（4 项红）→ final 35 OK；CLI/报告
  等价、SDK-free、checker 见上。
* 边界：`soundfile`/板端 SDK 主机不可用属既有边界，测试以注入 runner/fixture
  覆盖；板端推理 not-run。

### samples/speech/kws（16→16 测试，零改动）

* 入口复核：`main.py:114` 已是显式构造 + `predict(audio, rate)`；报告/参数
  块保留在 main（不过长）。终态 16 OK、SDK-free exit 0、checker 0 violations、
  CLI 等价 4/4（文件未改）。
* 边界：PaddleAudio 前端与板端推理 not-run（主机无 paddle 环境，与基线一致）。

### samples/speech/paraformer（87→96 测试，skip 12 不变）

* 入口/应用：`main.py` 可见逐语句循环；`application.py` 助手化 +
  `run`/`execute` 兼容；`pipeline.py`/`stages.py`/`cif.py`/`frontend.py`
  未动。
* 模型类：`ParaformerPipeline`（encoder → predictor → CPU CIF → decoder，
  零 token 跳过 decoder、timings、StageError）不变。
* 实际后端：`runtime.load_runtime` → 三个 `NamedArrayRunner`（S100 HBM，懒
  `hbm_runtime`）；`ParaformerFrontend`（CPU FunASR）在 main 显式构造。
* 已运行测试：baseline 87 OK(12 skip) → final 96 OK(12 skip)；Torch/FunASR
  环境 skip 保持原样、未安装 Torch 去跑未改的数值前端，skip 不当已测数值。
* 边界：12 项 skip 为前端环境缺失（既有事实）；板端推理 not-run。

### samples/robotics/himloco（26→33 测试）

* 入口/应用：`main.py` 可见 warmup+逐输入 `predict` 循环；`application.py`
  `prepare`/`load_task`/`record_sample`/`complete` + `execute` 兼容组合。
* 模型类：`policy.py::HimLocoTask` 未动（规范四方法 + 兼容别名、逐调用
  latency/geometry）。
* 实际后端：X5 `himloco_go2_bayese_1x270.bin`（`RuntimeModelRunner`，注入
  `runtime_factory` 为 host-test seam）。
* 已运行测试：baseline 26 OK → final 33 OK；报告等价（HEAD `execute` vs
  工作树 `execute`）一致。
* 边界：板端推理与机器人控制 not-run。

## 边界与未执行项

* 板端 SDK 链接/推理、真实权重下载、OE/Mapper/HMCT 编译：全部 not-run
  （主机 fixture 通过不替代板端证据）。
* 未修改：其他样例、C++ 侧、根指南、plan/spec、`samples/_shared/`、checker、
  Catalog。multistage 批次报告
  `2026-10-05-all-sample-multistage.md` 已由该批执行器落盘，本批未重写。
* 最终全仓 51 样例行为验收仍 pending Codex 终审（coverage.json 的
  `implementation_evaluation: pending_final_review` 不因本报告改变）。
