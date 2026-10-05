# 全部本仓 Sample 可读 Runtime — speech_policy_native 批次报告

> 批次时点记录：最终代码已在本地提交，本文保留执行时的计数和状态。
> 全 51 Sample 的最终源码/主机验收结果见 [Codex 独立验收](2026-10-05-all-sample-codex-review.md)。

日期：2026-10-05
基线：`c1510ede652d30d83eeb691ff0e90b3f735a70a4`（分支 `codex/readable-model-examples-20261001`；批次进行中工作树 HEAD 已被 Codex 依次推进至 `9bc2fdc0` 以提交其他批次，本批次未依赖该推进，改动只落在下述五个样例目录与本报告）。
执行器：本地 Claude Code 2.1.276 + GLM `glm-5.3[1m]`；主机 Python 为 `rdk_model_zoo/.venv/bin/python`（3.14.7）。
设计依据：`docs/superpowers/specs/2026-10-05-all-sample-readable-runtime-design.md`；范例为 ResNet `classify.py`。C++ 侧按用户补充范围：Gemma 仅整理交互式入口 main.cpp 的应用边界，不重构引擎算法；MiniCPM5 若已符合原生标准则复核保留。

> **批次快照声明**：本报告是本批次的执行快照。asr/kws/himloco 的入口（及
> paraformer 入口）其后由 speech/policy 入口批次继续重写（本报告成文时仍在
> 进行中，如 asr 逐 chunk `predict(return_details=True)` 已出现在工作树）；
> 本报告的测试计数与"入口原样达标"等结论均为批次时点记录，**最终状态以
> Codex 终审报告为准**，本文不构成行为验收结论。
外部证据日志目录：`local-execution/20261005-all-sample-readable-runtime/speech_policy_native/`（baseline-/tdd-fail-/post-/final- 日志、contract- 日志、gemma-chatapp-first/second-run.log、progress.json）。代码与测试未提交，由 Codex 按路径评估提交。

## 统一改动模式（Python 三样例：asr / kws / himloco）

三者的任务类与入口在基线时已具备真实三阶段（`pre_process`/`forward`/`post_process`
与 `predict`）且 main 已显式构造任务对象，故本批次不做结构性重写，只补齐规范
命名并保持一切算法、CLI 与结果契约不变：

* **任务类**（`asr.py ASR`、`kws.py KWS`、`policy.py HimLocoTask`）：实现改名
  为规范主线 `preprocess`/`infer`/`postprocess`，`predict` 类内串联规范方法；
  旧拼写 `pre_process`/`forward`/`post_process` 变为薄兼容委托（同一实现、
  两个名称，无第二套实现）。构造参数、frontend（ASR 的
  mono/Fourier 重采样/z-score/补零，KWS 的 PaddleAudio fbank 与固定
  60000/373/80 参数）、observations 数据（HimLoco 的 270 值打包与逐调用
  geometry/latency 记录）逐字节保留。
* **入口**：`kws/main.py`（构造 `KWS(...)` 后 `predict(audio, rate)`）与
  `himloco/main.py → application.execute → HimLocoTask.predict` 原流程已符合
  标准，未改动。`asr/main.py` 的逐块流式循环改调规范方法并加注释说明：报告
  需要每块自身的 valid_samples 几何，故 CLI 每块组合三个公开阶段，
  `task.predict(waveform, rate)` 是等价单调用 API（流式语义与 report 字段
  不变）。
* **README/README_cn（runtime/python 级）**：三阶段说明、集成示例与阶段表
  改用规范名，并加"旧名称仍是可导入薄别名——同一实现，两个名称"说明；
  参数表、troubleshooting、source provenance 未动。
* **测试（先红后绿）**：每样例新增 `CanonicalStageTests` 3 项——规范名与旧
  别名结果逐字段一致（含张量 owned/contiguous 与 `infer` 返回即注入 fixture
  的纯度）、`predict` 与显式规范三阶段一致 + runner 恰好按调用计数、连续
  调用 geometry/state 不串扰。实现前运行记录 `tdd-fail-*.log`
  （`AttributeError: object has no attribute 'preprocess'`），实现后转绿；
  既有全部测试同时回归通过。

## 逐样例状态

### samples/speech/asr

* 模型类：`samples.speech.asr.runtime.python.asr.ASR`；规范四方法
  `preprocess`/`infer`/`postprocess`/`predict`，兼容别名三个。
  实际后端：S100/S600 `asr.hbm`（`RuntimeModelRunner`）或注入 runner；3503
  宽 logits 的 f32/int SCALE 双解码路径与 CTC/legacy 模式不变。
* 入口：`runtime/python/main.py`（逐块流式 + 规范阶段；chunked report 契约
  不变；`--dry-run/--list-models/--help` 主机可用，已验证）。
* 测试：基线 27 OK → 终态 30 OK（+3，红绿日志已录）。contract checker
  0 violations。

### samples/speech/kws

* 模型类：`samples.speech.kws.runtime.python.kws.KWS`；规范四方法 + 兼容
  别名。实际后端：S100 `kws.hbm`；PaddleAudio fbank 前端与 [0,1] 概率契约
  不变（无二次 sigmoid）。
* 入口：`runtime/python/main.py`（构造 `KWS` → `predict`；未改动流程）。
* 测试：基线 13 OK → 终态 16 OK。contract checker 0 violations。

### samples/robotics/himloco

* 模型类：`samples.robotics.himloco.runtime.python.policy.HimLocoTask`；规范
  四方法 + 兼容别名；`PreparedInput`/`RawOutputs`/`HimLocoResult` 逐调用
  状态与 latency 语义不变。实际后端：X5 `himloco_go2_bayese_1x270.bin`。
* 入口：`main.py → application.execute`（warmup + 逐输入 `predict` + action
  dump/report，未改动）。
* 测试：基线 23 OK → 终态 26 OK。contract checker 0 violations。

### samples/llm/gemma4-e2b（原生 C++，无 Python runtime —— 不新增假 Python 支持）

应用边界整理，引擎与算法零改动：

* **新增 `inc/gemma4_chat_app.hpp` + `src/gemma4_chat_app.cpp`**：应用
  facade `gemma4::chat::InteractiveChatApp`。构造函数按历史顺序打印 banner
  与加载信息并构造 `VisionEngine`、`TextEngine`、`TokenizerBridge`（Vision
  先于 Text，成员以 `unique_ptr` 在构造体内建立以保持输出顺序）；
  `Run()` 承接从旧 main.cpp 逐字节搬移的 REPL：UTF-8/GB18030 终端归一
  （iconv）、`/help` `/reset` `/context` `/image` `/quit` 命令、对话历史
  JSON、4096-token 预算下的最旧轮次裁剪、前缀失配重置、流式回显与耗时
  统计。模型执行全部委托真实运行时：图文轮次 `LoadImage → PredictVision →
  VisionEngine::Infer` 与 `BuildPromptHidden` 注入 +
  `TextEngine::ContinueGenerateStream`，纯文本轮次直接
  `ContinueGenerateStream`；`Generate`/`ContinueGenerateStream` 等既有接口
  原样保留，未拆引擎、未造统一假协议。
* **`src/main.cpp` 重写为薄入口（≈90 行）**：gflags 解析 → `$GEMMA4_HOME`
  默认路径解析 → `--max_tokens`/`--min_response_tokens` 校验（错误文案与
  返回码 2 不变）→ 构造 `InteractiveChatApp` → `app.Run()`；异常面
  `ERROR: <what>` + 返回 1 不变。flag 名/默认值/用法串与基线逐字节一致。
* **CMakeLists**：`main` 目标源改为 `main.cpp + gemma4_chat_app.cpp`；
  `gemma4_server`/`gemma4_demo`/`gemma4_text_bench`/`gemma4_golden_verify`
  与 `gemma4_runtime` 静态库目标未动。
* **主机可验证能力（新增 `tests/test_cpp_chat_app.py` + native 支撑）**：
  编译生产 `src/gemma4_chat_app.cpp` 与真实 `src/main.cpp`，链接
  `tests/native/chat_app_doubles.cpp` 引擎替身（明确标注：不加载 HBM、不跑
  BPU、不分词、不解码图片）、`tests/native/sdk_fixtures` SDK 头替身、
  `tests/native/app_stubs/` 中明确标注的 tokenizers-cpp / OpenCV 头编译桩，
  以及 nlohmann/json（`GEMMA_JSON_INCLUDE`）与主机 gflags（真实链接）。
  8 个 REPL 场景驱动重定向 stdin 验证会话逻辑：引擎构造信息与顺序、流式
  回显（"WW"、tok/s 行、`[context] prompt=`）、`/reset` `/context` 输出、
  图文轮次接线（`LoadImage`/`PredictVision`/`Infer` 各恰一次、prompt hidden
  注入标志、430080 features 提示）、5000-token 超长 prompt 拒绝且不生成、
  最旧轮次裁剪（裁剪后 prompt 必须回到预算内且 ResetSession）、
  `--rebuild_context_each_turn` 每轮重置、跨轮上下文增长；1 个场景用原始
  GB18030 字节验证终端转换提示。3 个入口测试链接真实 gflags 验证薄
  main：构造→`/quit` 干净退出（banner/加载/KV 行）、`--max_tokens=-1`/
  `--min_response_tokens=0` 返回 2、`--max_tokens=512` 到达会话输出。
  编译警告面 `-Wall -Wextra -Werror` 通过。
* **README/README_cn（runtime/cpp 级）**：目录树加入
  `gemma4_chat_app.hpp/.cpp` 并改 main 描述；新增"交互式对话入口：应用
  facade 与模型类"一节，明确 facade 是应用类而非模型类、不存在执行控制台
  IO 的 predict 型引擎 API、引擎从不隐式打印；主机回归一节登记新的
  chat-app 检查入口、替身/桩边界与 not-run 范围。
* 测试：基线 30 OK → 终态 41 OK（+11 全为新 chat-app 检查；既有 30 项含
  launcher、KV、text/vision tensor、stages、golden 全部回归通过）。
  contract checker 0 violations。新生产源有可复现主机编译/链接/运行检查
  （`test_cpp_chat_app.py`），非仅旧 text-engine 测试。

### samples/llm/minicpm5-2b（复核保留，零改动）

按用户补充范围复核：`runtime/cpp/src/main.cc`（44 行薄入口：gflags →
构造 `minicpm5::MiniCPM5` → `Generate(prompt, new_chat)` → RESULT JSON 展示）
与 `inc/minicpm5.hpp`/`src/minicpm5.cc`（公开 `pre_process`/`infer`/
`post_process` 阶段函数 + `Generate` 串联、`validate_metrics` 拒绝非有限/
负值、配置与文件 IO 在 `runtime_config.cc`）已符合原生标准，runtime README
两语对该链路的描述与代码一致。为不为增加改动而引入假 Predict API，本批次
未修改任何文件；既有主机覆盖（`tests/native/s600_stages.cpp` 等以 SDK 替身
编译生产 `minicpm5.cc`+`runtime_config.cc`、`LegacyCliTests` 执行真实 legacy
`main.cc`）维持原状，终态 20 OK 复核通过。

## 汇总

| 样例 | 语言 | 基线 | 终态 | contract | 备注 |
| --- | --- | --- | --- | --- | --- |
| speech/asr | Python | 27 OK | 30 OK | 0 violations | 规范阶段 + 流式 main 注释 |
| speech/kws | Python | 13 OK | 16 OK | 0 violations | 规范阶段；入口原样达标 |
| robotics/himloco | Python | 23 OK | 26 OK | 0 violations | 规范阶段；入口原样达标 |
| llm/gemma4-e2b | C++ | 30 OK | 41 OK | 0 violations | 应用 facade + 薄 main + 主机场景检查 |
| llm/minicpm5-2b | C++ | 20 OK | 20 OK | 0 violations | 复核保留，零改动 |

验证命令（均自仓库根、每样例独立进程，日志在批次外部目录）：

* 基线/终态：`$VENV -m unittest discover -s samples/<sample>/tests -v` →
  `baseline-*.log`（5/5 exit 0）/ `final-*.log`（5/5 exit 0）。
* 红绿循环（Python 三样例）：新增测试实现前运行 → `tdd-fail-*.log`
  （AttributeError: no attribute 'preprocess'）。
* 契约检查：`$VENV tools/sample_contract/check.py --sample samples/<sample>`
  → `contract-*.log`（5/5 exit 0）。
* 主机 SDK-free 入口：asr/kws/himloco 的 `--help`、`--list-models`、显式
  target `--dry-run` 均验证可用（asr s100、kws s100、himloco x5）。
* Gemma chat-app 主机检查：`$VENV -m unittest discover -s
  samples/llm/gemma4-e2b/tests -p test_cpp_chat_app.py -v`（11/11 OK）。

每样例独立进程 discover，未与其他样例混跑；未运行其他批次或全仓回归
（其他执行器并行修改中）。本批次修改文件共 17 个 + 新增 7 个（见 git
status：三 Python 样例各 asr/kws/policy + main(仅 asr) + tests + README×2；
gemma main.cpp/CMakeLists/README×2 + 新增 chat_app 头源、测试 5 件套），
全部位于批次专属目录；本报告为本批次唯一的 docs 写入。

## 未执行（not-run）

* 板端推理与真实 SDK 链接：gemma4-e2b 的 hb_dnn/hbucp/OpenCV/tokenizers-cpp
  真实栈链接与板端生成（vision/text/server/demo/bench 全部模式）、
  minicpm5-2b 的 OELLM 运行时链接与 S600 生成，均 not-run；主机替身/桩/
  注入 runner 的通过不构成以上证据，也不代替板端验证。
* 真实权重下载、真实 ONNX 导出、OE/Mapper/HMCT 编译、量化精度与数据集
  评测；conversion 配方与量化事实未动。
* Catalog build/check、全仓 shared 测试与干净 checkout 入口复现属于最终
  集成阶段，不在本批次范围。
