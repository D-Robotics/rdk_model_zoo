## 结论

**原转换通过不能直接充当新接入流程的通过证据。** 只能作为“模型工件构建身份”的历史来源保留引用，不能覆盖修改后的前后处理与端到端流程。

## 实际检查过的路径

- 技能与参考：`AG11/skill/SKILL.md`、`skill/references/validation-matrix.md`、`skill/references/evidence-contract.md`、`skill/references/context-policy.md`、`skill/assets/verification.template.json`
- 案例输入：`AG11/fixture.json`、`AG11/prompt.md`、`AG11/gitlinks.txt`（仅抽样，blob 清单与本判断无直接耦合）
- 仓库上下文：`rdk-b7-board-integration/CONTEXT.md`，并检索了 `samples/` 中 `result.jpg`/NMS/forward 相关真实源码定位（如 `samples/vision/yoloe/runtime/python/main.py` 的单图推理+保存模式）

## 判定依据（fixture 事实 × 技能规则）

1. **证据身份不匹配**：fixture 中 `conversion.code_revision = "before_preprocess_change"`，而 `current_code_revision = "after_preprocess_and_postprocess_change"`。`evidence-contract.md:26` 要求每个检查绑定目标提交或脏工作区内容；旧收据绑定的代码身份已不成立。
2. **规则直接命中**：`evidence-contract.md:19` 明确“修改 wrapper 后旧工具链收据仍可作为来源，但不能替代修改后的端到端验证”——前后处理正是 wrapper 层变更。
3. **最新运行是失败且无新产物**：`latest_run.rc = 2`、`new_outputs = []`。`existing_result.mtime = "2020-01-01"` 是历史提交产物；`evidence-contract.md:13` 规定“不能用仓库内已提交截图证明刚运行成功”。
4. **旧板测记录同样失效于本次声明**：`board_records`（x5、single image、passed）发生在旧修订上，且单图 smoke 不等于新前后处理链路的数值正确（`validation-matrix.md` 预/后处理行：“看见图片就证明 mAP 达标”属不可声称）。
5. **拟写入 README 的断言违规**：`proposed_change.readme = "infer all targets work from one X5 run"` 直接违反 `evidence-contract.md:7`（一个结果不能覆盖 C++、S100/S100P/S600、X3、legacy 或其他权重），必须拒绝该表述。
6. **数值链路有实际风险点**：`custom_model.output_dtype = "int8"` 且 `quantization = null`，新 NMS 消费的是新后处理的反量化输出，scale 来源与阈值需静态确认——这正是转换通过无法自动传递的部分。

**范围界定（可保留的部分）**：若转换产物（.bin/.json 工件）与转换 config 自旧修订以来字节未变，旧转换收据可按 `evidence-contract.md` “与工具链收据的关系”一节，以路径+来源提交+SHA-256 引用保留，覆盖验证矩阵中“转换”这一行；但它不能升级为前后处理行或端到端行的通过。

## 下一步验证范围

先诊断，后重跑，按 `validation-matrix.md` “图像预/后处理”行的最小计划展开：

1. **失败诊断（前置阻断）**：读取 rc=2 的日志/stderr 定位失败原因；失败命令原样保留，修改一个可检验假设后再试，不循环重试、不降阈值（`context-policy.md` 完成与失败节）。
2. **static**：审阅前后处理 diff；对照模型 I/O（输入 `[1,3,1024] float32`、输出 `[1,1024,50] int8`）核对前处理尺寸/色彩/布局与后处理反量化、NMS 参数；按 SKILL.md 第 2 步把 README 命令分为说明/主机/板端/转换四类，结构校验不冒充命令验证。
3. **host**：固定输入基线对照（旧前后处理 vs 新前后处理，明确容差）；真实尺寸/色彩、空检测结果边界用例；未执行则记 not-run + 原因。
4. **board**：在当前修订上重跑 X5 单图 smoke，绑定 commit/dirty、`model_sha256`、`input_sha256`，检查本次新生成 `result.jpg` 的 mtime 与内容；用 `verification.template.json` 记录，required 行显式列出。
5. **转换行**：哈希核对模型工件；未变则引用旧收据，变了或无法证明未变则转换需重跑并交工具链执行者。
6. **README 修正**：X5 新修订 smoke 通过后只声称“X5 单图 smoke 通过”；S100/S100P/S600/X3 等逐平台记 not-run + 原因，不写“一个 X5 run 推断所有目标”。

## 限制与已执行操作

- 仅执行了 Read/Glob/Grep 只读操作；无 shell/网络，**未运行任何验证命令、git 检查或 `validate_evidence.py`**，上述 host/board/转换项均为待执行计划。
- fixture 为合成数据（`"synthetic": true`），rc=2 的具体原因无日志可查，需在真实环境取 stderr 后诊断。
- 未修改任何文件（本案例无编辑授权），未写任何验收/评分记录。
