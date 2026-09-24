# B6 independent board review — in progress

Review: changes-required. Closed: no. The earlier host review remains historical evidence; it did not establish Python 3.10 board compatibility.

## B6-B1 — evaluator hashing requires an unavailable Python API

On X5 8GB and S100, GitHub checkpoint `73a6de135ddbcc343922a2a31e31c22729f09296` successfully downloaded each sample's exact encoder/decoder pair. Both EfficientSAM and MobileSAM then failed before inference in `samples/_shared/sam_evaluator.py:32`: `hashlib.file_digest` is unavailable in the board Python 3.10 environment, contrary to the documented Python 3.10+ requirement. Four independent run records reproduce the same failure. The earlier metadata projection repair has not yet been exercised by these runs.

The failure evidence also exposes a return-code inconsistency: persisted comparison.json says return_code=2 while the uncaught exception exits the actual CLI with 1. Preserve the failed report and make the public command's documented execution-error code agree with its record.

[Original execution and comparison records](evidence/2026-09-24-b6-initial-board/) include full stdout/stderr and the four failed reports. Model preparation succeeded; inference and numerical comparison did not run. No metadata/payload checks may be declared passed from these failures.

## Remediation and closure requirements

A local Claude Code + GLM task is addressing the exact compatibility and error-propagation issues in an isolated worktree. Require a regression under absence of file_digest, known SHA tests including multiple chunks, failure-report preservation and CLI rc coverage, followed by independent host review and GitHub-delivered reruns on the same two boards. Continue the outstanding X5 4GB/S100P/S600 target matrix after the tools run successfully; do not infer their status from X5 8GB/S100. Conversion, dataset quality and historical latency measurements are separate scopes.

## B6-B1 复验与 B6-B2 新发现

Python3.10修复经独立主机复核（shared124、ResNet52、EfficientSAM19、MobileSAM17）后，作者分支提交 `1bfd8fa`、集成 `6adb0b4`；为避免板测下载无关历史大证据，使用基于既有73a6de1加同一补丁的GitHub检查点 `f888c8fa5eb006690a05f6ee0805bca96fc10ee1`。两板从GitHub获取后复用原模型，新的证据目录不覆盖初次失败。

S100两样例全部6项比较通过：输入、raw、mask、mask_index、IoU、low_res。EfficientSAM14份数组、MobileSAM16份数组全部核验hash/shape/dtype/finite；每case14个部署代码摘要也与f888c8f逐一一致。X5两例真实退出码为2，与JSON一致；已跑过源数值流程，但统一侧encoder调度失败，分别保留7/8份已捕获数组，不能算完整对照通过。[四例完整复测证据](evidence/2026-09-24-b6-python310-board-recheck/) 同时保存成功和失败材料。采集器首轮误把两个sample的数组数都设为14，MobileSAM实际多两侧boxes输入共16；按完整协议修正覆盖断言，未更改板端数组或数值门槛。

B6-B2根因：`sam_runner.py`对X5传scalar priority，但真实SDK需要`{model_name: priority}`；现有分类runner已使用后者。固定X5源SAM也传scalar但吞掉TypeError，所以不能声称其请求调度已成功应用。已派本地GLM修复统一runner并明确evaluator的同调度控制记录；不得修改固定源、静默吞错或伪称原legacy CLI调度正确。改后须X5两样例复测，若evaluator同调度控制有改变，补S100受影响范围。

B6-B1兼容/退出码问题已由真实运行确认修复；B6整体仍changes-required/Closed=no，X5 4GB/S100P/S600尚待执行。

## B6-B2 独立复验：修复确认，完整目标矩阵继续

修复作者提交 `b8de6ed`，板测 GitHub 检查点 `a72f92b41d4265e36b9fc7c5a85d0b15800cee8d`，集成提交 `11e7d28`。独立主机检查 shared128、EfficientSAM19、MobileSAM17 全部通过。X5 8GB 两个 sample 各测试默认 priority=0 与 priority=7；S100 两个 sample 默认入口回归，共六次真实运行全部 rc=0，输入/raw/mask/mask_index/IoU/low_res 六项检查全部通过。

[本轮原始证据](evidence/2026-09-24-b6-scheduling-board-recheck/)包含六个数组包、完整 comparison、两板执行日志和独立校验清单。共90份数组核验 SHA-256/shape/dtype/finite，每例14个代码摘要与板测检查点一致。独立检查实际调度记录：X5 固定源 helper 的 scalar 调用被拒并保留 TypeError，随后 evaluator 明确施加按模型名的 Mapping；统一侧使用同参数且调用成功。S100 原 helper 与显式控制均成功。这里证明的是明示调度控制下的源对照，不声称旧 X5 CLI 自身实现了调度。

B6-B2 在上述范围独立确认修复；尚未覆盖 X5 4GB/S100P/S600，不关闭 B6。运行代码仍在已推送功能/集成分支；develop 保存审核证据并不意味着全部迁移代码已合入。
