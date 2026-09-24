# B3 板测工具独立评审

Review: changes-required. Board: not-run. 这是未完成整改检查点，不是板测通过或作者完整交付。

GLM在整改期间遇到五小时额度上限（服务返回2026-09-24 20:56:39后恢复），进程已终止，保留全部文件。独立重跑28项测试通过，但下面的额外反例成立。

## B3-TOOL-R1：预加载依赖绕过固定源隔离

在进程内先import根目录`utils.py_utils.file_io`再调用工具加载legacy，源模块的file_io仍指向根目录对象。工具替换父package却未移除已缓存的子模块；真实依赖记录中file_io/visualize的matches_pin=false，但run_comparison未据此拒绝执行。现有测试只覆盖干净sys.modules，因此未捕捉此问题。

要求：保存并暂移除完整相关模块集合，从已验证源闭包加载；finally完整恢复原对象；在实际推理前校验真实加载文件摘要。补充预加载污染与完整恢复的回归用例。判据和sample算法保持不变。

[反例与当前代码摘要](evidence/2026-09-24-b3-partial-independent-review.json)。客户README与B3台账不据此改成passed。
