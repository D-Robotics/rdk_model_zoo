# Develop 交付成熟度验收与优化

用户目标：交付完全完成重构的 develop，使其源码、文档、验证和发布记录成熟到可以直接作为 main 的对外入口。沿用已认可的可读模型架构，不重新设计框架。

当前基线：本地 worktree `codex/readable-model-examples-20261001`，`4ef21d6bec9f51e26c0f0e23ca423186ffa13705`；本地 develop 为 `bfbe6aafcea7c234a322436c128f7362b46f746a`。重新观察的 origin/develop 为 `eed26ce610d7fba03a68d1c0ee6e62603cd9b85d`，已是当前工作分支祖先；远端当前不存在 main，默认分支仍为 rdk_x5。本轮准备并同步 develop，不擅自创建正式发布、tag 或切换默认分支。

## 延续的用户约束

- 本地 Claude Code + GLM 实现，Codex 负责方案、派发、独立评审和验证、Git 整合。
- 不去服务器，不执行 SSH。ACT/Pi0 独立 gitlink 不纳入重构、不初始化以扩大测试。
- 51 本仓 Sample（49 Python、2 原生 C++ LLM）。薄 main、可见 predict、可读阶段、真实局部差异和旧接口兼容保持。
- 历史 platforms 只在 Git 历史，不恢复进活动源码。现有导出/量化 README 是可信配方，继承来源，不执行权重下载、导出、校准、OE/Mapper/HMCT 或板测；这些项目不重新变成交付阻塞项。
- 主机合成输入、数学、绑定、CLI 和原生 SDK 替身测试在范围内。支持声明、主机结果、历史板测和本轮板测须分开。

## 已发现的交付缺口

1. GitHub 只有契约和 Catalog 工作流，缺少全 Sample 回归；契约只对 develop push，Catalog 只对 main push，换主线后门禁覆盖不同。
2. 新 Gemma 应用测试硬编码 Homebrew 路径、无条件 -liconv；Gemma/MiniCPM JSON 头默认指向仓库外 .coordination，主机结果依赖个人机器状态。
3. 上轮回归只覆盖各 Sample 顶层 tests，YOLOE conversion/evaluator 子套件和额外原生 CTest 需明确纳入可复跑流程。
4. 根/索引/贡献文档仍有旧迁移进行中状态，容易与当前完成范围混淆。缺少统一源码 VERSION 和清晰的 main 提升/回退说明。
5. Catalog 工作树链接固定到 develop；准备 main 时应让生成物绑定实际源码提交，保留历史平台制品版本和 Benchmark 原条件。

## 完成条件

- 本仓所有适用 Python 测试目录自动发现、独立进程执行；VLA 明确排除。遗漏、测试 0 项、子进程异常、非声明 skip、源码漂移均不能报告成功。
- Gemma/MiniCPM 主机依赖从标准系统/pkg-config 或显式环境变量发现；不存在个人路径要求，macOS/Linux 链接选项正确。缺依赖与编译错误区分，CI 必须执行原生替身测试。
- CI 在 develop/main 与 PR 同等覆盖契约、全部主机回归、现有主机原生 CTest、Skills 与 Catalog；明确固定 Python/Node 支持范围与依赖，完整 Git 历史用于固定源比较。
- 全新 Python 依赖环境和独立完整 Git clone 复跑同一维护者命令；无模型/SDK、无 VLA 子模块、从 clone 外运行 model-free 入口。既有依赖复用须如实记录；Linux 结果由实际 CI 证明。
- 用户指南准确描述统一架构、目标/模型选择、最短运行、源码集成、导出/编译可信配方、每类特例；历史审查不冒充当前状态，中英对应。Agent 与用户使用同一真实入口。
- 统一源码初版候选 VERSION 为 2.0.0，作为本地源码版本而非已发布 tag；平台制品 VERSION 和 Skills 1.1.0 各自保持。源码 tag 使用 zoo-vX.Y.Z，Skills 不因源码版本而重标；不执行 tag/Release/Hub/默认分支修改。
- Catalog 新工作树产物链接解析 HEAD 为不可变完整提交，不硬编码开发分支；历史来源仍固定原提交/tag，资产 URL/SHA/Bench 数值不改写。
- Codex 独立核对最终 diff、全部适用门禁、支持/验证矩阵和源码证据；正式更新 develop 前确认两处 checkout 干净、远端无新增分叉，保持全部历史。
- 将完成的重构和本轮修复合入本地 develop、同步远端 develop；核对远端哈希和对应 CI 实际状态。main 对外提升流程可执行、不要求再改模型代码；正式发布由用户以后操作。

## 架构与执行

新增 tools/host_validation 只面向维护者测试，不是用户全仓推理 CLI、不引入模型框架或逐 Sample workflow 配置。测试目录与 CTest 项目从真实源码布局获得，机器报告绑定实际命令/提交/源码摘要，不以人工状态替代执行结果。

分三包：原生测试依赖可移植性；完整维护者验收与 CI；用户/发布文档和不可变 Catalog 引用。包内先复现问题，修改、回归。独立且不重叠的文件可并行；Codex 串行评审、按路径提交、集成。最后以实际 develop 快照验证交付，而非仅评审工作分支。
