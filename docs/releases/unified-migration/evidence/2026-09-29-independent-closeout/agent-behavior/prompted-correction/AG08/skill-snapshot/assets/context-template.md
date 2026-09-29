<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# 仓库上下文记录模板

模板不是采集结果。未知值使用“未知/未读取”，不填写猜测的提交号或路径。

| 字段 | 填写要求 |
|---|---|
| 用户目标与交付来源 | 原请求/Issue/PR 的链接或摘要及精确范围 |
| SKILL_ROOT | 当前可信 Skill 安装目录；不是目标模型仓库 |
| REPO_ROOT / repository | 本地实际目标根、远端身份（脱敏）或明确的远程只读来源 |
| commit / branch / tag / dirty | 命令或 API 的真实输出；detached HEAD 合法 |
| 平台、SoC、OS/runtime | 分开记录声明与观测；未上板则不要填“探测通过” |
| sample / variant / task / language | 唯一定位或明确候选集合 |
| 规范和需求 | 每项来源路径/章节、版本、作用域、required/convention 等 |
| 目录与资源入口 | 该目标实际存在的 docs、Manifest、runtime、utils |
| 需要阅读的文件 | 只列与当前目标相关的文件及阅读理由 |
| 主 Skill / 子任务 | 选一个主工作流，列明确交接点，不调用全部 Skills |
| 权限与副作用 | 已批准和未批准的动作 |
| 未决事实 | 阻断哪一步、继续哪些不依赖步骤 |

本记录可以在对话中输出。持久化前选择用户同意的工作目录；不要默认把临时记录提交进 sample。
