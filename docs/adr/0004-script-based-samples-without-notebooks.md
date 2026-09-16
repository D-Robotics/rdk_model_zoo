# 使用脚本交付 Sample，移除 Notebook

状态：accepted，用户选择 Python 默认方案，并明确要求 Notebook 全部移除。

新主线以 Python 为默认运行入口，保留已有 C/C++ 能力，新增双语言实现按 Sample 的交付要求确定。Notebook 不再作为运行入口或教程文件保留；迁移时先把独有执行逻辑整理为脚本、教学内容整理为 Markdown，原 Notebook 留在历史 tag。这样人和 Agent 可直接运行、检查和复现相同流程，同时避免代码单元状态造成隐含依赖。

这是对过渡兼容范围的明确限定：旧脚本可以提供适配，但不为保留 Notebook 文件再维护一套入口。本轮仍在需求阶段，尚未删除任何 Notebook。
