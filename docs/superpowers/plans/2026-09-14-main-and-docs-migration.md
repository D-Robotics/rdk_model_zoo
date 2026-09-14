# Main 分支与文档站仪表盘整合 Implementation Plan

**Goal:** 两个仓库均在本地 main 维护；模型仓库统一容纳 X5/S/X3 实现并发布版本化目录数据，文档站承载仪表盘。
**Architecture:** 第一阶段保留平台内部路径，以 platforms/x5、platforms/s、platforms/x3 隔离不兼容运行时；模型端工具生成 catalog 数据包，文档端通过锁文件消费，不依赖模型源码 checkout。已有 tags、远程默认分支、下载地址不修改。
**Tech Stack:** Python / shell samples, TypeScript catalog publisher, Vite dashboard, Docusaurus manual.
**Spec:** 用户在当前会话批准的两仓库职责与 main 统一方案。

- [ ] 保存当前 UI/清单与 S 分类修复的本地整合基线，不覆盖原分支和工作区。
- [ ] 模型 main 按 platforms 收纳各平台完整运行环境、文档、清单、子模块声明；提供平台中立入口。
- [ ] 将数据构建及 schema 归模型仓库，发布固定版本/校验和 catalog 包；移除 X5 发布特权。
- [ ] 仪表盘源码迁入文档 main，与文档构建集成；使用可验证锁定数据包，开发支持本地导入。
- [ ] 文档 Benchmark 手写重复数值改为仪表盘入口，保留方法说明及旧路由兼容；中英导航同步。
- [ ] 校验平台迁移路径、数据完整性、模型数据构建、仪表盘测试、Docusaurus 构建与本地预览。
- [ ] 本地 main 提交可审查结果；不推送，不改远程默认分支，不部署线上。
