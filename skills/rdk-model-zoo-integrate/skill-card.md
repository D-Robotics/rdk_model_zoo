<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# rdk-model-zoo-integrate 治理卡

| 字段 | 内容 |
|---|---|
| Owner | RDK Model Zoo maintainers（角色负责人；合入时按仓库 CODEOWNERS 分配具体 reviewer） |
| Skill 版本 | 1.0.1 |
| Pack 版本 | 1.0.1 |
| 文档许可证 | CC-BY-4.0；frontmatter 的 Apache-2.0 为 Hub 兼容字段，不覆盖内容许可 |
| 脚本许可证 | Apache-2.0 |
| 数据分类 | Skill 文本 public；用户模型、数据与运行证据不因此变为公开 |
| 主要用途 | Use when integrating a custom model artifact or changed I/O contract into an RDK Model Zoo sample, including class-count, shape, wrapper, or cross-platform adaptation. 触发词：自训练接入、替换权重、接口适配。Do not use to implement PTQ/QAT or to review an unchanged sample. |
| 执行边界 | 先读后执行；宿主权限/用户授权优先；不承诺自动安装或后台能力 |
| 验收状态 | 本地结构与工具测试见交付验证报告；Agent 行为/真实板卡/生产 Hub 验收分别记录，不由此卡宣称通过 |
| 已知风险 | 上游规则漂移、未安装依赖、语义误判、产物/runtime 不匹配、把局部证据泛化 |
| 更新触发 | 相应仓库规范、目录/Manifest schema、工具链交接或 Hub 契约变更 |

## 评测

使用随包 `evals/tasks.yaml`；正向、负向、权限和故障场景分别判断。测试定义不等于已执行结果。

## 变更审阅

新增或实质修改行为时增加一个能暴露旧行为缺陷的场景；先记录 baseline，再加载新 Skill 对比。不得只根据 contains 关键词判断任务成功。
