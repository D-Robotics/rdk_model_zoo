## 更正

**错误承认**：我此前声称目标仓库根目录没有 README，这是错误的。实际读取确认，仓库根同时存在 `README.md`（英文，第 1–124 行）与 `README_cn.md`（中文，第 1–124 行），二者互为语言链接，内容对应。

**README 关键事实**（两份一致）：

- 本仓库为 RDK Model Zoo（`develop` 是 X5/S 全量整合中的开发分支，非客户发布版），提供模型准备、前处理、BPU 推理、后处理及应用验证示例；每个 Sample README 是用户与 Agent 共同的操作入口。
- 统一 Sample 共 51 个：45 视觉、3 语音、1 机器人策略、2 大模型；ACT/Pi0 以固定 Git 子模块集成，独立于 51 个样例。
- 目录结构：`samples/`（统一实现）、`platforms/{x5,s}/`（保留源材料）、`platforms/x3/`（历史 X3 发行）、`docs/release/`、`docs/sample-standards/`、`datasets/`、`utils/`、`tools/`。
- 制品与目标：X5 用 `.bin`（bayes-e），S100/S100P/S600 用 `.hbm`（nash-e/m/p），互不可替代；BPU 推理需匹配板端 SDK 和硬件，主机 dry-run 不等于推理验证。

**保留原有范围判断**：X3 为历史资源的结论不变——两份 README「板卡、制品与环境」表均明确标注 RDK X3 为「历史目录」，保留原始文档与发布记录，**不属于本轮新增适配目标**（`platforms/x3/` 仅作历史发行保留；X3 上游无许可文件，也未补造）。因此新后端/新适配工作范围仍限于 X5 与 S 系平台。

## 边界声明

本次仅使用 Read 工具只读读取了 SKILL.md、目标仓库根 `README.md`、`README_cn.md` 三个文件。未运行 skill 自带的 `inspect_repo.py` 盘点脚本，未执行任何 shell、git、网络、板卡、模型转换或量化操作；除上述文件外未访问其他目标源码。
