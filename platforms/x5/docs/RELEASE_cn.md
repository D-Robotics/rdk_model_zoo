# RDK Model Zoo 发版规范

本文规定 RDK Model Zoo 的发版流程。当前阶段保持流程简洁，不要求设置 RDK 板卡测试门禁。自动检查用于校验发版 Manifest 并构建目录数据包，不代表模型已经通过板端运行认证。

三条硬件版本线各自独立发布，互不阻塞。聚合目录在 `main` 分支上从三份平台目录树构建，每个清单都读取自身 `platforms/<id>` 检出。不存在跨平台的锁定文件，也不存在发布顺序要求：任何版本线都不读取其他版本线的活动分支，任何分支都不部署网站。GitHub 仓库级 Latest 指向 X5；三个硬件版本仍独立维护。

每次发布须按实际条目重新核对汇总统计，从 Tag 原样导出两份 YAML 附件，并附上它们的 `SHA256SUMS`。该文件只校验清单附件，不代表外部模型文件已有哈希。已推送的准备版 Tag 保留不动，元数据修正使用新的补丁 Tag。

## 1. 版本线与命名

X5、S、X3 三条版本线独立维护。每条版本线使用 [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html)：`MAJOR.MINOR.PATCH`。

| 平台 | 发版分支 | 稳定版 Tag | 发版示例 |
| --- | --- | --- | --- |
| X5 | `rdk_x5` | `x5-vMAJOR.MINOR.PATCH` | `x5-v1.0.0` |
| S | `rdk_s` | `s-vMAJOR.MINOR.PATCH` | `s-v1.0.0` |
| X3 | `rdk_x3` | `x3-vMAJOR.MINOR.PATCH` | `x3-v1.0.0` |

候选版本在后面追加 `-rc.N`，例如 `x5-v1.0.0-rc.1`。平台分支中的 `VERSION` 文件只写不带平台前缀的版本号，例如 `1.0.0`。

三条平台版本线不使用统一的仓库全局版本号。Tag 同时标识平台，以及生成该版本的准确提交。

## 2. 发版必备文件

每个发版提交都必须更新或确认以下文件：

- `VERSION`：平台版本号。
- `CHANGELOG.md`：面向用户的变更和已知限制。
- `docs/release/models.yaml`：该版本的模型 manifest。
- `docs/release/benchmarks.yaml`：该版本已经记录的性能与精度实测结果。
- `docs/releases/<tag>.md`：用于 GitHub Release 的发版说明。

Manifest 记录本版本提供的模型和资源、示例路径、下载脚本或 URL、文件格式，以及已知的校验和。未知的 SHA-256 必须明确写为 `null`，不能猜测或伪造。只要 manifest 中存在 `sha256: null`，发版说明就必须披露校验和覆盖不完整。

模型 Manifest 与 Benchmark Manifest 描述已发布的源码资源和文档中已经记录的测量结果。YAML Manifest、GitHub Release 资产和模型 README 是权威来源。缺失的性能或精度指标表示该版本尚未完成或记录对应实测，不表示数据受限或保密。除来源文档明确记录的条件外，这些内容不代表板端运行或兼容性认证。

## 3. 人工发版流程

1. 选择一条平台分支，并确认发版内容属于该平台。
2. 更新 `VERSION`、`CHANGELOG.md`、`docs/release/models.yaml`、`docs/release/benchmarks.yaml` 和 `docs/releases/<tag>.md`，两个 Manifest 都必须填写新的 Release Tag。
3. 复核 Manifest：示例路径和下载脚本必须存在，URL 必须正确，未知校验和必须写为 `null`，每条 Benchmark 必须引用不可变的仓库证据；检查发版文件中的 Tag、分支、平台和版本一致。
4. 从干净检出中复核源码差异，并执行相关 Sample 与 Manifest 检查：

   ```bash
   git diff --check
   ```

   确认工作区干净，并确认目标 Tag 尚不存在。
5. 将复核后的发版提交合入对应平台分支。Tag 必须指向该分支的最新提交。
6. 创建并推送 annotated Tag：

   ```bash
   git switch rdk_x5
   git pull --ff-only origin rdk_x5
   git tag -a x5-v1.0.0 -m "RDK Model Zoo X5 v1.0.0"
   git push origin x5-v1.0.0
   ```

   S 或 X3 平台需要替换对应的分支名、Tag 和提交说明。禁止创建 lightweight Tag。

7. 使用推送后的 Tag 创建 GitHub Release，使用对应的发版说明文件，并上传两个 Manifest 附件：

   ```bash
   gh release create x5-v1.0.0 \
     "docs/release/models.yaml#models.yaml" \
     "docs/release/benchmarks.yaml#benchmarks.yaml" \
     --title "RDK Model Zoo X5 v1.0.0" \
     --notes-file docs/releases/x5-v1.0.0.md \
     --verify-tag
   ```

8. 平台发布完成后，将平台分支合入 `main`。
9. 复核 GitHub Release、Release 附件、Tag、分支提交、`VERSION` 与仓库 Manifest 是否指向同一个平台版本。项目流程需要时，在变更记录或发版记录中记录 Release URL 和提交号。

## 4. Tag 和 Release 的不可变性

Tag 发布后不得移动、强制推送或删除。已发布版本不能复用到其他提交。如果发版包含错误，应保留原 Tag，将 GitHub Release 标记为撤回或被替代，并使用修正后的文件和发版说明发布新的补丁版本。

历史版本线或历史快照使用 `archive/<platform>-v<version>` 的归档 Tag，例如 `archive/x5-v0.0.1`。归档 Tag 同样使用 annotated Tag，并且不可修改。

## 5. 补丁、撤回和前置条件

补丁版本用于修复文档、下载元数据、脚本或其他向后兼容的发版缺陷。补丁版本必须重新完成必备文件复核，并使用新的补丁 Tag，例如 `x5-v1.0.1`。

撤回版本时，应在 GitHub Release 说明中写明原因，保留已发布 Tag 以便追溯；用户需要修正版时，再发布替代版本。撤回不能通过静默改写历史完成。

本简化规范不要求 self-hosted runner、新增基准测试任务或发版前 RDK 板卡测试。目录流程只校验现有 Manifest 证据并构建目录数据包，不运行模型或数据集。发版说明不得声称完整模型集合已经通过板卡测试。实际执行过的测试或人工检查可以明确列出其范围。
