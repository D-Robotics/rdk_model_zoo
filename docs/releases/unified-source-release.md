# 统一源码发布候选与支持/验证矩阵（2026-10-06 更新）

状态：**统一源码 2.0.0 候选，全部 51 个本仓 Sample 已完成重构并合入 Develop**。
实施验收提交为 `545a3b5874ae723663d2c817ae9ef964bc495746`，已合入 Develop；
整合时本地与实际远端一致，该提交的完整干净克隆及 Linux 3.10/3.12、
macOS 3.12 CI 已通过。
本页与验收摘要为其后的文档收尾；交付时仍须核对当前 Develop HEAD 的等价 CI，
不能用实施基线的 CI 代替后续提交的结果。实际计数、可选范围、失败修复链与 workflow
run IDs 见 [2026-10-06 交付验收](2026-10-06-develop-delivery-review.md) 和
[机器可读摘要](2026-10-06-develop-delivery-review.json)。

2026-10-05 的历史观察保留：存在 `develop`，默认分支为 `rdk_x5`，无 `main`。
2026-10-06 整合时仍未创建 main/tag/Release、未切换默认分支或发布网站。
源码版本、平台制品与 Skills 的独立生命周期见
[ADR-0006](../adr/0006-unified-source-releases-and-platform-matrix.md)。
板测、真实权重下载、导出/量化与工具链编译本轮仍为 **not-run**；按用户确认范围，
不构成本轮源码交付阻断，也不记为通过。第 2 节保留 2026-10-05 的逐 Sample
架构验收计数与历史板测条件；完整主机/CI 新证据由上面的交付记录补充。

## 1. 三条版本线（互不替代）

| 版本线 | 当前值 | 位置/形式 | 说明 |
| --- | --- | --- | --- |
| 统一源码 | **2.0.0（候选）** | 仓库根 `VERSION`；tag 方案 `zoo-vX.Y.Z`（首个为 `zoo-v2.0.0`，**尚未创建**） | 源码快照版本，代表整个统一仓库；不是平台制品版本，也不是 Skills 版本 |
| 平台制品 | X5 `1.1.3`（`x5-v1.1.3`）、S `1.1.2`（`s-v1.1.2`） | `docs/release/x5/VERSION`、`docs/release/s/VERSION` 及各自清单 | 平台清单按需推进；历史 `x5-v*`/`s-v*`/`x3-v*` tag 永不移动 |
| Skills 包 | Pack `1.1.0`（`unreleased-candidate`，成员各自有版本） | `skills/pack.json`、`skills/VERSION` | 按 ADR-0006 独立发版；源码发版不重标 Skills，也不改变其源码/分发 |

ADR-0006 遗留的“裸 `vX.Y.Z` 与统一源码版本协调”问题在此裁定：**统一源码 tag 一律
`zoo-vX.Y.Z`**；`x5-v*`/`s-v*`/`x3-v*` 前缀保留给历史平台制品线；Skills 包沿用其自身
独立的版本体系（Pack 与成员版本见 `skills/pack.json`），与 `zoo-v*` 互不占用。仓库根
`VERSION` 的引入使 `CHANGELOG.md` 中“暂不引入根 VERSION”的历史说明由本候选接替
（历史正文原样保留）。

## 2. 支持/验证矩阵（按真实声明数据构建）

矩阵的每一列是**独立维度**，不得互相顶替：

- **Languages / Artifact manifest**：实现语言与制品清单入口（X5=`docs/release/x5`，
  S=`docs/release/s`）。这是**支持声明**的仓库级事实；每个 target × variant × 语言的
  精确支持/实测三态（supported-verified / supported-not-run / not-supported）以**各
  Sample 自己的支持矩阵**为准（`samples/<domain>/<sample>/README.md`）。
- **Host acceptance (2026-10-05)**：可读 Runtime 重构的独立主机验收（[覆盖表](unified-migration/2026-10-05-all-sample-coverage.json)、
  [独立验收](unified-migration/2026-10-05-all-sample-codex-review.md)）：51 套共 1871
  项测试，1859 项执行通过、12 项显式跳过（Paraformer 可选依赖 Torch/FunASR/ONNX-ORT
  缺失，未改为通过）。`accepted_host` 只证明源码架构与主机行为，**不证明板端**。
- **Historical board evidence**：迁移台账（[x5-s-migration-map.md](unified-migration/x5-s-migration-map.md)）
  记录的历史板测事实，**保留原始范围**：scoped 通过仅覆盖括号内列出的板/变体/输入；
  partial 表示部分目标有证据、其余未测；not-run 保持 not-run。任何一例通过不外推为
  全家族/全板通过，也不因代码重构自动失效或自动生效。

汇总：51 个本仓 Sample（49 个 Python 入口、2 个原生 C++ LLM）。历史板测维度分布：
9 个 scoped 通过、2 个部分通过部分 not-run、7 个 partial、**33 个 not-run**。ACT/Pi0
为固定上游 gitlink，不在 51 个之内（[VLA 指南](../../samples/vla/README.md)），其板端
与实机控制未执行。

| Sample | 迁移批次 | Languages | 制品清单 | Host acceptance (2026-10-05) | 历史板测证据（passed/partial 的确切范围见括号；not-run 即未运行） |
| --- | --- | --- | --- | --- | --- |
| [3dresnet](../../samples/vision/3dresnet/README.md) | B5 | Python | S | accepted_host (20 tests) | not-run (kept as an open board dimension; no result is treated as passed) |
| [bytetrack](../../samples/vision/bytetrack/README.md) | B7 | Python | S | accepted_host (16 tests) | partial: S100 four synthetic frames and S100/S600 first 30 real video frames source-comparisons passed; S100P source URL 404; not full-video MOT accuracy |
| [clip](../../samples/vision/clip/README.md) | B5 | Python | X5 | accepted_host (19 tests) | not-run (kept as an open board dimension; no result is treated as passed) |
| [convnext](../../samples/vision/convnext/README.md) | B3 | Python | X5 | accepted_host (44 tests) | not-run (kept as an open board dimension; no result is treated as passed) |
| [depth_anything_v2](../../samples/vision/depth_anything_v2/README.md) | B8 | Python | S | accepted_host (22 tests) | not-run (kept as an open board dimension; no result is treated as passed) |
| [diffusiondrive](../../samples/vision/diffusiondrive/README.md) | B8 | Python | S | accepted_host (29 tests) | not-run (kept as an open board dimension; no result is treated as passed) |
| [dinov2](../../samples/vision/dinov2/README.md) | B5 | Python | S | accepted_host (22 tests) | not-run (kept as an open board dimension; no result is treated as passed) |
| [edgenext](../../samples/vision/edgenext/README.md) | B3 | Python | X5 | accepted_host (42 tests) | not-run (kept as an open board dimension; no result is treated as passed) |
| [efficient_sam](../../samples/vision/efficient_sam/README.md) | B6 | Python | X5+S | accepted_host (23 tests) | partial: X5 8GB default/priority7 and S100 default full-source comparisons passed; X5 4GB logs-only, S600 unfinished, S100P unverified |
| [efficientformer](../../samples/vision/efficientformer/README.md) | B2 | Python | X5 | accepted_host (41 tests) | passed (scoped): B2 x5 dual-board l1/l3 |
| [efficientformerv2](../../samples/vision/efficientformerv2/README.md) | B2 | Python | X5 | accepted_host (42 tests) | passed (scoped): B2 x5 dual-board s0/s1/s2 |
| [efficientnet](../../samples/vision/efficientnet/README.md) | B2 | Python | X5+S | accepted_host (44 tests) | passed (scoped): B2 x5 dual-board b2/b3/b4 + S100/S600 lite0–lite4; S100P explicit-rejection negatives |
| [efficientvit](../../samples/vision/efficientvit/README.md) | B2 | Python | X5 | accepted_host (43 tests) | passed (scoped): B2 x5 dual-board m5 |
| [fasternet](../../samples/vision/fasternet/README.md) | B3 | Python | X5 | accepted_host (44 tests) | not-run (kept as an open board dimension; no result is treated as passed) |
| [fastvit](../../samples/vision/fastvit/README.md) | B3 | Python | X5 | accepted_host (44 tests) | not-run (kept as an open board dimension; no result is treated as passed) |
| [fcos](../../samples/vision/fcos/README.md) | B7 | Python | X5 | accepted_host (42 tests) | partial: x5 8GB/4GB efficientnetb0/b2/b3 full-source comparisons passed |
| [googlenet](../../samples/vision/googlenet/README.md) | B4 | Python | X5 | accepted_host (26 tests) | not-run (kept as an open board dimension; no result is treated as passed) |
| [hgnetv2](../../samples/vision/hgnetv2/README.md) | B4 | Python | X5 | accepted_host (32 tests) | not-run (kept as an open board dimension; no result is treated as passed) |
| [lanenet](../../samples/vision/lanenet/README.md) | B8 | Python + C++ | S | accepted_host (28 tests) | not-run (kept as an open board dimension; no result is treated as passed) |
| [lprnet](../../samples/vision/lprnet/README.md) | B7 | Python | X5 | accepted_host (27 tests) | partial: x5 8GB/4GB lpr.bin full-source comparisons passed |
| [mobile_sam](../../samples/vision/mobile_sam/README.md) | B6 | Python | X5+S | accepted_host (21 tests) | partial: X5 8GB default/priority7 and S100 default full-source comparisons passed; X5 4GB logs-only, S600 unfinished, S100P unverified |
| [mobilenetv1](../../samples/vision/mobilenetv1/README.md) | B1 | Python | X5+S | accepted_host (33 tests) | passed (scoped): B1 x5 8GB + S100 + S600 published variants |
| [mobilenetv2](../../samples/vision/mobilenetv2/README.md) | B1 | Python + C++ | X5+S | accepted_host (40 tests) | passed (scoped): B1 x5 8GB + S100 + S600 python and S100 cpp; S600 cpp not-run |
| [mobilenetv3](../../samples/vision/mobilenetv3/README.md) | B1 | Python | X5+S | accepted_host (33 tests) | passed (scoped): B1 x5 8GB + S100 + S600 published variants |
| [mobilenetv4](../../samples/vision/mobilenetv4/README.md) | B1 | Python | X5+S | accepted_host (36 tests) | passed (scoped): B1 x5 8GB + S100 + S600 small/medium |
| [mobileone](../../samples/vision/mobileone/README.md) | B4 | Python | X5 | accepted_host (26 tests) | not-run (kept as an open board dimension; no result is treated as passed) |
| [modnet](../../samples/vision/modnet/README.md) | B7 | Python | X5 | accepted_host (17 tests) | not-run (kept as an open board dimension; no result is treated as passed) |
| [paddle_ocr](../../samples/vision/paddle_ocr/README.md) | pilot | Python + C++ | X5+S | accepted_host (46 tests) | passed (scoped): pilot x5 8GB/4GB + S100 det+rec; S100P/S600 not-run |
| [pointnet](../../samples/vision/pointnet/README.md) | B8 | Python | S | accepted_host (33 tests) | not-run (kept as an open board dimension; no result is treated as passed) |
| [pp_liteseg](../../samples/vision/pp_liteseg/README.md) | B8 | Python | X5 | accepted_host (22 tests) | not-run (kept as an open board dimension; no result is treated as passed) |
| [repghost](../../samples/vision/repghost/README.md) | B4 | Python | X5 | accepted_host (24 tests) | not-run (kept as an open board dimension; no result is treated as passed) |
| [repvgg](../../samples/vision/repvgg/README.md) | B4 | Python | X5 | accepted_host (26 tests) | not-run (kept as an open board dimension; no result is treated as passed) |
| [repvit](../../samples/vision/repvit/README.md) | B4 | Python | X5 | accepted_host (26 tests) | not-run (kept as an open board dimension; no result is treated as passed) |
| [resnet](../../samples/vision/resnet/README.md) | B1+pilot | Python + C++ | X5+S | accepted_host (72 tests) | passed (scoped): pilot x5 8GB/4GB + S100 resnet18; B1 S100+S600 resnet50/152; S100P has no approved asset |
| [resnext](../../samples/vision/resnext/README.md) | B4 | Python | X5 | accepted_host (26 tests) | not-run (kept as an open board dimension; no result is treated as passed) |
| [siglip](../../samples/vision/siglip/README.md) | B5 | Python | S | accepted_host (23 tests) | not-run (kept as an open board dimension; no result is treated as passed) |
| [ultralytics_yolo](../../samples/vision/ultralytics_yolo/README.md) | B9+pilot | Python + C++ | X5+S | accepted_host (173 tests) | passed (scoped pilot: x5 dual + S100 + S100P yolov8n/yolo26n detect; S600 not-run); rest of the unified family not-run |
| [unet](../../samples/vision/unet/README.md) | B8 | Python | X5 | accepted_host (24 tests) | not-run (kept as an open board dimension; no result is treated as passed) |
| [unetmobilenet](../../samples/vision/unetmobilenet/README.md) | B8 | Python + C++ | S | accepted_host (23 tests) | not-run (kept as an open board dimension; no result is treated as passed) |
| [vargconvnet](../../samples/vision/vargconvnet/README.md) | B4 | Python | X5 | accepted_host (26 tests) | not-run (kept as an open board dimension; no result is treated as passed) |
| [vit](../../samples/vision/vit/README.md) | B5 | Python | S | accepted_host (29 tests) | not-run (kept as an open board dimension; no result is treated as passed) |
| [yolo26_depth](../../samples/vision/yolo26_depth/README.md) | B8 | Python + C++ | X5+S | accepted_host (47 tests) | not-run (kept as an open board dimension; no result is treated as passed) |
| [yoloe](../../samples/vision/yoloe/README.md) | B9 | Python + C++ | X5+S | accepted_host (48 tests) | not-run (kept as an open board dimension; no result is treated as passed) |
| [yolov5](../../samples/vision/yolov5/README.md) | B7 | Python + C++ | X5+S | accepted_host (83 tests) | partial: x5 nine variants × 8GB/4GB and S100/S600 x-672 python source comparisons, S100P rejection negatives, four-board native smoke passed; C++ source-value comparison unfinished |
| [yoloworld](../../samples/vision/yoloworld/README.md) | B7 | Python | X5 | accepted_host (23 tests) | partial: x5 8GB/4GB dog-prompt/image full-source comparisons passed; not full-vocabulary accuracy |
| [asr](../../samples/speech/asr/README.md) | B10 | Python + C++ | S | accepted_host (35 tests) | not-run (kept as an open board dimension; no result is treated as passed) |
| [kws](../../samples/speech/kws/README.md) | B10 | Python | S | accepted_host (16 tests) | not-run (kept as an open board dimension; no result is treated as passed) |
| [paraformer](../../samples/speech/paraformer/README.md) | B10 | Python + C++ | S | accepted_host (96 tests, 12 skipped) | not-run (kept as an open board dimension; no result is treated as passed) |
| [himloco](../../samples/robotics/himloco/README.md) | B10 | Python + C++ | X5 | accepted_host (33 tests) | not-run (kept as an open board dimension; no result is treated as passed) |
| [gemma4-e2b](../../samples/llm/gemma4-e2b/README.md) | B11 | C++ (native) | S | accepted_host (41 tests) | not-run (kept as an open board dimension; no result is treated as passed) |
| [minicpm5-2b](../../samples/llm/minicpm5-2b/README.md) | B11 | C++ (native) + legacy | S | accepted_host (20 tests) | not-run (kept as an open board dimension; no result is treated as passed) |

**显式保留的资产/配置缺口（不因本候选改判）**：MODNet X5 制品为 manual（未找到可靠
来源，不以模拟代替）；Paraformer S100 需本地 `am.mvn` 与 `paraformer_config.yaml`
（随仓提交，非下载制品）；S100P 仅限已发布组合，无资产时不回退 S100；SAM 双样例的
X5 4GB（仅日志）、S600（未完成）、S100P（未验证）缺口见
[板端交接记录](unified-migration/2026-09-24-board-resume.md)；ByteTrack S100P 源 URL
404；Gemma/MiniCPM 的厂商 ABI、真实模型与量化精度验收未关闭；X3 为历史材料，仅经
固定提交访问，绝非新适配目标。SHA-256 未知值保持 `sha256: null (unknown)`。

## 3. 提升到 main 的门禁（等价 CI，不重写运行时代码）

提升流程**不要求修改任何 Sample 运行时/测试代码**；只要求以下门禁在**同一个源码提交**
上全部执行通过：

1. **维护者全量主机验证**：`python tools/host_validation/run.py --repo PATH --report PATH [--python PYTHON]`
   （由 2026-10-05 host-validation 工作提供；覆盖原生 Sample Python 套件、嵌套
   conversion/evaluator 套件、共享模块（含父仓 VLA gitlink 完整性守卫；ACT/Pi0 上游
   不初始化、不运行）、受影响工具/Skills 测试、静态契约、
   适用原生 CTest 与 Catalog）。实施提交的实际 CI 见
   [交付验收](2026-10-06-develop-delivery-review.md)；本页后续文档提交仍须在其实际
   Develop HEAD 单独确认等价 CI，不能沿用实施提交的结果。
2. **静态契约**：`python3 tools/sample_contract/check.py --scope migration`（本轮基线：
   51 samples，0 violations，87 个显式 policy skips，0 exemptions）。
3. **Catalog**：`npm --prefix tools/catalog-publisher run check`（本轮 136 项测试、
   schema 校验、可重复构建）。
4. **CI 工作流等价**：`develop`、`main` 与 PR 触发同一组门禁（host-validation、
   sample-contract、model-catalog-data）。以**实际运行观察**为准，不以 YAML 阅读代替。
5. Python 3.10/3.12 支持范围与依赖以 host-validation 工作声明的为准；缺失依赖必须
   显式失败或显式声明为可选范围，不得伪通过。

## 4. main 提升、发布与回退流程

前置事实（2026-10-05 观察，执行时须重新核对）：远端存在 `develop`；默认分支为
`rdk_x5`；**无 `main`**。以下步骤为可执行流程，本文**不执行**任何一步：

1. **冻结候选提交 C**：develop 上的提交，第 3 节全部门禁在 C 上通过；本地与远端
   develop 一致，无未推送差异。
2. **创建 main**：`git branch main <C> && git push origin main`。这是对外动作，由
   用户/授权执行者操作；**不切换默认分支**（`rdk_x5` 保持默认，切换默认分支是用户
   之后的独立决策）。
3. **观察等价 CI**：C 上的 main push 触发与 develop 相同的门禁并全部通过；未观察
   到的 CI 不算通过。
4. **打 tag 并发布**：在 C 上创建附注 tag `zoo-v2.0.0` 并推送；GitHub Release 以
   `CHANGELOG.md` 候选段为说明基础，附件为在 C 上构建的 Catalog 数据包
   （`tools/catalog-publisher/dist/`）。发布后 **tag 永不移动**。
5. **Catalog 不可变来源**：`sources.json` 工作树来源配置 `link_ref: "HEAD"`，构建时
   解析为 C 的完整提交 SHA；产物链接为 `blob/<C>/...`，绝无 `/tree/HEAD` 或分支名。
   历史来源（X3 固定提交、`--pin` 历史 tag）保持原引用。

**回退**：

- **代码回退**：在 develop/main 上追加 revert 提交，不重写已发布历史；已发布 tag 下
  的缺陷以补丁版本（如 `zoo-v2.0.1`）修复，不移动 tag。
- **Catalog 回退**：以回退后的提交重建数据包；`catalog.meta.json` 的 SHA-256 契约使
  任何陈旧产物可被检出。
- **main 撤回**：仅当没有任何 tag/Release 指向其独有提交时才可删除分支；否则向前
  修复。全程不动默认分支，`rdk_x5` 上的客户入口不受 main 回退影响。

## 5. 本候选不声明的内容

未发布任何 tag/Release/Hub/默认分支变更；板测维度按第 2 节保留（33 个 not-run 与
各 scoped/partial 范围原样）；既有 Benchmark 数值、资产 URL 与 SHA-256 不改写；历史
证据保留原始条件；本候选记录的是**源码发布准备状态**，不是客户迁移发布完成的声明。
