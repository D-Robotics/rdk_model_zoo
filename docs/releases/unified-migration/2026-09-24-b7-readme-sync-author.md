# B7 README 实测状态同步作者报告（fcos / lprnet / yoloworld / modnet）

本报告是 [B7 native SDK 独立核对](2026-09-24-b7-native-sdk-review.md) 所记录板端证据的**作者侧 README 同步**交付，只改四个样例的客户双语 README 与本报告，不改任何运行代码。

**状态口径（不得被本报告改写）：**

- B7 独立评审的 **changes-required 结论原样保留**；**Closed = no**。本报告不声明独立验收，不改台账既有独立评审结论。
- 本任务**未执行任何新的板端运行**；README 中的板端事实全部转写自本分支已合入的证据（develop `4753431` 同步，见下）。
- 全程**未 commit / 未 push / 未 merge / 未 SSH**，未进入 B8，未触碰 yolov5/bytetrack 文件（另一 worker 负责）。
- 转换（export/校准/编译）在四个样例中**仍然 not-run**；历史 benchmark 数字保持"仅历史"标注，未冒充复测。

基点：分支 `codex/b7-glm-readme-20260924` @ `07f9a09`（"merge: synchronize develop evidence into B7 review integration"，含 develop `4753431` 证据），独立 worktree，起点工作树 clean。证据来源：`evidence/2026-09-24-b7-binding-recheck/`（FCOS B0 8GB、LPRNet 8GB）、`evidence/2026-09-24-b7-other-x5-variants/`（FCOS B2/B3 8GB，FCOS B0/B2/B3、LPRNet、YOLOWorld 4GB）、`evidence/2026-09-24-b7-python-comparison/`（YOLOWorld 8GB）。

## 1. 变更范围（26 个文件，全部为 README 与本报告）

对每个样例逐层审阅 root / model / runtime/python / evaluator / conversion 的 `README.md` + `README_cn.md`；实际修改按事实差异最小化，未新增/删除章节锚点，未改任何命令、参数表、目录结构或代码。

### FCOS（8 个文件改 6 个）

- 根 README（双语）：support-matrix 三行 `supported-not-run` → `supported-verified`；删除"no X5 board was used"旧说明，替换为 2026-09-24 B0/B2/B3 各在一块 X5 8GB 与一块 X5 4GB 的同板 source/unified 对照（`bus.jpg`、direct resize、`conf=0.5`、`IoU=0.6`，rc=0、全部检查 true）+ 证据链接 + 覆盖边界（非 COCO 精度/时延测量）；prerequisites 的 "board validation is not-run" 改为两块板实测口径；expected-results 补充"两块板上统一输出与固定源完全一致"并保留演示图为历史图片的表述。
- model README（双语）：`formats-checksums` 拆为 Publisher SHA-256（保持 `null (unknown)`，来源 manifest）与 Observed SHA-256（B0 `fd184f35…`、B2 `adf6436d…`、B3 `7044d6c9…`，均转写自 comparison 记录），并明确 observed 不认证来源。
- evaluator README（双语）：`reference-results` 新增板端一致性行（条件含统一侧板测提交 `73a6de1`）；补充板端加载制品时 HBRT 库/模型构建小版本不一致警告原样保留且不影响对照；`boundaries` 从"板端证据 not-run"改为"已有三变体两板证据，其他板/图/阈值仍需自行运行"。
- conversion README（双语）：validation 中"conversion and board smoke are not-run"改为精确口径——转换 not-run（无配方）；已发布制品的板端执行由 evaluator 对照覆盖，但不覆盖 `hrt_model_exec` 检查与转换可复现性；known-gaps 同步。
- runtime README 未改（无过时声明；stage-io 已含 2026-09-24 dict 顺序证据说明）。

### LPRNet（10 个文件改 8 个）

- 根 README（双语）：support-matrix 由 "board binding remediated … board re-run pending" 改为 `supported-verified` + 两板对照事实与证据链接；明确验证范围是 native `(1,68,18,1)` logits 单输入一致性，不是车牌精度。
- model README（双语）：preparation 的 "This migration did not execute the command" 改为"2026-09-24 两板运行执行的就是本文档模块命令（均 rc=0）"；`formats-checksums` 记录两板观测 SHA-256 `f4803915…`（发布者仍 null）。
- runtime README（双语）：environment 中 `(1,68,18,1)` "实测板端协议"补充"经 X5 8GB/4GB 对照确认"；3D 布局"仅旧 host/API 兼容、未观察到已发布 SDK 制品"的原有边界**原样保留，未编造历史 SDK**。
- evaluator README（双语）：`reference-results` 由"未运行板端对照 / not-run / closed=no"改为两板对照事实（rc=0、全部检查 true、输入/logits/plate `max_abs_diff` 均为 0.0、统一侧提交 `73a6de1`、版本警告原样保留）；`boundaries` 同步为"已记录两板，其它板/输入仍需各自运行，不外推车牌精度"。
- conversion README（双语）：validation 拆分口径——转换与 `hrt_model_exec` 检查 not-run，已发布制品板端一致性见 evaluator README；known-gaps 的 "conversion and board validation are not-run" 相应改为仅转换 not-run。

### YOLOWorld（10 个文件改 6 个）

- 根 README（双语）：support-matrix 由 "board not-run" 改为 2026-09-24 两板 `dog` 提示 + `test_data/dog.jpeg` 对照通过 + 证据链接，并加一句覆盖边界（张量一致性，非全词汇精度/时延）；prerequisites 修正事实错误——词向量 JSON 随 sample 提供，不需"单独准备"（仅模型需显式下载）。
- model README（双语）：`formats-checksums` 记录两板观测 SHA-256 `bc8fd742…`（发布者仍 null，不认证来源）。
- evaluator README（双语）：reference-results 表新增 2026-09-24 板端一致性行；正文从"板端执行 not-run 直到本命令生成证据"改为"已记录证据只覆盖 `dog` 提示/图片组合（统一侧提交 `ae0f185` 与 `73a6de1`），版本警告原样保留；其它提示/图片/板卡需自行运行，不外推全词汇精度"。
- runtime / conversion README 未改（无过时声明；conversion 保持"无法由本树复现"口径）。

### MODNet（2 个文件改 2 个）

- 状态不变：manual 制品无 URL、无下载器（`download` 仅打印要求返回 2）、板端 not-run。唯一修改是在根 README（双语）support-matrix 状态中明确"2026-09-24 X5 板端批次没有拿到该手工制品，因此没有可记录的下载或推理"，使跨样例对比时 not-run 原因可见；其余层级 README 经逐层复核无过时或自相矛盾表述，未改。

## 2. 一致性核对

- **证据事实转写**：所有新增数字（rc=0、checks 全 true、`max_abs_diff` 0.0、三个 FCOS / LPRNet / YOLOWorld 的观测 SHA-256、提交 `73a6de1` / `ae0f185`）均逐一从 `evidence/2026-09-24-b7-binding-recheck/`、`evidence/2026-09-24-b7-other-x5-variants/`、`evidence/2026-09-24-b7-python-comparison/` 的 comparison/execution JSON 复核（脚本验证所有被引用记录 `checks` 全 true、max_abs_diff 全 0）。发布者 SHA-256 与观测 digest 在全部表格与正文中分列，未混淆。
- **历史 benchmark**：FCOS 323.0/70.9/38.7 FPS 与 9/16/20 ms、LPRNet 266 FPS/3.75 ms、MODNet 89.88/130.49 ms 行保留原"仅历史/未复测"标注；YOLOWorld 保持"源无 benchmark 表"。
- **中英一致**：26 处修改按 README.md / README_cn.md 成对编写，逐段 diff 复核事实集合一致（矩阵三态、条件、数字、链接、边界句均对应）。
- **链接可解析**：对 24 份样例 README 的全部相对链接（含新增证据目录链接与 `#reference-results` 片段目标）做存在性校验，broken = 0；证据目录链接与既有迁移文档（native-sdk-review）用法一致。
- **命令与 parser**：未修改任何命令行；checker 的 `R-CLI-DEFAULTS`（import 模式）继续通过，即文档参数表与 `build_parser()` 实际默认值一致。

## 3. 验证（CI 同款命令）

| 步骤 | 命令 | 结果 |
| --- | --- | --- |
| 四样例 checker（CI parser-mode import） | `python tools/sample_contract/check.py --sample samples/vision/{fcos,lprnet,modnet,yoloworld} --parser-mode import` | 每样例 `0 violations`（各 1 个 policy skip：main.py CLI 层，既有行为） |
| 全量 migration checker（CI 同参数） | `python tools/sample_contract/check.py --scope migration --parser-mode import --exemptions tools/sample_contract/baselines/ultralytics-readme-debt.json --report /tmp/…` | `36 samples, 0 violations, 39 skips, 84 exemptions applied`（与基线一致，未新增豁免、未放宽规则） |
| 相对链接解析 | 自写脚本遍历 24 份 README 的相对链接/片段 | broken = 0 |

环境说明：本 worktree 无 `.venv`，import 模式使用主 checkout 的仓库 `.venv`（Python 3.14.7 / NumPy 2.5.3 / OpenCV 4.14.0 / PyYAML 6.0.3，与 yoloworld README 所述主机 fixture 版本一致）执行；CI 本身用 Python 3.11 + pip 安装同组依赖，语义等价。未下载任何模型、未加载 SDK、未接触板卡。

## 4. 未验证 / 未做（明确边界）

1. **本任务没有新的板端运行**：所有 verified 状态均引用已合入证据；README 中的统一侧身份是板测提交 `73a6de1`（FCOS B0 8GB/LPRNet 两板/4GB 各样例）与 `ae0f185`（YOLOWorld 8GB），与当前分支 HEAD 不逐一相同，已在条件列写明，未声称"本 checkout 原样复测"。
2. **观测 digest 未在本地重算**：模型二进制不入库，观测 SHA-256 转写自证据 JSON 的 `model_sha256` / downloader stdout，属证据记录而非本地复算。
3. **MODNet 板端仍 not-run**：手工制品不可得，只改状态说明，未虚构任何下载或运行。
4. **转换边界未动**：四个样例 export/calibration/compile 缺口照旧（无配方不伪装）；FCOS/LPRNet 的 conversion README 仅把"板端 not-run"拆分为精确口径，未声称转换已执行。
5. **C++ 覆盖**：四个样例均无 C++ runtime，所有 README 继续声明"无 C++/not provided"，未变化。
6. **父级索引**：`samples/README.md`（及其双语）尚无 B7 样例条目，因此没有需要同步的过时表述；B7 批次索引何时进入父级 README 属集成决策，本任务未代做。
7. **manifest 未改**：观测 digest 只写入 README，未写入 `docs/release/x5/models.yaml`（发布者 hash 仍为 null，manifest 规则不变）。
8. yolov5 / bytetrack README 与 native 工具不在本任务范围，未触碰。
