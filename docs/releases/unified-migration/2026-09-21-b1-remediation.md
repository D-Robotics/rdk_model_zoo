# B1 整改记录（B1-R1–R6，2026-09-21）

> 作者：原执行者（Claude Code）。本文是 [B1 独立评审](2026-09-21-b1-independent-review.md)
> 六项必需整改（B1-R1–R6）的逐项整改记录：位置、内容、回归证据。
> **独立 reviewer 复核通过前，B1 Review 仍为 changes-required、Closed=no。**
> X5 8GB 补测（评审关闭条件 2）已于同日完成并回填
> [board evidence](evidence/2026-09-21-b1-board-smoke-evidence.json)
> （x5-8g @192.168.3.208，5/5 pass）。

## 整改前基线

独立评审审阅 HEAD `c218e862`；整改基于其后的 `5ec7e1d`（x5-8g 板测回填）。
整改提交见文末"提交清单"。

## B1-R1 — ResNet50/152 转换能力迁入

- **位置**：`samples/vision/resnet/conversion/`（新增
  `get_calibration_data.py`、`resnet152_config.yaml`、`x86_inference.py`，
  逐字节来自 `rdk_s@380e1a2:samples/vision/resnet152/conversion/`）；
  `conversion/README.md` + `README_cn.md` 按变体（18/50/152）全重写；
  `tests/test_conversion_layout.py` 新增。
- **内容**：152 全配方（发布 ONNX wget + 用户自备校准图 + `hb_compile`，
  S600 改 march `nash-p`）；50 维持指针式（源分支无配方，OE `13_resnet50`
  为权威，不虚构）；18 导出-only 缺口如实声明。逐字节属性用 SHA-256
  固定（d8a39491…e6910 / 20eaf2cb…d742 / f2c5738e…13fa）；YAML 输出前缀
  与 Manifest 文件名（`resnet152_224x224_nv12.hbm`）一致性入测试。
- **回归证据**：resnet 套件 46 → **52 OK**（+6：provenance 3 + YAML/脚本
  一致性 + 前缀/清单一致 + 变体覆盖文档 3）；单样例 checker
  `--parser-mode import` 0 violations / 2 skips。
- **台账**：resnet 行 Refactor 由 `in-progress（R1 …）` 翻转为
  `done（R1 整改：转换配方逐字节迁入并 SHA 固定）`。

## B1-R2 — MobileNetV2 C++ launcher S100P 身份

- **位置**：`samples/vision/mobilenetv2/runtime/cpp/run.sh`（身份 gate
  重写）；`tests/test_cpp_launcher_identity.py` 新增；cpp 双语 README
  Supported boards 章同步。
- **内容**：launcher 现读取 `soc_name` **与** `board_type`
  （`/sys/class/boardinfo/board_type`），镜像 `_shared/platforms.py:
  match_target` 的登记语义：soc `s100` + board_type `s100p`/`rdk s100p`
  → S100P 拒绝；soc `s100p` 直接拒绝；soc `s600` 不受 board_type 细分
  （与 match_target 一致）；未知/不可读身份文件按未知板拒绝（原先
  `set -e`+pipefail 会在重定向失败处裸退 rc=1，一并修复）。gate 位于
  构建/模型加载之前。`SOC_NAME_FILE`/`BOARD_TYPE_FILE` 环境变量供主机
  fixture 注入身份来源。
- **回归证据**：`tests/test_cpp_launcher_identity.py` 7 项——S100/S600
  正例（gate 放行、停在显式模型准备提示，证明 gate 先于构建）、显式
  `s100p` 拒绝、`s100`+`s100p`/`rdk s100p`/`RDK S100P`（大小写/空格
  归一化）三形态拒绝、`s600`+s100p board_type 放行（语义负例）、双文件
  缺失报 unknown、未知 soc 报错。mobilenetv2 套件 17 → **24 OK**。
  21 格身份矩阵（soc×board_type）手工验证输出与预期全等。
- **板端复测**（2026-09-21 补做，随用户指令"仅重跑受行为修改影响的
  范围"；R2 是六项整改中唯一有板端可见行为变更的项）：launcher
  sha256 `01c0c6d4…`（主机与两板部署副本三方一致）。s100 正例
  （root@192.168.3.116，soc `S100`+空 board_type）：gate 放行、增量
  构建、TOP-1 zebra prob=9.30961（与整改前基线 9.309612274169922
  一致）、rc=0，双跑确定。s100p（root@192.168.3.191）4/4：真实身份
  `S100P` 显式拒绝 rc=2；登记别名 `s100`+`s100p`（即 B1-R2 finding
  形态）拒绝 rc=2；`s100`+`RDK S100P`（大小写/空格归一化）拒绝
  rc=2；对照例 `s100`+空 board_type gate 放行并停在显式模型准备
  提示（板端证明 gate 先于构建/模型加载）。日志与命令见
  [board evidence](evidence/2026-09-21-b1-board-smoke-evidence.json)
  的 `r2_launcher_board_recheck` 节。边界不变：s100p 无正向推理
  （无已发布制品）、s600 cpp 维持 not-run、x5 不受 R2 影响。

## B1-R3 — V3/V4 校准/编译文档按 target×variant 落实

- **位置**：`samples/vision/mobilenetv3/conversion/README(_cn).md`、
  `samples/vision/mobilenetv4/conversion/README(_cn).md`（校准、编译、
  已知缺口章重写）。
- **内容**：逐 config 声明 `cal_data_dir`、YAML 声明的布局（RGB/BGR、
  224/256、mean/scale）与脚本产出的关系：S 侧（v3 bgr / v4 small
  bgr224 / v4 medium bgr256 两行注释开关）改源目录即可产出；X5 侧
  `calibration_data_rgb_f32` **无产出配方**（校准器 BGR-only），作为
  缺失前提声明，明确"改名不是修复"。v4 X5 medium 的
  `mobilenetv4_conv_medium_deploy.onnx` 无保留脚本能产出、导出器 256
  vs 发布 224 的三重缺口逐条列出。脚本硬编码的旧树源目录、脚本与
  S YAML 的 mean 常量差异（103.94/116.78/123.68 vs 103.53/116.28/
  123.675，均源分支原样）如实披露。编译表逐 config 增加"配置引用的
  输入"列（onnx 匹配状态 + 校准目录可得性）。
- **回归证据**：静态路径/参数对照——README 所列每个 cal_data_dir 与
  onnx 名均与对应 YAML 字段逐一核对（grep 证据见整改工作记录）；两
  sample checker 0 violations；v3 套件 17 OK 不变。

## B1-R4 — V4 medium S 验收 shape 256

- **位置**：`mobilenetv4/conversion/README(_cn).md` 验证章（统一 224
  改为按 target×variant 表）；`tests/test_conversion_readme_shapes.py`
  新增。
- **内容**：S medium 验收形状更正为 Y `[1,256,256,1]`、UV
  `[1,128,128,2]`（small 仍 224/112；X5 两变体 224 packed），并明确
  "用 224 验收正确的 S medium 制品属于验收错误"。
- **回归证据**：新测试 3 项解析两语言验证表的 Y/UV 形状并与
  `model_binding.BINDING_TABLE.facts` 逐 (variant,target) 比对
  （medium s100/s600 = 256/128 回归锚点；X5 行假设双变体同几何并
  与 binding 校验）。mobilenetv4 套件 17 → **20 OK**。

## B1-R5 — 板后证据同步客户 README + 台账拆行

- **位置**：4 个 mobilenet 根 README（双语支持矩阵+尾段）、4 个
  evaluator README（双语 reference-results+边界尾注）、v2 cpp README
  （双语尾段）、resnet 根 README（双语变体行+尾段）共 20 文件 90 处
  断言式替换；`x5-s-migration-map.md` B1 区 5 行聚合行拆为 20 行
  target×variant×language 行 + 1 行 s100p 负例基础设施行。
- **内容**：矩阵行按实测翻转（x5 8GB+4GB、s100、s600 python →
  supported-verified 2026-09-21；cpp s100 verified、cpp s600 保持
  not-run 不扩大声明；s100p 行标注负例验证）。evaluator host 行更新
  为整改后计数（17/24/17/20）并保留 B1 收尾 17 的来源说明；板卡对照
  行 passed 并链接 board evidence；S600 复测尾注改为已完成。resnet
  s600 resnet18 行理由修正（板卡已恢复、不在 B1 冒烟集），50/152 翻
  verified。作者自评与独立评审保持分离（Review 列仍
  changes-required）。
- **回归证据**：scope 模式 checker 对拆行后台账 0 R-SCOPE 违规；
  7 samples / 84 violations（全部为 R6 基线对象）与拆行前一致，未引入
  新违规。

## B1-R6 — 84 条 YOLO 文档欠账的 CI 显式限界

- **位置**：`tools/sample_contract/check.py`（exemption 增加可选
  `message` 精确锚定；修复多 sample 运行时 exemption 被跨 report 重复
  记为 unused 的缺陷）、`tools/sample_contract/baselines/
  ultralytics-readme-debt.json`（新增，84 条逐 message 锚定）、
  `.github/workflows/sample-contract.yml`（Check migration scope 步
  加 `--exemptions`）、checker README、map B9 行（删除义务）。
- **行为验证**（真实仓库三态）：
  1. CI 同命令带基线：`7 samples, 0 violations, 10 skips, 84
     exemptions applied`，rc=0；
  2. 不带基线同命令：rc=1（门槛未降低）；
  3. 修复任一锚定问题（临时补上 `overview` 锚点）：1 条 unused
     exemption → rc=1（过期基线检测，基线只能收缩）；恢复后回到 1）。
- **负例**：checker 单测 24 → **27 OK**（+3：message 锚定只容忍精确
  finding、message 不匹配即 unused、多 sample 无幻影 unused——后者为
  基线暴露的原有缺陷的回归锚）。无 continue-on-error、无整目录豁免、
  无规则降级；B9 行与基线文件头均写明删除义务。

## 全量回归（整改后）

| 项 | 结果 |
| --- | --- |
| resnet / v1 / v2 / v3 / v4 / yolo / ocr 套件 | 52 / 17 / 24 / 17 / 20 / 59 / 44 OK |
| `_shared` 测试 | 71 OK |
| checker 单测 | 27 OK |
| CI 命令（--scope migration --parser-mode import --exemptions 基线） | 0 violations，84 exemptions applied，rc=0 |

## 提交清单

- `e2cb210` 独立评审文件（reviewer 交付，先行提交）
- `5ec7e1d` x5-8g 板测完成回填（board evidence / map / 评审文件）
- `e43718b` B1-R1：resnet 转换配方逐字节迁入 + provenance/layout 测试（52 OK）
- `c1bb46b` B1-R2：mobilenetv2 cpp 启动器 board_type 身份 gate + fixture 测试（24 OK）
- `7cfc186` B1-R3/R4：v3/v4 转换 README 按 target×variant + medium 256 shape 测试（v4 20 OK）
- `97dc639` B1-R5：16 份 mobilenet 客户 README 同步板测证据（resnet 根与 v2 cpp 随 R1/R2 提交）
- `ea68ea6` B1-R6：checker message 锚定豁免 + 84 条欠账基线 + workflow 旗标（checker 27 OK）
- `25a8187` 台账拆行 + 本整改记录；`160abb8` 整改提交自评；`81b0a92` board evidence 闭环注记
- `8f7fa0f` B1-R2 板端复测（关闭条件 2"仅重跑受影响范围"）：s100 正例复现基线 +
  s100p 4/4 拒绝，见 board evidence `r2_launcher_board_recheck` 节与本文件 B1-R2 节

## 整改提交自评（步骤 review，2026-09-21，HEAD `25a8187`）

逐提交 `git show --stat` 核对，文件分组与提交信息一致；六个提交均带
`Co-Authored-By: Claude Code` 署名行；提交后工作树 clean；在提交后树上
重跑 CI scope 命令（`--scope migration --parser-mode import
--exemptions baselines/ultralytics-readme-debt.json`）：
**7 samples / 0 violations / 10 skips / 84 exemptions applied，rc=0**。

两处跨组文件的落位说明（均已在对应提交信息或本节披露，无未声明内容）：

1. resnet 根 README 双语 2 件（R5 的矩阵/尾段改动）随 R1 提交 `e43718b`
   落库——该提交以 `git add samples/vision/resnet/` 整目录暂存；R5 提交
   信息已注明"resnet 根与 v2 cpp 随 R1/R2 提交"。v2 cpp README 双语
   （R2 身份章 + R5 尾段）同理随 `c1bb46b`。
2. 台账 `x5-s-migration-map.md` 同时含 R5 拆行与 R1 resnet 行翻转、R6
   的 B9 行删除义务注记，统一随台账提交 `25a8187` 落库，避免拆 hunk。

单元测试证据均取自与提交内容逐字节一致的工作树（提交仅暂存，未改动
内容）；全量电池见上表。

## 待独立 reviewer 复核项

1. B1-R1–R6 逐项按本文位置与回归证据复核（关闭条件 1）。
2. X5 8GB 补测证据（关闭条件 2，已完成，见 board evidence x5-8g 节）。
3. 双语 README/台账拆行/自评与独立评审分离（关闭条件 3）。
4. 复核通过后由独立 reviewer 记录新 HEAD 与关闭证据，B1 方可
   Closed=yes 进入 B2（关闭条件 4）。
