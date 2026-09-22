# Phase 1 — A6 根/文档设施落位（2026-09-21）

A6 把两条交付分支 tip 的仓库级文档设施迁入 develop：根 `CHANGELOG.md` / `LICENSE`、
`docs/tros/`、`docs/source_reference/`、两份 API 指引，以及本规范
（`docs/Model_Zoo_Repository_Guidelines.md`）的章节合并。来源固定：
`rdk_x5` @ `ac11571`（x5-v1.1.3）、`rdk_s` @ `380e1a2`（s-v1.1.2）。

## ① 逐项落位与来源裁定

| 交付物 | 来源 | 裁定依据 |
| --- | --- | --- |
| `CHANGELOG.md`（根，756 行） | 双侧合并 | 前言声明来源与合并原则；两段 body **逐字节保留**（仅去掉各自重复的 `# Changelog` 标题行）。锚点 `x5-v1.1.x-details` / `s-v1.1.x-details` 平台前缀天然唯一，合并后全部内链仍可解析（`grep -o 'id="…"' \| sort \| uniq -d` 为空） |
| `LICENSE`（根） | `rdk_s` @ 380e1a2 | 双侧均 Apache-2.0，唯一差异是行尾换行（s 带）；取 s 字节 |
| `docs/tros/{README.md,README_cn.md}` | `rdk_x5` @ ac11571 | s 的 `tros/README.md` 为 0 字节空文件；x5 双语均有内容 |
| `docs/source_reference/`（10 文件 + tarball） | 10 文件取 x5；`bpu_sample_docs_html.tar.xz`（5,976,660 B）仅 s 有 | 9 文件双侧 blob 相同；`sphinx/make.bat` 仅行尾差异（x5=LF、s=CRLF），按"10 文件取 x5 + tarball 取 s"规则统一；tarball 被双侧字节相同的 `source_reference/README.md` 引用 |
| `docs/Python_API_User_Guide.md`、`docs/UCP_User_Guide.md` | `rdk_s` @ 380e1a2 | s 独有（hbm_runtime Python 包 / libdnn+libucp 指引） |
| `docs/Model_Zoo_Repository_Guidelines.md` | 双侧章节合并 | 见② |

全部提取文件经 `git hash-object` 与 tip blob 逐一比对，**字节一致**（18/18 OK）。

## ② 规范合并（本步核心）

develop 的 Q1 基线入口文件（67 行）按其自设的 A6 任务
（"将其编码、注释与任务规范章节并入本文件并按 develop 目录结构改写目录章节"）
扩展为完整规范：

- **保留 develop 层级骨架**：定位与规范层级表（AGENTS / readme-contract /
  inference-contract / 迁移记录 / ADR / skills）不变——README 内容与阶段职责的
  **单一权威仍是两份契约**，并入的章节只提供结构基线，两处冲突以契约为准
  （文中以"迁移窗口说明"显式标注，不静默弱化任何规则）。
- **编码规范 / 各类任务规范**：整体框架（Python Config/Model 两类、C/C++ 配置
  结构体/模型类/推理函数、run.sh）双侧字节相同，原样并入；**任务规范取 s 版**
  ——x5 的分类/分割/姿态是 TODO 占位，s 填全了 分类/分割/实例分割/姿态/OCR/ASR
  的 C++ 结构体与 Python 输出契约（C++ 定义指向 `utils/c_utils/inc/model_types.hpp`，
  develop 上 A1 已带入，七类结构体逐一经 grep 核实在位）。
- **robotics 类别规范**：取 x5（s 删去了 robotics 章节），标注 himloco 随 B10 迁入。
- **目录章节**：按 develop 实际树重写（A1–A5 已落位的 utils/、datasets/、skills/、
  `docs/release/{x5,s}` 清单与 VERSION、本步新增文档；platforms/ 标注冻结快照去留）。
- **文档规范**：保留各文件的结构要求与职责划分，yolov5 参照改为
  "交付分支历史参照 + develop 迁移参照 resnet"；`docs/README.md` 条目删除
  （develop 无此文件，未造）；`docs/tros` 条目从 TODO 补为实际描述。
- **注释规范**：双侧相同，原样并入；修正源文档链接
  `docs/source_reference/README.md` → `source_reference/README.md`
  （原相对路径在两侧 tip 上即已断链，属顺带修正，记录在案）。
- **跨平台规范**：C/C++ 宏方案原样；Python 节**改写**——develop 的规则是
  `samples/_shared/platforms.py` 按板卡事实解析、未知即报错、无默认回退；
  交付分支 `get_soc_name()` 失败回退 `"s100"` 的行为显式标注"仅存在于交付分支，
  不得引入 develop"。

## ③ 明示不做与理由

- **不落根 `VERSION`**（对计划 A6 行 "VERSION" 的显式偏离）：x5=1.1.3、s=1.1.2
  在仓库根互斥；per-platform 版本已随清单落位 `docs/release/{x5,s}/VERSION`（A5）。
  仓库根版本线属 ADR-0006 统一发布线，自首个统一 release 起才有值——现在落一个
  只能是伪造。管线已不读根 VERSION（A5 核实），无兼容缺口。CHANGELOG 前言已说明。
- **不迁 `docs/assets/`、`docs/README.md`、`docs/RELEASE.md`**：`docs/assets/` 仅
  含 model_zoo_logo.jpg，develop 活树零引用（引用全部位于 platforms/ 冻结快照内，
  快照自洽）；`docs/README.md` 两侧均为导航页且 develop 无对应文件集；
  `docs/RELEASE.md` 属 rdk_x5 文档站发布流程（ADR-0001 排除）。留待收尾统一处理。
- **不补 `source_reference/build_docs.sh`**：`source_reference/README.md` 把它写成
  "推荐入口"（L84/L126），但该脚本在 rdk_x5、rdk_s 两个 tip 上均不存在——上游缺口
  而非搬运遗漏；实际构建入口是 `sphinx/Makefile`。不伪造脚本，如实记录。

## 验证（2026-09-21，主机）

- 18 个提取文件 `git hash-object` 对 tip blob 全部字节一致。
- 规范内相对链接逐一声存在性核验（AGENTS/契约/ADR/台账/skills/tros/source_reference 全 OK）；
  标题无重复；`utils/c_utils/inc/model_types.hpp` 七类结构体在位。
- 回归：`samples/_shared/tests` 18 OK；`tools/sample_contract/tests` 23 OK；
  skills 三件套（sync_references 无 drift、validate_pack valid、skills/tests 50 OK）。
- CHANGELOG 合并后锚点唯一、内链全可解析；新文件均不被 `.gitignore` 命中。
- catalog-publisher 未重跑：本步不改其任何输入（sources.json 只读
  `docs/release/{x5,s}`、`platforms/` 与 VERSION 文件；grep 证实管线无 CHANGELOG/
  LICENSE 消费者），A5 的绿色结论不受影响。
- 未运行：板端冒烟（用户门禁）、GitHub Actions（推送后触发）。

## 结论

A6 交付物全部落位且来源可溯（逐文件 blob 比对），规范合并以 develop 契约为单一
权威、交付分支章节为结构基线，无静默弱化；三处显式不做（根 VERSION、快照内引用的
资产/导航页、build_docs.sh）均有独立依据并记录。Phase 1 余：A7 台账补遗（携带细化
后的 cls 旗标与 s tip 新增 sample 清单）。
