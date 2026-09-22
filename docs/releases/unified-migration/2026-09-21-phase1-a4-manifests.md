# Phase 1 — A4 Manifest 搬迁与资产权威统一（2026-09-21）

来源：`rdk_x5 @ ac11571` 与 `rdk_s @ 380e1a2` 的 `docs/manifests/`
（分支 tip，非 develop 的 `platforms/{x5,s}/docs/release/` 冻结快照——快照
落后 tip：x5 快照 v1.1.2 vs tip v1.1.3；s 快照与 tip 同标 v1.1.2，但 tip
含 20 个 cls 文件名修正，快照没有）。按计划 A4 单 commit 完成搬迁 +
`samples/_shared/assets.py` 改路径 + 测试与 `sources.json` 同步更新。

## 搬迁结果

| 来源（分支 tip） | 去向（develop） | 变化 |
| --- | --- | --- |
| `rdk_x5:docs/manifests/models.yaml`（v1.1.3） | `docs/release/x5/models.yaml` | 试点行路径修正（下表）+ summary 重算 |
| `rdk_x5:docs/manifests/benchmarks.yaml` | `docs/release/x5/benchmarks.yaml` | 逐字节原样（239 条记录不动） |
| `rdk_x5:docs/manifests/schemas/*.json` | `docs/release/x5/schemas/` 与 `docs/release/s/schemas/` | 原样；两侧使用同一（x5 超集）版本 |
| `rdk_s:docs/manifests/models.yaml` | `docs/release/s/models.yaml` | 试点行修正 + 新增 3 行 + summary 重算 |
| `rdk_s:docs/manifests/benchmarks.yaml` | `docs/release/s/benchmarks.yaml` | 逐字节原样（538 条记录不动） |

两侧 tip 的 `docs/manifests/README.md` 与 x5 的 `catalog-sources.json`
**不搬迁**：前者描述 rdk_x5 网页目录管线（ADR-0001 排除在 develop 之外）
与 s-v1.1.1 历史修正说明（已折叠进 yaml 本身），后者是 rdk_x5
`docs/catalog/` 的输入。schema 取 x5 版：diff 证实其为 s 版超集
（新增 `role`/`display_name`/`asset_filenames`/`repository_url` 与 3 个
unit，全部可选），两侧清单均通过校验。

`platforms/{x5,s}/docs/release/` 冻结快照**原样保留**（收尾阶段统一
删除）；本 commit 后 `samples/_shared/assets.py` 不再读取它们。

## 试点行路径修正（保留旧 ID 与资产事实，仅改 develop 上的定位）

| 清单行 | 字段 | 旧值（分支布局） | 新值（develop 统一布局） |
| --- | --- | --- | --- |
| x5 `paddleocr` | sample_path | `samples/vision/paddleocr` | `samples/vision/paddle_ocr` |
| x5 `paddleocr` | download_scripts | `.../paddleocr/model/download_model.sh` | `[]` + notes（经 `main.py --target x5 --prepare` 获取，统一 sample 无下载脚本） |
| x5 `ultralytics_yolo26` | sample_path | `samples/vision/ultralytics_yolo26` | `samples/vision/ultralytics_yolo` |
| x5 `ultralytics_yolo26` | download_scripts | `.../ultralytics_yolo26/model/{download_model,fulldownload}.sh` | 统一 sample 同名脚本（两个都存在） |
| s `paddle_ocr` | download_scripts | `.../paddle_ocr/model/download_model.sh` | `[]` + notes（同上，`--target auto --prepare`） |
| s `resnet18` | sample_path / scripts | `samples/vision/resnet18` + `download_model.sh` | `samples/vision/resnet` + `model/download.sh`（variant 经 `--variant resnet18` 选择） |
| s `ultralytics_yolo26` | sample_path / scripts | `samples/vision/ultralytics_yolo26` + 其 `download_model.sh` | `samples/vision/ultralytics_yolo` + 其 `download_model.sh`（notes 追加搬迁说明，保留原 YOLO26n 资产口径） |

x5 `resnet`、x5 `ultralytics_yolo`、s `ultralytics_yolo`、s `paddle_ocr` 的
sample_path 在 tip 清单中已是统一路径，无需修改。

## S 侧补漏：3 个源分支未登记的 sample（计划 A4"逐项核对并补漏"）

逐行证据来自 rdk_s tip 的下载脚本与 README（见 evidence JSON）：

1. **`yoloe26_seg`**（+10 资产）：n/s/m/l/x × nash-e（S100）/ nash-m（S100P），
   `yoloe_26{v}_seg_pf_{nashe,nashm}_640x640_nv12.hbm`。sha256 全部 null——
   脚本对远端 `manifest.json` 逐文件校验，仓库不记录摘要（不伪造）；
   `.json`/`.names` 伴随文件沿用本清单"视觉 sample 只登记模型制品"的既有
   约定不登记（与 ultralytics 各行一致）。notes 声明来源 SHA 与待 B9 迁移。
2. **`yoloe11_seg`**（+1 资产）：S100-only（脚本对 S600 显式报错退出）；
   单文件 `yoloe_11s_seg_pf_nashe_640x640_nv12.hbm`，脚本直接 wget 到
   sample `model/` 根目录（无目标子目录），故文件名无前缀。conversion
   文档中出现的 `yoloe_v8s_seg_pf_nashe_640x640_nv12.hbm` 无已记录下载
   URL，不登记（notes 说明）。
3. **`minicpm5-2b`**（+3 资产，sha256 齐全）：s100/s100p/s600 三个
   `minicpm5-2b_{board}_oellm{1,2}_w8_ctx4096_2026xxxx.tar.gz`，摘要是脚本
   内固定并校验的值（可信来源）；S600 需内部 OELLM 2.0 beta SDK（README
   声明，notes 保留）。

三个行的 `sample_path`/`download_scripts` 暂用分支布局（与清单中其余
未迁移行一致），B9/B11 批次迁移时统一改指 develop 路径。台账补充由 A7
落账。

## summary 重算（与 `release-summary.ts` 同口径）

| 计数 | x5 | s |
| --- | --- | --- |
| sample_count | 37（不变） | 35 → **38** |
| download_script_count | 40 → **39**（paddleocr 脚本移除） | 33 → **35**（−1 paddle_ocr，+3 新行） |
| asset_count | 177（不变） | 368 → **382**（+10/+1/+3） |
| downloadable / manual | 176 / 1（不变） | 366→**380** / 2（不变） |
| sha256 有/无 | 24 / 153（不变） | 2→**5** / 366→**377** |
| benchmark / perf / acc | 239 / 636 / 419（不变） | 538 / 1367 / 2716（不变） |

## 同 commit 配套改动

- `samples/_shared/assets.py`：`_models()` 与 `Asset.source_path` 从
  `platforms/{group}/docs/release/models.yaml` 改为
  `docs/release/{group}/models.yaml`（唯一权威路径）。
- `samples/_shared/tests/test_assets.py`：新增
  `test_manifest_authority_is_the_unified_release_location`（钉住新位置与
  `source_path` 报告值）；其余测试路径无关，原样通过。
- `tools/catalog-publisher/sources.json`：x5/s 改为 `path: "."` +
  `manifest_root: docs/release/{x5,s}` + `link_ref: develop` +
  `link_prefix: ""`；x3 不动（仍在 `platforms/x3`，收尾再改）。

## 连带修复：ultralytics_yolo 分类分辨率规则（源分支事实驱动）

s tip 清单把 20 个 cls 文件名键从 224x224 修正为 640x640（nashe/nashm 的
yolo11/yolov8；nashp 即 S600 保持 224x224），rdk_s tip 的
`download_model.sh` 同样按 640x640 构造文件名与 URL——可执行事实一致。
develop 的 `yolo_assets.py` 原规则（"s 全平台 cls=224x224"）随清单切换而
失效（`test_syntax_and_public_downloads` 实测暴露）。按新事实改为：

- YOLO26 cls：全部目标 224x224（x5/s 一致，未变）；
- S600（nash-p）：全部 cls 224x224（未变）；
- 其余家族在 X5/S100/S100P：640x640（修正点）。

同步更新：`CLASSIFICATION_RESOLUTION` 平台映射改为按家族/目标的常量集合；
`classification_resolution(profile, family)` 签名增加 family（无外部消费者，
已核实）；`run.sh` 注释示例、双语 README 的"已发布分类文件名"行、
`test_execution_target.py` 的合成清单路径（`docs/release/x5/models.yaml`）。

**发现的源分支不一致（已披露，不静默修复）**：s tip 清单这 20 行的
filename 键是 640x640，但 URL 仍是 224x224 字符串，与其自身 tip 下载脚本
（640x640 URL）矛盾——两者必有一处与归档服务器不符。本 commit 按计划
"保留资产事实"原样保留 tip 字节，不替上游猜测归档真名；待 B9 收编或
板端实际下载时以服务器事实裁定后修补。已同步记入 A7 台账补充范围。

## 验证（2026-09-21，主机）

- jsonschema（draft 2020-12）：4 份清单全部通过。备注：PyYAML(1.1) 会把
  s 侧未加引号的 `released_at: 2026-09-08` 解析为 date 触发类型报错，
  管线实际使用的 JS YAML 1.2 解析器（`yaml` npm）保持 string——已用
  node 实测确认，s 文件保持 tip 原始字节。
- benchmarks.yaml 含重复 YAML 锚点名（tip 原样）；JS 解析器容忍，
  Python 校验侧做了"别名指向最近定义"的等价重写后计数。
- summary 计数逐项与内容重算一致（脚本核对，见 evidence）。
- 主机测试全绿：resnet 39、paddle_ocr 43、ultralytics_yolo 59、
  `_shared` 18（含新增 1）、sample_contract 23。
- 试点入口：paddle_ocr `--list-models` 双平台 4 个引用可解析；
  resnet `--list-models` 报告 `source_manifest: docs/release/x5/models.yaml`。
- 迁移范围检查器：3 samples / 84 violations / 6 skips——与 A4 前一致
  （全部为 ultralytics_yolo R-README-SECTIONS 的既定过渡态，未新增）。
- `git status`：`platforms/` 下零改动（冻结快照原样）。

## catalog-publisher 状态（如实记录，A5 收口）

- 基线（A4 前）：`npm run check` 的 vitest 已有 1 个失败
  （`catalog release workflow is the only workflow` 期望只有
  model-catalog-data.yml——Q3 引入 sample-contract.yml 后未同步，属遗留）。
- A4 后：vitest 42 failed（测试夹具锁定旧 `platforms/{x5,s}` 布局与旧
  总数，sources.json 翻转后失效）；`validate:sources` 对 x5/s 报
  benchmark 证据路径 `SOURCE_NOT_FOUND`（引用按平台根解析，未迁移 sample
  在仓库根不存在；x3 正常）。清单本体（schema/发布标识/唯一 ID/引用/
  归一化）在新区位置全部通过。
- A5 工作清单（由此明确）：① 证据校验改为按 record 的不可变 `ref`
  解析（`git cat-file`）而非工作区路径；② vitest 布局/总数夹具更新；
③ 工作流触发路径补 `docs/release/**`；④ workflow-name 断言纳入
  sample-contract.yml；⑤ 全量 `npm run check` 转绿。
- 未运行：板端冒烟（用户门禁）、网络拉取（含 s cls URL 真伪裁定）。

## 结论

A4 交付物完整：manifests 于 `docs/release/{x5,s}/` 落位，资产权威单一路径，
试点行对齐统一布局，S 侧 3 个漏登 sample 补录且摘要诚实（知则记、不知
为 null），连带暴露并修复了 develop 侧唯一的名称漂移（yolo cls 分辨率）。
遗留项均已有明确归属（A5 管线收口、A7 台账补遗、B9/板测裁定 URL 真名）。
