# Phase 1 — A5 管线与工作流收口（2026-09-21）

A4 把 manifests 搬到 `docs/release/{x5,s}/` 后，catalog-publisher 处于
"清单本体通过、管线红" 的过渡态（vitest 42 failed / validate:sources x5+s
SOURCE_NOT_FOUND）。本步按 A4 记录的工作清单收口：证据按 record 不可变
`ref` 解析、per-platform VERSION 落位、vitest 夹具对齐统一布局、工作流触发
路径补齐、全量 `npm run check` 转绿。

## ① 证据校验改为按不可变 ref 解析（核心变更）

`validateRepositorySources`（`src/pipeline/manifest-validation.ts`）不再从
工作区读 benchmark 证据文件，改由 `git show <ref>:<path>` 从仓库对象库读取
record 自带的三元组 `(ref, path, section)`：

- **为什么工作区读取在统一布局下是错的**：341/777 条 x5+s 记录引用的
  sample 尚未迁移（仓库根不存在该路径）；已迁移 sample（resnet、
  ultralytics_yolo 等）的 README 在 Q4/B 批次被合法重写，标题已移动。ref
  才是 record 声明的出处，工作区副本不是。
- 777 条 x5+s 与 20 条 x3 记录的 `ref:path` 全部在本地对象库可解析（逐条
  `git cat-file -e` 核实，0 失败）；x5 侧全部 239 条为 40 位 commit SHA。
- **tag 源不再跳过**：原来 `kind !== "worktree"` 直接 return（理由是"逐文件
  读取需要检出树"）；对象库读取使该理由消失，pinned-tag 构建同样校验。
- **非文本证据只查存在**：`acc.jpg` 类截图按 caption 定位、不做标题匹配，
  原实现整读文件；新实现用 `git cat-file -e` 只验存在（顺带修掉实测踩到的
  707KB 二进制 blob 触发 execFile `maxBuffer` 抛错被误报为
  SOURCE_NOT_FOUND 的问题）。
- **读缓存**：`readRepositoryBlob` 进程内按 `(root, ref, path)` 缓存，多次
  引用同一 README 的记录只读一次；全量 validate:sources 1.4s。
- CI 前提不变：`model-catalog-data.yml` 已是 `fetch-depth: 0`（全分支+tag
  可达，record 引用的 rdk_x5/rdk_s 历史 commit 均在同仓库分支上）。

## ② per-platform VERSION（第二个结构性变更）

`manifestPair` 校验 manifest `release.version` 与平台 VERSION 一致。统一
布局下 x5（1.1.3）与 s（1.1.2）不可能共用一个仓库根 `VERSION`——这正是
A4 后 10 个 x5-evidence 测试 ENOENT 的根因。处理：

- 新增 `docs/release/x5/VERSION`（=ac11571:VERSION，1.1.3）与
  `docs/release/s/VERSION`（=380e1a2:VERSION，1.1.2）——版本文件随发布
  身位移入清单目录；x3 默认值不动（`platforms/x3/VERSION`）。
- `sources.json` 每源新增可选 `version_file`（默认 `"VERSION"`，相对平台
  根）；`SourceEntry`/`PlatformSource`/三种解析分支同步。
- A6 注意：届时从 rdk_x5 带回的**仓库根** `VERSION`/`CHANGELOG` 是 ADR-0006
  统一版本线，与本处的 per-platform 发布版本是两个概念，管线已不读根
  VERSION，两者并存不冲突。

## ③ vitest 夹具与断言对齐（42→0 failed）

| 套件 | 变更 |
| --- | --- |
| x5-evidence（10） | 全部因根 VERSION ENOENT 失败，②修复后直接通过 |
| sources（2→1 失败→0） | 默认解析断言改为统一布局：worktreeRoot/linkRef/linkPrefix/manifestDirectory 分平台期望，新增 versionFile 断言 |
| multiplatform-catalog（11→4→0） | platformTags x5→v1.1.3；provenance path x5/s→"."；variant 链接戳按平台分叉（x5/s: develop+空前缀，x3: main+platforms/x3）；总量基线 54/584/820→**57/595/820** |
| benchmark-coverage（5→1→0） | EXPECTED_MODEL_PATHS 对齐试点行（paddleocr→`samples/vision/paddle_ocr`；yolo26 并入 `samples/vision/ultralytics_yolo`，目录数 37→36）；记录数/指标数不变（239/636/419） |
| platform-registry（1） | registry 行更新后断言按平台分叉（见④） |
| release-workflow（1，基线遗留） | "唯一工作流"断言接纳 Q3 的 sample-contract.yml，防网站部署断言扩展到两个工作流 |
| artifact / family-merge / gemma / model-naming / s-classification（16） | 均为 VERSION/布局连带失败，①②修复后通过，未改动 |

**新总量基线核算**（对照旧基线逐项解释）：families 54→57 = +yoloe26_seg
（10 资产→10 变体）、+yoloe11_seg（1→1）、+minicpm5-2b（3 个 tar.gz 挂
卡片、0 变体）；variants 584→595（+11）；benchmarks 820 不变。资产
569/可下载 567。无未解释漂移。

**fixtures 重定向到真实历史**：`valid` 夹具 section 改为 `## Performance
Data`、非文本证据用 tag 上真实存在的 `test_data/cheetah.JPEG`、
`non-heading-section` 用真实 README + `### QuickStart`（层级不匹配证明
精确匹配）。合成样例树 `tests/fixtures/samples/` 删除；prose/链接/行内
代码/围栏不构成标题的负例改由新导出的 `hasExactMarkdownAtxHeading`
直接单测覆盖（内容原样内联）。

## ④ platforms/registry.json 过渡期对齐

x5/s 行仍描述冻结快照（v1.1.2、`platforms/x5/docs/release`），与
sources.json/VERSION 校验冲突。更新为当前事实：path `.`、
manifest_directory `docs/release/{x5,s}`、x5 release_tag/version →
x5-v1.1.3/1.1.3；readme/guidelines/license/release_notes 继续指向冻结树
（文件仍在，未迁移文档的诚实去处）；描述改为说明迁移期双轨。最终
`docs/release/registry.json` 收敛仍归收尾阶段。

## ⑤ 工作流

- `model-catalog-data.yml` 触发路径补 `docs/release/**`（x3 归档树
  `platforms/**` 保留）；未复制 rdk_x5 的 Pages workflow（ADR-0001）。
- rdk_x5 tip 仅有 `model-catalog-ci.yml`（docs/catalog 网站侧 CI）与
  `model-catalog-pages.yml`（部署）；"必要数据校验"在 develop 已由
  model-catalog-data.yml 的 `npm run check`（validate:sources + vitest +
  build + checksum 契约）完整承担，**无额外可移植项**，结论记录于此。

## 验证（2026-09-21，主机）

- `npm run check` 全绿：validate:sources 三平台 ok（x5 37/239、s 38/563†、
  x3 15/20；†563=538 清单记录+25 条 s-v1.1.2 errata 合成记录，归一化后
  口径）；vitest 17 文件全过；tsc --noEmit 过；构建产物可复现且 checksum
  契约通过（`catalog-v1.0.0-1678b0f7cd6d23c2`，57 families / 820
  benchmarks）。
- cls 归一化行为复核：catalog 侧 s-v1.1.2 errata 仍把 nashe/nashm cls 展示
  名归一为 224x224（"Archive migration … Legacy 640 URLs remain
  compatible"），与 A4 记录的 s tip 矛盾（tip 清单 filename 键 640、URL
  224、tip 下载脚本构造 640 文件名+640 URL）三方并存——catalog 展示、
  清单字节、sample 按 manifest 键解析各自自洽，互不改写；真名仍待
  B9/板端网络事实裁定（A7 台账旗标保留并细化）。
- Python 侧回归：`samples/_shared/tests`、`tools/sample_contract/tests`
  全 OK（registry.json 变更无 Python 消费者，已 grep 证实）。
- `git status`：改动限于上表文件 + 两个新 VERSION；`platforms/{x5,s}` 冻结
  快照本体零改动（registry.json 是唯一例外，属发布声明而非快照内容）。

## 结论

A4 遗留的管线债清零：证据校验语义升级为"按 record 不可变 ref 解析"（比
旧工作区读取更严格也更真实），VERSION 随发布身份落位，夹具/断言/注册表
全部对齐统一布局，CI 门禁全绿且产物可复现。A5 工作清单①–⑤全部完成；
rdk_x5 工作流移植评估结论为"无可移植项"。后续：A6 根/文档设施（注意根
VERSION 语义区分）、A7 台账补遗（携带细化后的 cls 旗标）。
