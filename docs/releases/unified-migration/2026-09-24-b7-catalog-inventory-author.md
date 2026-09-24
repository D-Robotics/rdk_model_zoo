# B7 catalog inventory 过时断言修正（2026-09-24，作者报告）

基点：`07f9a09`（已包含 8 个恢复的 YOLOv5 X5 制品与 `9692699` 的 X5 汇总 177→185 修正）。本报告范围仅为 catalog inventory 回归修正，不是 B7 验收。develop 侧 `1cc3000`（YOLO11 URL 变更）与本任务无关，未合并。

## 现象与根因

协调者 `npm --prefix tools/catalog-publisher run check`（Node 22）：119 项中仅 `tests/multiplatform-catalog.test.ts:91` 失败——reviewed inventory 期望 595 个配置，实际 603。根因：恢复制品的提交 `5d91f38` 只在 X5 manifest 追加 8 个 asset 并由 `9692699` 修平 summary，但 reviewed baseline 断言（57 families / 595 variants / 820 benchmarks）没有随之更新。

## 三方对照证明 +8 恰为恢复的 8 个配置

用 `git archive` 导出三棵树（临时目录挂只读 alternates + refs 快照，供 benchmark source ref 校验读取 git 对象），以 `tools/catalog-publisher` 分别构建 catalog：

| 树 | families | variants | benchmarks | asset_count |
| --- | --- | --- | --- | --- |
| head = `07f9a09` | 57 | **603** | 820 | 577 |
| minus8 = head 回退 `5d91f38` 的 8 asset hunk + `9692699` 汇总 | 57 | **595** | 820 | 569 |
| develop = `01c8ddc`（未含恢复的 develop tip） | 57 | **595** | 820 | 569 |

- head 与 minus8 的变体集合差**恰好 8 行**，全部是 `yolov5` 家族 x5 的 asset-only 行（`yolov5-{s,m,l,x}v{20,70}-640x640-nv12-object-detection-x5`）；反方向差为空，其余 56 个家族逐一相同（仅 yolov5 36→44）。
- head 与 develop（含 YOLO26 激活/路由）的差同样恰为这 8 行，证明 YOLO26 路由与 S 侧脚本改名未改变任何配置数。
- families 57、benchmarks 820（全局与 yolov5 家族 98、变体级 799）三方一致，未变。
- 三棵 catalog 均无重复 variant key；4 个共享 asset URL（s100/s600 `yolov5x_672x672`、paddle_ocr det/rec）为恢复前已存在，head 与 minus8 完全一致；8 个新 URL 互不相同。

## YOLOv5 九变体真实 asset 映射

X5 manifest 的 `yolov5` 家族共 9 个 tag 制品，逐一映射到唯一 filename + 唯一 URL（`…/rdk_model_zoo/rdk_x5/<filename>`），sha256 均为声明的 `null`：

- `yolov5n_tag_v7.0`（恢复前已有）——benchmark `yolov5n-v7-640` 显式声明 `asset_filename`，asset 挂在实测行 `yolov5n-v7-640-object-detection-x5` 上；
- 恢复的 8 个（s/m/l/x × v2.0/v7.0）——对应实测行不声明 `asset_filename`，且 `canonicalTuple` 把文件名 `v2.0` 归一为 `lv20`、benchmark id `v2` 归一为 `lv2`，tuple 不兼容无法合并，故为 8 个独立 asset-only 行，与 8 条无 asset 的实测行并存（沿用 YOLO26 同款的"下载行/实测行"结构，非回归）。

## 测试修正（仅测试文件，未动生产代码/manifest/lock）

`tools/catalog-publisher/tests/multiplatform-catalog.test.ts`：

1. reviewed baseline 注释与断言更新为 57 families / **603** configurations / 820 benchmarks，注明 8 个 B7 恢复的 asset-only 变体；断言强度不变。
2. 新增回归 `keeps the nine YOLOv5 X5 tag artifacts as nine distinct downloadable configs`：钉死 9 个 tag filename 逐一出现且仅出现一次、URL 精确匹配、9 行一一对应（防重复/防丢失）、仅 n 行带 benchmark 而其余 8 行为 asset-only（防静默合并）、家族 44 变体 / 49 benchmark 记录（防 benchmark 漂移），并对 X5 manifest 本身做 185 条 asset 全唯一 + 与 summary 一致的守卫。

突变验证（改后恢复，工作树终态仅测试文件被修改）：

- 回退 8 asset（minus8 manifest）→ 新回归失败 `expected [ Array(1) ] to deeply equal [ …(9) ]`；
- 复制 `yolov5x_tag_v7.0` 条目并把 summary 抬到 186 → 新回归失败 `expected 185 to be 186`（`uniqueAssets` 会在 catalog 前折叠纯重复，靠 manifest 级守卫捕获）。

## 结果

`npm --prefix tools/catalog-publisher run check`（Node v22.23.2）全绿：**18 文件 / 120 测试通过**（原 118 通过 + 修正 1 + 新增 1），typecheck 通过，catalog 可复现校验通过（`catalog-v1.0.0-454bf082534bbbf8`，sha256 `a0c93928…`，与协调者 `2058537` 证据及本报告 head 独立构建一致）。证据：[evidence/2026-09-24-b7-catalog-inventory-author.json](evidence/2026-09-24-b7-catalog-inventory-author.json)。

## 边界

未 commit/push/merge/SSH，未改其它 worktree；未改生产代码、manifest、package-lock；未降级任何断言；板端不在本任务范围（host checks only）。等待 Codex 复核。
