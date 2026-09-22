# Phase 1 — A7 台账补遗：快照↔tip 漂移勘定（2026-09-21）

A7 的任务是保证"每能力有去向"：勘定 `platforms/{x5,s}` 冻结快照相对各自提取源
tip（x5 `ac11571` / s `380e1a2`）的漂移，把快照缺失的能力补入迁移台账，并把
A4/A5 细化后的 cls 文件名旗标落到台账。

## ① 漂移勘定（逐文件 comm 比对）

**x5：samples/ 零漂移。** tip 相对快照多出的 307 个文件全部是 docs/catalog（201，
ADR-0001 排除）、skills/**（A3 已从同一 tip 落位）、docs/manifests 与 docs/tros
（发布设施，A4/A6 已处理）、workflows（2，排除）、AGENTS.md。快照多出的 17 个文件
为旧位置清单（docs/release→docs/manifests 迁移残留）、历史 release notes、
`__pycache__`、`.gitattributes`、根 `tros/`——均已被 tip 取代。**x5 侧无台账补遗项。**

**s：53 个 tip 独有文件，能力级去向：**

| tip 独有内容 | 规模 | 去向 |
| --- | --- | --- |
| `samples/vision/yoloe26_seg/` | 30 文件 | B9 yoloe 行（表内已注明 "s tip yoloe26_seg"） |
| `samples/llm/minicpm5-2b/` | 13 文件（evaluator/legacy 脚本 + results/{s100,s100p}-{generation,wikitext2}-full.json + test_data） | B11（本步把该行细化为 "cpp + evaluator（legacy 脚本 + results/*.json）+ test_data" 并标注"快照无，tip 新增"） |
| `samples/vla/{act,pi0}` + `.gitmodules` | 2 gitlink + 1 配置 | B11（gitlink SHA 326ea043/a32de276 与计划一致；develop 现行 `.gitmodules` 仍指 platforms/s 快照路径，B11 改指 `samples/vla/`） |
| `docs/manifests/*` | 5 文件 | 清单在 tip 已由 docs/release 迁至 docs/manifests；develop 侧 A4/A5 已按 tip 内容落位 `docs/release/s` |
| `docs/tros/README.md`（空）、`skills/README.md`（占位） | 2 文件 | A6 裁定不迁（tros 取 x5 双语版；skills/README.md 取 x5 实文） |

**清单补充核算**：develop `docs/release/s/models.yaml` 资产 382 = tip 368 + 14
（yoloe26_seg 10 + yoloe11_seg 1 + minicpm5 tar.gz 3），与 A5 的 57 families/
595 variants 逐项核验一致；benchmarks.yaml 与 tip 字节相同。快照多出的 17 个文件
（旧清单、release notes、pycache、`.gitattributes`、
`tests/test_yolo_cls_resolution.py`、根 tros）均为 tip 取代/废弃内容，不构成迁移义务。

## ② cls 文件名旗标细化（落为台账未核定项 6）

新证据链（本步发现）：快照 `tests/test_yolo_cls_resolution.py` 断言 cls 制品名
`*_cls_<march>_224x224_nv12.hbm` 且 `model_url` 返回 224 URL；**s tip 删除了该测试**，
同时 sample 代码与下载脚本改为构造 640x640 文件名与 URL，清单 filename 键 640、
URL 仍 224。结论：tip 处于 224→640 改名中途（文件名已改、URL 未跟随）；develop errata
按 URL 证据展示 224。三方自洽、互不改写；真名待网络实测（用户门禁），裁定前 B9
不得合并或改写任何一侧。已写入 `x5-s-migration-map.md` 未核定项 6 + 勘误脚注。

## ③ 台账编辑清单

`docs/releases/unified-migration/x5-s-migration-map.md`：
- 未核定项新增 6（cls 旗标全证据链）、7（快照↔tip 漂移勘定与去向表）；
- B11 minicpm5-2b 行细化（含 results/test_data，标注 tip 新增）；
- 勘误脚注补：yoloe26_seg/minicpm5/vla 仅 tip 存在、B9 先裁定 cls 旗标。

## 验证

- 本步纯台账（无代码/管线改动）；漂移数据来自 `git ls-tree -r` vs 快照 `find` 的
  逐文件 comm 比对，清单数字来自 blob 哈希与行数核对（382=368+14 复算吻合）。
- gitlink SHA 与 `.gitmodules` URL 已对照 tip 核实；与计划 B11 的 pinned SHA 一致。
- 未运行：网络实测（cls 真名裁定，用户门禁）、板端。

## 结论

快照↔tip 漂移全部勘定且有去向：x5 无补遗项，s 的四类新增全部落在既有 B9/B11 行
（其中 minicpm5 行按能力细化）；cls 旗标以完整证据链固化进台账。**Phase 1（A1–A7）
至此完成**。下一步：Q5 Skills 补强（Phase 0.5 收尾），随后 Phase 1.5 H1–H6。
