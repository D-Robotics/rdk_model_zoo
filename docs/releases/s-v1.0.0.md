# RDK Model Zoo S v1.0.0

## English

This is the first formal S-series baseline release from the `rdk_s` branch.
It freezes source ref `s-v1.0.0` on 2026-09-04 and records the repository
inventory in `release/models.yaml` and `release/benchmarks.yaml`.

Tagged manifests: [models.yaml](https://github.com/D-Robotics/rdk_model_zoo/blob/s-v1.0.0/release/models.yaml) and [benchmarks.yaml](https://github.com/D-Robotics/rdk_model_zoo/blob/s-v1.0.0/release/benchmarks.yaml).

The release contains 33 releasable sample entries and 308 model or supporting
assets: 306 downloadable references and 2 files provisioned locally by a
committed helper. Unknown SHA-256 values are written as `null` and are not
treated as verified checksums. ACT and Pi0 remain external gitlink samples and
are omitted from these totals. No source files were changed to represent these
external samples.

The benchmark manifest includes numeric performance and accuracy evidence that
already appears in repository documentation. It preserves the original test
conditions where they are stated and leaves unpublished or non-numeric values
unreported. This release has no repository-wide RDK board test gate and makes
no board validation claim.

The GitHub Release attaches both `models.yaml` and `benchmarks.yaml` so that
the exact release inventory can be downloaded alongside these notes.

## 简体中文

这是 `rdk_s` 分支的首个正式 S 系列基线版本。版本固定于 2026-09-04，源引用为
`s-v1.0.0`，模型清单与 benchmark 清单分别位于 `release/models.yaml` 和
`release/benchmarks.yaml`。

带标签的清单链接：[models.yaml](https://github.com/D-Robotics/rdk_model_zoo/blob/s-v1.0.0/release/models.yaml) 和 [benchmarks.yaml](https://github.com/D-Robotics/rdk_model_zoo/blob/s-v1.0.0/release/benchmarks.yaml)。

本版本包含 33 个可发布 sample 和 308 个模型或配套资产，其中 306 个为可下载引用，
2 个由仓库内提交的脚本在本地提供。未知 SHA-256 值统一写为 `null`，不表示已完成校验。
ACT 和 Pi0 是外部 gitlink sample，不计入上述总数；为表示这些外部 sample，没有修改源代码。

benchmark 清单只收录仓库文档中已经公开的数值化性能与精度证据，并保留原文已经说明的测试条件；
未公开或无法数值化的内容不作推断。本版本不设置全仓库 RDK 板卡测试门禁，也不作板卡验证声明。

GitHub Release 会同时附加 `models.yaml` 与 `benchmarks.yaml`，便于下载与本版本对应的完整清单。
