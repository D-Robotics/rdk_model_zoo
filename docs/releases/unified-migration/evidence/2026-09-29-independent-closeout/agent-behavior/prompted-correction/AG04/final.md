## 更正后的回答

**原错误承认**：我此前声称“S 平台没有 companion benchmarks.yaml”是**事实错误**。`docs/release/s/benchmarks.yaml` 实际存在（约 667KB，`schema_version: 1`，`release.platform: s`，`version/tag: s-v1.1.2`，branch `rdk_s`，发布于 2026-09-08）。

**EfficientNet 清单与性能核对**（仅只读，已逐项核对）：

- `docs/release/s/benchmarks.yaml:308-396` 含 5 条 EfficientNet 记录（`efficientnet-lite0`..`lite4`），asset 为 `s100/efficientnet_lite*_…_nv12.hbm`，runtime `hrt_model_exec`，`environment.hardware: "unspecified in source documentation"`。
- 数值与目标仓库源文档一致：`samples/vision/efficientnet/evaluator/README.md:105-114` 的 "S source release (rdk_s @380e1a2, s-v1.1.2; conditions unstated)" 表中 Lite0=0.448 ms / 2107.815 fps（单线程）… Lite4=0.915 ms / 1064.339 fps，与 yaml 中 concurrency 1/2 的记录完全对应。条件未声明的 qualifier 双方均保留，未跨平台代填。

**S100P 结论不变，理由更正**：

- 该清单内 S100P 条目仅限明确标注 `hardware: RDK S100P` 的样本（diffusiondrive、dinov2、efficient-sam、mobile-sam）；**EfficientNet 无 S100P 条目**。
- `samples/vision/efficientnet/README.md:60` 明确：`s100p | any | python | not-supported (no s100p asset row in the release manifest; selection is an explicit error, no fallback)`；`README.md:131-132` 进一步说明 S100P 是显式"no-published-asset error"。
- 因此拒绝 S100P 回退仍然正确，但现在依据是清单与 sample 文档的直接证据，而非之前基于“清单不存在”的错误前提。

**边界声明**：本次仅使用 Read/Glob/Grep 做只读检查，未执行任何 shell、git、板卡、模型或量化操作；未运行任何命令。S100P 支持范围仅按 EfficientNet 自身的 sample/资产证据判断，未从清单总兼容范围外推。
