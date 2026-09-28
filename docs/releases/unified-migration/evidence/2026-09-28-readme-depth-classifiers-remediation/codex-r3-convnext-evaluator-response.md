# R3 整改对照（DOC-CLASS-R3：convnext evaluator 与根 README 的 atto 矛盾）

整改时间：2026-09-28。范围仅 2 个文件：
`samples/vision/convnext/evaluator/README.md`、`README_cn.md`。
命令块（含 `--asset-id x5:convnext:ConvNeXt_atto_224x224_nv12.bin` 两处）
未改动。

## 核查

- `git show ac11571:samples/vision/convnext/README.md` 的 "Performance
  Data" 表含 atto 行：`ConvNeXt_atto | 224x224 | 1000 | 3.69 | 73.25% |
  69.75% | 1.96 | 732+`。
- `platforms/x5/docs/release/benchmarks.yaml`（归档不可变快照）条目
  `convnext-atto-x5`：latency 1.96 ms（single-frame, single-thread,
  single-BPU-core）、throughput 732 fps（4-thread concurrent）、top-1
  73.25（float）/ 69.75（quantized），源 ref `1e1c64d`。与 Codex 核对
  结果一致。
- 源 `evaluator/README.md`（ac11571）本身只有 Evaluation Types 与共享
  评测脚本说明，无性能表；当前 evaluator 的历史表是迁移时从源根 README
  拷贝的，"not atto" 声明是迁移动作引入的错误。

## 整改（中英同步）

改前（EN）：

> The published table covers nano/pico/femto — **not atto**, the only
> variant with a published artifact; no benchmark row exists for the
> downloadable model. Recorded as published, not re-measured or
> explained here.
>
> The quantized Top-1 (72.50%) stays close to the float value (73.75%)
> in the published record; this is recorded as published, not
> re-measured here.

改前（CN）：

> 已发布表覆盖 nano/pico/femto——**不含 atto**（唯一有已发布制品的
> 变体）；可下载模型没有基准行。按发布原样记录，此处不复测。

改后（两语同步）：

1. 表中恢复 atto 行（73.25% / 69.75% / 1.96 / 732+），与根 README 及
   源表完全一致。
2. 说明改为：四行（含 atto）同源于 rdk_x5 @ac11571 "Performance Data"，
   归档 `platforms/x5/docs/release/benchmarks.yaml` 条目
   `convnext-atto-x5` 记录相同 atto 数值；按发布原样记录，此处不复测。
3. EN 顺带修正陈旧句：原 "quantized Top-1 (72.50%) ... float (73.75%)"
   中 72.50 不在源表任何行中（femto 量化值为 72.25）。改为逐变体表述
   （femto 72.25% vs 73.75%、atto 69.75% vs 73.25%），中文版补齐对应句
   （原中文版无此句，属双语缺失）。
4. "not re-measured / not-run" 范围声明与全部命令块原样保留。

## 验证

- `tools/sample_contract/check.py --sample samples/vision/convnext`：
  0 violations / 1 policy skip（既定 CLI 豁免类）/ 0 exemptions。
- convnext 测试套件：28 OK（README 命令解析用例覆盖 evaluator README，
  命令块未受影响）。
- diff：仅 evaluator 两文件、+20/−9，删除行全部为错误声明与陈旧句；
  见 [checks-after-r3.log](checks-after-r3.log) 与刷新后的
  [readme-depth-classifiers.patch](readme-depth-classifiers.patch)。
- 作者报告已追加 R3 小节。不自行关闭 finding，待 Codex 复核。
