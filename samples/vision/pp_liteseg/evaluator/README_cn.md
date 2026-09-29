[English](README.md) | 简体中文

# PP-LiteSeg 验证

<a id="dataset"></a>
## 数据集

随附 street.png/test.jpg 仅支持单图检查。Cityscapes 是类别词表，仓库未内置验证集或数据集评估循环。计算 mIoU 时须取得适用的标注 split，记录标签 ID 映射／忽略规则，并在完全相同的样本上评估浮点和编译模型。

<a id="environment"></a>
## 环境

单图入口与 runtime 一致，需要 X5 OS 3.5.0+、Python 3.10+、NumPy/OpenCV/PyYAML 和板端 SDK。--help 可在主机使用。OE hb_perf 在转换容器执行，hrt_model_exec 在 X5 执行，两者不是同一测量环境。

<a id="command"></a>
## 命令

```bash
# cwd: repository root, on X5; model explicitly prepared
bash samples/vision/pp_liteseg/model/download.sh --target x5
python3 samples/vision/pp_liteseg/evaluator/infer_board.py --model samples/vision/pp_liteseg/model/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin --image samples/vision/pp_liteseg/test_data/street.png --output outputs/pp_liteseg/eval.png --alpha 0.55
```

兼容参数：--model、--image 必填，--output 默认 result.jpg，--alpha 默认 0.55。推理全部委托统一 runtime。成功返回 0，runtime 错误返回 2。保留源入口的图片／模型参数，不维护第二份推理实现。

<a id="metrics"></a>
## 指标

infer_board.py 不计算数据集分数或计时指标。其输出支持逐像素精确 ID 比较、类别检查和可视化复核。源文档的 logits 余弦 ≥0.95 仅适用于显式匹配的 logits 探针，不能用于类别图。mIoU 与延迟需要独立实测，记录数据集、预热、重复次数、核心配置和工具版本。

<a id="outputs"></a>
## 输出

以 --output outputs/pp_liteseg/eval.png 为例，输出该三面板图像及同目录的 eval.labels.npy、eval.report.json。mask 为 int32 512×1024、类别 0..18。报告保留运行时元数据／版本、制品身份／路径和实际类别名，不硬编码预期类别列表。已有输出文件会替换。

<a id="reference-results"></a>
## 参考记录

源 README 写有 1024×512 单核约 95 FPS、10.5 ms 的预期，但未提供可复现测量记录；此处保留为未验证的源预期，不作为验收阈值或新结果。主机源行为对照和 fixture 不证明模型精度。本轮没有板端、OE 或数据集结果。参见[源审计](../../../../docs/releases/unified-migration/evidence/2026-09-26-b8-ppliteseg-audit.json)。

<a id="boundaries"></a>
## 边界

板卡环境不可用，本轮板测为 not-run。本示例不支持 S100/S100P/S600 或 C++。本地编译模型须满足相同张量契约；声明 asset-id 和未知发布 SHA 不能证明来源。实际性能命令和图对照前提见[转换验证章节](../conversion/README_cn.md#validation)。
