# MODNet 评估器

<a id="dataset"></a>
## 数据集

源没有抠图 benchmark 数据集或 ground-truth alpha mask。`../test_data/person.jpg` 和 `../test_data/bg.jpg` 是一组推理/合成 fixture，不是精度数据集。

<a id="environment"></a>
## 环境

`compare.py` 可在装有 Python、NumPy 和 OpenCV 的主机运行，不加载 `hbm_runtime`。生成源/统一 matte 文件需要两侧使用相同外部模型和 X5 runtime。下表和条件完整保留源历史数据，本轮未复测。

| 模型 | 尺寸 | 输入格式 | 延迟 (ms) | FPS |
|---|---|---|---:|---:|
| MODNet | 512x512 | Float32 NCHW RGB | 89.88 | 11.12 |
| MODNet（2 threads） | 512x512 | Float32 NCHW RGB | 130.49 | 15.27 |

条件：RDK X5、CPU 8xA55@1.8G、BPU 1xBayes-e@1G（10TOPS INT8）；单线程延迟使用单帧、单线程、单 BPU core，多线程 FPS 使用 2 个并发线程。

<a id="command"></a>
## 评估命令

先将源和统一板端运行产生的完整 uint8 灰度 matte 保存到唯一路径，然后在仓库根目录运行：

```bash
python3 samples/vision/modnet/evaluator/compare.py \
  --legacy-matte /tmp/modnet-run-legacy/matte.png \
  --unified-matte /tmp/modnet-run-unified/matte.png \
  --atol 0 \
  --output /tmp/modnet-run-unified/compare.json
```

命令要求几何相同，只有 uint8 最大绝对差不超过 `--atol` 时返回 `0`，否则返回 `2`。它不下载、不推理，也不宣称板测。

<a id="metrics"></a>
## 指标

报告包含 matte shape、dtype、最大绝对差、平均绝对差和精确相等性。源没有 ground truth，因此不报告数据集 SAD/MAD/IoU。上面的性能值保留源测试条件并标为历史值。

<a id="outputs"></a>
## 输出

`compare.py` 打印并可保存 JSON。完整输入 matte 和报告应放在唯一 run 目录；合成图是独立的视觉制品，不参与 matte tensor 对照。

<a id="reference-results"></a>
## 参考结果

源历史参考见上表。当前主机比较 fixture 通过；源/统一板端和精度评估为 `not-run`。

<a id="boundaries"></a>
## 边界

本评估器是自包含的已保存 matte 对照工具，不是人像数据集评估器，不虚构质量标签，也不会把主机 fixture 结果升级为板端验证。
