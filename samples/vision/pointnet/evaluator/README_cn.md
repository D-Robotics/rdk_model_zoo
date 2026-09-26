[English](README.md) | 简体中文

# PointNet 验证与历史结果

<a id="dataset"></a>
## 输入数据

`../test_data/chair.pts` 是交付的一份 XYZ 点云，不是带标签评估子集。源中没有数据集下载器、
标签数组、split 版本或全数据集精度评估器。功能 smoke 使用未修改的交付文件；真实指标需另行准备带标签数据。

<a id="environment"></a>
## 环境

主机测试依赖 NumPy、PyYAML；板端 smoke 另需配套 S100 HBM/SDK，绘图可选 matplotlib。
测试使用与 CLI 相同的 PointNet 阶段和 binding。本目录保存记录，不另写推理实现。

<a id="command"></a>
## 命令

```bash
# cwd: repository root; host numerical/contract checks, no SDK
python3 -m unittest discover -s samples/vision/pointnet/tests
```
```bash
# cwd: repository root; on S100 with the published HBM already prepared
python3 samples/vision/pointnet/runtime/python/main.py --target s100 --no-plot --output-dir outputs/pointnet-check
```

Smoke 参数：`--target s100`（parser 默认 auto→s100）、`--no-plot`（默认 false）、
`--output-dir outputs/pointnet-check`（默认 outputs/pointnet）。输入默认交付椅子点云；
其余参数见[完整运行时表](../runtime/python/README_cn.md#parameters)。主机测试使用小 fixture；
本轮未测板端耗时。

<a id="metrics"></a>
## 指标解释

功能运行应得到 N 个 0..3 标签，计数之和为 N。可通过下方历史图对照点云区域。
每份点云必须出现四种颜色不是通用成功条件，也不是精度指标。新入口的数据集 mIoU、
逐部件 IoU 和吞吐均未测。源记录的 trans/pred 没有明确指标定义，不能称作 mIoU。

<a id="outputs"></a>
## 输出

`outputs/pointnet-check/labels.npy` 保存点标签，`result.json` 保存计数、归一化和 metadata。
去掉 `--no-plot` 还会生成 `result_orig.png`、`result.png`，不从图像虚构指标摘要。

![原始椅子](../test_data/readme_img/chair.png)
![源分割图](../test_data/readme_img/chair_res.png)

<a id="reference-results"></a>
## 历史参考结果

原 S 分支 evaluator 记录以下 `hrt_model_exec` 数值，但未注明延迟单位、完整调用命令、SDK
版本或制品 digest。下表原样保留，不能当作当前统一入口的测量结果。

| Threads | Frames | Total Latency | Average Latency | FPS |
| --- | --- | --- | --- | --- |
| 1 | 100 | 143.63 | 1.43 | 689.61 |
| 2 | 100 | 216.20 | 2.16 | 914.32 |
| 4 | 100 | 429.76 | 4.30 | 910.70 |
| 8 | 100 | 839.86 | 8.35 | 910.84 |

原转换记录另有 int16 “trans > 0.9999”和“pred > 0.98”，见[保留截图与限制](../conversion/README_cn.md#calibration)。
统一入口板测状态：**not-run**。主机检查见上方命令，fixture 通过不等于硬件验收。

<a id="boundaries"></a>
## 边界

此处没有带标签数据集评估器、BPU benchmark 封装或本轮板端数值对照。复现历史性能需要补齐
制品/环境/命令身份。S100P/S600 无已发布 PointNet 制品，不能静默映射至 S100。
代码遵循 [Apache-2.0](../../../../LICENSE)。
