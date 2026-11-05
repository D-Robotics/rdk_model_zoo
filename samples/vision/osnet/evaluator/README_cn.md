[English](README.md) | 简体中文

# OSNet 评测

<a id="dataset"></a>
## 数据集

按数据集条款取得 [Market-1501-v15.09.15](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view)，解压到 /data/Market-1501-v15.09.15。预期目录：bounding_box_train（12936）、query（3368）、bounding_box_test（排除 pid=-1 后 15913）。原记录归档 MD5 为 65005ab7d12ec1c44de4eeafe813e68a。训练图只作校准，完整评测使用 query + gallery。

<a id="directory"></a>
## 目录

main.py 采集或复算特征；metrics.py 定义检索协议；evidence/ 保存原始测量及来源身份。

<a id="environment"></a>
## 环境

X5 采集使用 Python Sample 相同的 Runtime 和预处理。主机 ONNX 采集另需 ONNX Runtime；采集/评分需要 NumPy/Pillow。已有特征可在主机无板端 SDK 时评分，每份特征矩阵约 40 MB，另需数据集空间。Runtime 性能用独立 C++ 程序测量。

<a id="command"></a>
## 命令

在仓库根目录执行，输出目录必须是新目录。第一条在 X5 执行，后两条在主机执行。总耗时取决于设备且包含采集/评分，不代表 BPU 延迟。

```bash
python3 samples/vision/osnet/evaluator/main.py \
  --dataset /data/Market-1501-v15.09.15 \
  --model-path samples/vision/osnet/model/spatial-rgb-late8.bin \
  --output-dir /data/osnet-eval-x5
python3 samples/vision/osnet/evaluator/main.py --backend onnx \
  --dataset /data/Market-1501-v15.09.15 \
  --model-path /work/weights/osnet_x1_0_spatial.onnx --output-dir /data/osnet-eval-float
python3 samples/vision/osnet/evaluator/main.py \
  --dataset /data/Market-1501-v15.09.15 \
  --capture-dir /data/osnet-eval-x5 --output-dir /data/osnet-score-raw --raw
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| --dataset | required | Market1501 root / 数据集根目录 |
| --output-dir | required | New result directory / 新结果目录 |
| --capture-dir | null | Re-score saved raw features / 复算特征 |
| --backend | x5 | x5 or onnx / 执行后端 |
| --model-path | null | Required for capture / 采集必需 |
| --variant | spatial-rgb-late8 | Registered X5 variant / X5 登记版本 |
| --raw | false | Score without L2 / 不做 L2 检索 |

<a id="metrics"></a>
## 指标

按文件名排序，排除 pid=-1，保留 pid=0 干扰图。每个 query 排除同身份同摄像头的 gallery。Rank-1 为首个有效检索命中率；mAP 平均每个 query 在正确匹配位置的精度。默认使用 L2 归一化后的平方欧氏距离，排序等价于余弦；--raw 使用未归一化特征。不做 reranking，无有效匹配的 query 不计入分母。

<a id="outputs"></a>
## 输出

每个结果目录保存原始 features.npy、有序 manifest.json 和 accuracy.json，字段含 rank1_percent/mAP_percent/valid_queries/distance/provenance。复算会核对清单与数据集顺序。ONNX 和缓存评分记录 board_test=not-run，仅成功的 X5 采集记录 executed。

<a id="reference-results"></a>
## 参考结果

| x1.0 variant | Rank-1 | mAP | C++ E2E P50 / P95 |
| --- | --- | --- | --- |
| FP32 reference | 94.2399% | 83.5696% | — |
| spatial-rgb-late8 | 94.0618% | 83.3643% | 8.614 / 9.641 ms |
| spatial-rgb-int16 | 94.2102% | 83.5551% | 9.018 / 10.040 ms |
| spatial-rgb | 93.3195% | 81.7215% | 1.670 / 3.488 ms |

数值来自 2026-10-29 原实验：3368 query + 15913 gallery，L2 归一化平方欧氏距离，无 reranking。C++ 预热 50 次、计时 3000 次，batch=1、单路，BPU 996 MHz、CPU schedutil。E2E 为已解码 64×128 RGB 人体块 → Pillow 等价 NEON 缩放至 128×256 → 同步 Runtime → L2，不含解码、检测和匹配；其他尺寸不能套用这些延迟。

[原报告](evidence/2026-10-29-report.zh-CN.md) · [证据范围](evidence/README.md)。

<a id="boundaries"></a>
## 适用范围

完整 CLI 评测要求 Market1501 query/gallery 的完整规模和 512 维特征。32 张图的仿真对照不能代替检索精度。原 C++ 测试为单路，没有双路吞吐结论；重新编译或改变人体块预处理后须重新对照。

### Workbench 本地验收

先用 `evaluator/workbench.py measure --weights WEIGHTS --output MEASUREMENT` 实测推理路径的参数量和 Conv/Linear MAC（每 MAC 计 2 FLOPs，不含分类训练头）。提交已审阅源码，再执行 `freeze --workspace WB --dataset DATASET --model BIN --measurement MEASUREMENT --guide GUIDE --fixture FIXTURE --build-id ID --image-id IMAGE_ID`。IMAGE_ID 必须是完整的 sha256 镜像 ID。冻结文件记录源码、模型、完整图像字节清单、性能输入和精度预算。

ReID 固定 3,368 query + 15,913 有效 gallery，L2 归一化、平方欧氏距离，排除同身份同相机，不重排序。本地技术验收要求 Rank-1 和 mAP 相对同口径 FP32 ONNX 均下降不超过 0.01；发布授权另行确认，不能在看到结果后修改预算。

原生 evaluator 同时支持空间头 ONNX 与 X5 BIN。`--onnx-threads` 默认 4。`--capture-only` 保存完整特征和图像字节清单，不计算精度；之后在主机使用 `--capture-dir` 评分，并对照当前 Market1501 图像字节检查 `dataset-sha256.json`。保留 capture.json 和 accuracy.json，只有完成评分才算精度通过。

```bash
python3 samples/vision/osnet/evaluator/main.py --backend onnx --model-path /absolute/spatial.onnx --dataset /absolute/Market1501 --output-dir /absolute/new/float
python3 samples/vision/osnet/evaluator/main.py --backend x5 --model-path /absolute/spatial-rgb-late8.bin --dataset /absolute/Market1501 --output-dir /absolute/new/board --capture-only
python3 samples/vision/osnet/evaluator/main.py --capture-dir /absolute/board --dataset /absolute/Market1501 --output-dir /absolute/new/board-score
```

原生 hrt_model_exec 使用已核验的固定输入，1/2 并发，各自独立进程预热 20 帧，再运行 3 轮、每轮总计 200 帧。保存命令和原始日志。`workbench.py aggregate-runtime --logs LOGS --input INPUT_BIN --output PERFORMANCE` 检查真实完成帧数，以总帧数 / 共同墙钟计算吞吐量。C++ 按其指南运行 1/2 流，保留逐帧阶段数据及测试前、中、后的环境记录；不擅自调频。

对原 BIN 用 hb_perf 重新生成 OE 分析，记录工具版本，不将新报告冒充遗失的原转换报告。板端证据放入 `runs/benchmark/BUILD/work/board` 后，执行 `stage_local.py --workspace WB --build-id BUILD --float-run-id FLOAT_BUILD --oe-run-id OE_BUILD`。门禁检查全量数据身份、精度预算、模型与输入输出字节、6 轮 Runtime、C++ 阶段相加及墙钟吞吐量，再写入本地候选包。

`oe_report.html` 保留工具生成的独立 BPU 子图 HTML；完整报告目录保存在私有 conversion run。原始逐层量化相似度未恢复，保持缺失。制品、数据和完整报告不进 Git；候选包供新版 Web 本地预览使用，发布审批不会由脚本自动通过。

S 系列流程：[OSNet S100/S100P/S600 Workbench 验收](S_SERIES_cn.md)。
