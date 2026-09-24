# YOLOWorld 评估器

<a id="dataset"></a>
<a id="environment"></a>
## 数据与环境

评估器在**真实 X5 执行目标**上，用同一模型、图片、离线词向量和 prompt
逐阶段对比固定源实现与统一实现。输入图片是 `test_data/dog.jpeg`，prompt
是逗号分隔的词。主机单测注入 fake runtime，只验证主机边界，不构成板端或
精度证据。评估器要求通过板卡身份门禁，并需要 `hbm_runtime`、OpenCV、NumPy
和精确的 `yolo_world.bin`；命令不会安装依赖或下载模型。

<a id="command"></a>
## 命令

在仓库根目录先显式准备模型，再写入一个不存在的新目录：

```bash
python3 samples/vision/yoloworld/evaluator/compare.py --target x5 --output-dir /absolute/new-yoloworld-evidence --asset-id x5:yoloworld:yolo_world.bin
```

`--model-path /absolute/yolo_world.bin` 必须同时给出精确 `--asset-id`。
`--test-img`、`--vocab-file`、`--prompts`、`--score-thres`、`--nms-thres`、
`--priority` 和 `--bpu-cores` 会同时作用于两套实现。命令先检查真实 target，再
用同一图片、词向量与模型运行源实现和统一实现的 pre/forward/post，并捕获两边的
预处理输入、raw 分数/框张量与最终检测结果。仅当全部检查通过时返回 0，两边
不一致返回 1，运行失败返回 2。

<a id="metrics"></a>
<a id="outputs"></a>
## 指标与输出

这是张量对拍，不是数据集 mAP 或性能测试。输出为 `comparison.json` 以及每个记录
的输入、raw 输出与结果数组各自的 `.npy` 文件。manifest 用 SHA-256 绑定
`target`、`asset_id`、模型、图像、词向量与代码摘要，记录两边实际 runtime
metadata、prompt 列表、阈值、`argv`、`cwd`、`started_utc`/`finished_utc` 与
`return_code`，并逐个列出保存的数组文件及其摘要。输入与类别 ID 要求完全相等；
raw 张量 `atol=1e-5`，框 `1e-4`，分数 `1e-5`。执行失败时仍写出带 `error`、
`return_code: 2`、`passed: false` 的 manifest，任何输出目录都不会覆盖。

<a id="reference-results"></a>
<a id="boundaries"></a>
## 历史参考与边界

| 源记录 | 输入/协议 | 历史性能 | 状态 |
| --- | --- | --- | --- |
| 固定 X5 YOLOWorld sample | 640 图片；32×512 文本；8400 行分数/框 | 没有发布延迟或 mAP 表 | 保留事实；没有新测量 |

固定源 evaluator README 没有发布 benchmark 表。因此只保留可核对的历史事实：
`yolo_world.bin`、640 图片输入、32 个 512 宽文本槽、8400 行分数，以及
0.05/0.45 的 score/NMS 默认值；不宣称延迟或 mAP 数值。直到命令生成证据前，
板端执行和模型可用性均为 `not-run`。离线词向量是必需的模型伴随资产，不是
普通分类标签文件。
