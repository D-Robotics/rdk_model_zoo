[English](README.md) | 简体中文

# YOLOWorld 评估器

<a id="dataset"></a>

<a id="directory"></a>
## 目录结构

```text
evaluator/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── compare.py  # Python 脚本
└── source_reference.py  # Python 脚本
```

<a id="environment"></a>
## 数据与环境

评估器在**真实 X5 执行目标**上，用同一模型、图片、离线词向量和 prompt
逐阶段对比锁定的原始 X5 实现（从 Git 历史加载）与本 sample 的实现。输入图片是
`test_data/dog.jpeg`，prompt 是逗号分隔的词。主机单测注入 fake runtime。
评估器要求通过板卡身份门禁，并需要 `hbm_runtime`、OpenCV、NumPy
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
用同一图片、词向量与模型运行两套实现的 pre/infer/post，并捕获两边的
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
## 源记录参考与边界

| 参考记录 | 输入/协议 | 数值 | 来源 |
| --- | --- | --- | --- |
| X5 源 sample 协议 | 640 图片；32×512 文本；8400 行分数/框 | 没有发布延迟或 mAP 表 | 源 evaluator README |
| 板端一致性检查 | `dog` 提示、`test_data/dog.jpeg`、score/NMS 0.05/0.45 | 在准备好的 X5 上运行对照器；通过要求全部检查 true 且 `max_abs_diff` 0.0 | 本评估器 |

已发布的协议事实为：`yolo_world.bin`、640 图片输入、32 个 512 宽文本槽、
8400 行分数，以及 0.05/0.45 的 score/NMS 默认值。已记录的板端运行日志会打印
HBRT 库与模型构建小版本不一致的警告，证据中原样保留。其它提示、图片或板卡
仍需各自运行本命令。离线词向量是必需的模型伴随资产，不是普通分类标签文件。
