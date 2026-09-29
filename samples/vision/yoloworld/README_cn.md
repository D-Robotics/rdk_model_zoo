# YOLOWorld X5 开放词汇检测

<a id="overview"></a>
## 概述

本 sample 按固定源提交 `ac115717197920355fc390bb04299b20e6436864` 迁移 X5
YOLOWorld Python 协议。它从用户选择的离线词向量中检测词语；词向量 JSON
是必需的文本输入资产，不是普通标签表。算法参考为 [YOLO-World](https://github.com/AILab-CVC/YOLO-World)，属于开放词汇区域检测。

<a id="support-matrix"></a>
## 支持矩阵（support-matrix）

| 目标 | Python | C++ | 资产 | 状态 |
| --- | --- | --- | --- | --- |
| X5 | 本 sample 支持 | 未提供 | `x5:yoloworld:yolo_world.bin` | 注入 runtime 的主机测试通过；2026-09-24 在一块 X5 8GB 和一块 X5 4GB 上以 `dog` 提示和 `test_data/dog.jpeg` 完成 source/unified 对照并全部通过（[8GB](../../../docs/releases/unified-migration/evidence/2026-09-24-b7-python-comparison/)、[4GB](../../../docs/releases/unified-migration/evidence/2026-09-24-b7-other-x5-variants/)） |
| S100/S100P/S600 | 无发布资产 | 未提供 | 无 | 不支持 |

板端验证只覆盖该提示/图片组合，是张量一致性，不是全词汇精度或时延测量。

<a id="prerequisites"></a>
## 前置条件（prerequisites）

主机检查使用仓库 `.venv` 中的 Python 3.14.7、NumPy 2.5.3、OpenCV 4.14.0
和 PyYAML 6.0.3；这些是主机 fixture 版本，不能替代板端 SDK 依赖。板端
运行需要与系统镜像匹配的 X5 Python 和 `hbm_runtime`。模型必须用显式下载
命令准备；必需词向量 `test_data/offline_vocabulary_embeddings.json` 随
sample 提供。

<a id="quickstart"></a>
## 快速开始（quickstart）

模型准备是显式动作，可能访问发布归档：

```bash
bash samples/vision/yoloworld/model/download.sh --target x5
python3 samples/vision/yoloworld/runtime/python/main.py --target x5 --prompts dog
```

第二条命令要求识别到 X5。无网络协议检查使用：

```bash
.venv/bin/python samples/vision/yoloworld/runtime/python/main.py --dry-run --target x5 --prompts dog
```

<a id="expected-results"></a>
## 预期结果（expected-results）

模型输入为 `float32[1,3,640,640]` RGB NCHW 与
`float32[1,32,512,1]` 文本向量。图片按最长边缩放，放在零填充画布左上角，
不做 mean/std 归一化。原生 raw 输出逻辑形状为 `float32[1,8400,32]` 分数和
`float32[1,8400,4]` 框；源 runtime 可能额外暴露末尾 singleton，runner 原样保留，
由 post-process 消费且不做数值转换。后处理为每行选择最高文本槽、score 阈值 `0.05`、
按类别 NMS `0.45`、坐标还原，结果为 `boxes[N,4]`、`scores[N]` 和词汇
`class_ids[N]`。

<a id="directory"></a>
## 目录（directory）

`model/` 负责显式资产准备；`runtime/python/` 负责三阶段任务、懒加载 runtime
和 CLI；`conversion/` 记录源协议与转换缺口；`evaluator/` 生成同板源/统一证据；
`test_data/` 保存源图片和离线词向量；`tests/` 使用注入 runtime 做主机测试。

<a id="entry-points"></a>
## 入口（entry-points）

- `runtime/python/main.py`、`runtime/python/run.sh`：推理 CLI。
- `model/download.py`、`model/download.sh`：显式模型准备。
- `evaluator/compare.py`：同板对拍证据，不下载。
- Python API：`YOLOWorldTask.pre_process`、`forward`、`post_process`、`predict`。

<a id="license"></a>
## 许可证

迁移源代码使用 Apache-2.0；源文件头和
`platforms/x5/samples/vision/yoloworld` 中的来源记录适用。
