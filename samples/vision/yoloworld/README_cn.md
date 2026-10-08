# YOLOWorld X5 开放词汇检测

<a id="overview"></a>

## 概述

本 sample 在 X5 BPU 上从用户选择的离线词向量中检测词语；词向量 JSON
是必需的文本输入资产，不是普通标签表。算法参考为
[YOLO-World](https://github.com/AILab-CVC/YOLO-World)，属于开放词汇区域检测。
来源：X5 平台 sample 交付 @ `ac115717197920355fc390bb04299b20e6436864`。

<a id="directory"></a>
## 目录结构

```text
yoloworld/
├── conversion/  # 导出与量化配置
├── evaluator/  # 评估程序与指标
├── model/  # 模型文件与下载脚本
├── runtime/  # 推理程序
├── test_data/  # 示例输入
├── tests/  # 自动化测试
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

<a id="support-matrix"></a>
## 支持矩阵（support-matrix）

| 目标 | Python | C++ | 资产 | 状态 |
| --- | --- | --- | --- | --- |
| X5 | supported | 未提供 | `x5:yoloworld:yolo_world.bin` | supported |
| S100/S100P/S600 | 无发布资产 | 未提供 | 无 | 不支持 |

<a id="prerequisites"></a>
## 前置条件（prerequisites）

主机检查使用仓库 `.venv` 中的 Python 3.14.7、NumPy 2.5.3、OpenCV 4.14.0
和 PyYAML 6.0.3。板端
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

<a id="entry-points"></a>
## 入口（entry-points）

- `runtime/python/main.py`、`runtime/python/run.sh`：推理 CLI。
- `model/download.py`、`model/download.sh`：显式模型准备。
- `evaluator/compare.py`：实现对拍比较，不下载。
- Python API：`YOLOWorldTask.preprocess`、`infer`、`postprocess`、`predict`
  （`pre_process`/`forward`/`post_process` 为方法别名）。

<a id="license"></a>
## 许可证

本 sample 代码使用仓库 Apache-2.0 许可，保留 X5 平台交付
（`platforms/x5/samples/vision/yoloworld`）的源文件头。
