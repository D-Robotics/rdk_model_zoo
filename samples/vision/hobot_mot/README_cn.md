[English](README.md) | 简体中文

<a id="overview"></a>
## HobotMot（IOU 2.0）多目标跟踪器

HobotMot 是 D-Robotics 的 CPU 多目标跟踪器。逐帧输入检测框、置信度和类别，
即可得到目标的轨迹 ID，并获知哪些目标已经离开画面。可将 YOLO 等检测器的输出转换为本示例的输入格式后接入。

- 上游源码：<https://github.com/D-Robotics/mot>（Apache-2.0），固定在提交
  `0120205146d88c3474bfc2936522e349e169fa8d`
- 算法：IOU 2.0（`tracker_type: "IOU_2.0"`），可选卡尔曼滤波平滑轨迹
  （`use_kalman_filter: true`）
- 时间戳单位为毫秒且必须严格递增；输出行保持输入顺序（每个输入检测对应一行）

<a id="directory"></a>
## 目录结构

```text
hobot_mot/
├── model/               # 固定版本上游源码准备 + 跟踪器配置
├── runtime/python/      # Python 回放 CLI 与 HobotMotTracker 模型类
├── runtime/cpp/         # C++ 跟踪器、C ABI 绑定与回放可执行程序
└── test_data/           # 72 帧检测框示例
```

<a id="support-matrix"></a>
## 支持板卡

| 板卡 | Python | C++ |
| --- | --- | --- |
| RDK X5 | 支持 | 支持 |
| RDK S100 | 支持 | 支持 |
| RDK S100P | 支持 | 支持 |
| RDK S600 | 支持 | 支持 |

四种板卡采用相同的源码和配置，在各自板端构建。Python 与 C++ 均调用 HobotMot 原生库。

<a id="prerequisites"></a>
## 环境要求

- RDK X5、S100、S100P 或 S600（ARM64 Linux）
- C++17 编译器、CMake 3.16 及以上、Python 3.8 及以上
- 约 20 MB 可用空间
- 首次准备需要下载源码；离线方法见 [模型准备](model/README_cn.md)

<a id="quickstart"></a>
## 快速体验

在开发板上、于本样例目录执行：

```bash
python3 model/download.py
cmake -B runtime/cpp/build -S runtime/cpp && cmake --build runtime/cpp/build
./runtime/cpp/run.sh
```

`download.py` 拉取固定版本的上游源码，校验 SHA-256 后解压到
`model/vendor/`。构建生成跟踪器和 `hobot_mot_replay` 可执行程序。
`run.sh` 用默认配置（`model/configs/iou2.json`）回放
`test_data/detections.jsonl`，每帧打印一行；运行成功以退出码 0 结束。

<a id="expected-results"></a>
## 预期结果

使用自带示例序列时，最初几帧输出如下（C++ 运行时）：

```text
frame 0 ts 1000: [cls 0 id 1 (100,100,160,260) score 0.95 valid] [cls 0 id 2 (400,300,460,460) score 0.92 valid] [cls 1 id 1 (480,200,600,300) score 0.9 valid]
frame 1 ts 1040: [cls 0 id 1 (104,100,164,260) score 0.95 valid] [cls 0 id 2 (400,300,460,460) score 0.92 valid] [cls 1 id 1 (474,200,594,300) score 0.9 valid]
...
frame 10 ts 1400:
...
frame 70 ts 3800: [cls 0 id 1 (380,100,440,260) score 0.95 valid] [cls 0 id 2 (400,300,460,460) score 0.92 valid] [removed cls 1 id 1]
```

输出解读：每个输入检测对应一行（顺序不变），行走的人保持 `id 1`，静止的
人保持 `id 2`；第 10 帧为空输入（无输出行，轨迹老化）；第 12–14 帧的低分
干扰框保持 `id -1 invalid`（低于 `min_score`）；第 19 帧后离开的汽车在消失
窗口（`vanish_frame_count * time_gap` = 50 × 40 ms）到期时被报告一次
`[removed cls 1 id 1]`。传入 `--output results.json` 可同时写出完整结果
JSON（格式见 [runtime/cpp/README_cn.md](runtime/cpp/README_cn.md)）。

<a id="entry-points"></a>
## 入口索引

| 入口 | 用途 |
| --- | --- |
| [model/download.py](model/download.py) | 拉取并校验固定版本上游源码（见 [model/README_cn.md](model/README_cn.md)） |
| [runtime/python/main.py](runtime/python/main.py) | Python 回放 CLI 与 `HobotMotTracker` 类（见 [runtime/python/README_cn.md](runtime/python/README_cn.md)） |
| [runtime/cpp/run.sh](runtime/cpp/run.sh) | 运行 C++ 回放程序（见 [runtime/cpp/README_cn.md](runtime/cpp/README_cn.md)） |

<a id="license"></a>
## 许可

上游 D-Robotics/mot 源码为 Apache License 2.0（版权 D-Robotics），见
[runtime/cpp/THIRD_PARTY_LICENSES.md](runtime/cpp/THIRD_PARTY_LICENSES.md)。
本目录内的样例代码遵循仓库许可证。
