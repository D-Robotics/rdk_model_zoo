[English](README.md) | [简体中文](README_cn.md)

# DiffusionDrive 转换

两份源 PTQ YAML 逐字节保留。它们记录编译设置，本身不能复现缺失的导出和校准链路。本次迁移未导出 ONNX、执行 OE、下载模型或运行板卡。

<a id="source-model"></a>
## 源模型

源文档引用官方 [DiffusionDrive 项目](https://github.com/hustvl/DiffusionDrive)及 NAVSIM checkpoint，但未固定 checkpoint URL/摘要、上游提交或导出器。[论文](https://openaccess.thecvf.com/content/CVPR2025/html/Liao_DiffusionDrive_Truncated_Diffusion_Model_for_End-to-End_Autonomous_Driving_CVPR_2025_paper.html)说明算法，不能标识本次发布对应的模型字节。声明可复现重建前，必须取得匹配代码、权重和导出流程。

使用已发布资产推理时，直接按[模型准备](../model/README_cn.md)操作，无需自行转换。发布 HBM 的校验和只认证这些文件，不认证独立导出的模型。

<a id="toolchain-targets"></a>
## 工具链与目标

源记录的 x86 Linux 工具链镜像：

```text
registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0
```

| 目标 | 配置 | March | 相对本目录的输出 |
| --- | --- | --- | --- |
| S100P | `configs/diffusiondrive_r34_256x1024_s100p.yaml` | `nash-m` | `build/s100p/hbm/diffusiondrive_r34_256x1024_s100p.hbm` |
| S600 | `configs/diffusiondrive_r34_256x1024_s600.yaml` | `nash-p` | `build/s600/hbm/diffusiondrive_r34_256x1024_s600.hbm` |

两者均请求全图 INT16 激活/max 校准、O2/latency、1 核、32 个编译任务、开启缓存且输入/输出无 padding。源记录称 v3.7.0 不支持 INT16 GridSample，因此该算子保持 INT8。这是 INT16 优先的图，不代表每个算子或公开 IO 张量都是 int16。这里没有 S100/X5 配置。

<a id="export"></a>
## 导出契约与缺失代码

所需确定性 ONNX 有四个浮点特征输入：

| 名称 | 形状 |
| --- | --- |
| `camera` | `[1,3,256,1024]` |
| `lidar` | `[1,1,256,256]` |
| `status` | `[1,8]` |
| `noise` | `[1,20,8,2]` |

输出为轨迹 `[1,8,3]`、Agent 状态 `[1,30,5]`、Agent logits `[1,30]`、BEV logits `[1,7,128,256]`，名称见[运行契约](../runtime/python/README_cn.md#stage-io)。噪声保持显式输入。源说明要求将 ScatterND 式原位写入改为拼接，将固定自适应平均池化改为静态深度卷积。

未提供这些改写/导出的实现。调用者需准备相对本目录的 `build/diffusiondrive_navsim_bpu_clean_float.onnx`。把任意图改成该文件名不能建立 IO 契约或保持原模型行为，PTQ 前应先将导出浮点输出与目标 checkpoint 验证对照。

<a id="calibration"></a>
## 校准准备

源要求至少 100 个真实 NAVSIM mini 样本。每个样本分别在 `calibration_data/camera`、`calibration_data/lidar`、`calibration_data/status`、`calibration_data/noise` 提供对应的有限 float32 `.npy`。四目录保持同一样本配对，输入顺序为 `camera;lidar;status;noise`，形状如上。YAML 声明 featuremap/NCHW，`separate_batch: false`。

源未提供原校准集、特征准备/导出脚本或样本清单。六份演示归档不满足 >=100 样本配方；重复它们不能补足代表性数据。校准目录需要逻辑浮点特征，不能输入已量化的运行缓冲区。新建校准集时保留数据集身份、张量摘要和准备程序版本。

<a id="compile"></a>
## 前提齐备后的编译

在记录的工具链环境内，从仓库根目录进入本目录，让 YAML 相对路径按同一工作目录解析：

```bash
cd samples/vision/diffusiondrive/conversion
hb_compile -c configs/diffusiondrive_r34_256x1024_s600.yaml
hb_compile -c configs/diffusiondrive_r34_256x1024_s100p.yaml
```

这些是实际源编译命令，**不是本次迁移已执行的命令**。执行前必须准备 ONNX 和四份完整校准目录。保留编译器版本、日志、生成报告、算子落位报告和 HBM 摘要。编译返回码不能证明数值正确或可在目标上执行。这里没有自动下载工具链或伪造缺失输入的脚本。

<a id="validation"></a>
## 验证与历史选型依据

在各自匹配的板卡上先检查生成模型的实际 IO 元数据，再以有效输入执行运行入口。历史 HRT 形式如下，工作目录仍为本目录；model-info/perf 工具属于板端依赖：

```bash
hrt_model_exec model_info --model_file build/s600/hbm/diffusiondrive_r34_256x1024_s600.hbm
hrt_model_exec model_info --model_file build/s100p/hbm/diffusiondrive_r34_256x1024_s100p.hbm
```

性能测试需提供有效量化 `camera.bin,lidar.bin,status.bin,noise.bin`。任意随机数据可能超出 GridSample 支持范围。这些文件须反映模型实际类型/量化参数，不能直接使用浮点 NPZ 字节。显式准备后，历史单/双线程形式如下：

```bash
hrt_model_exec perf --model_file build/s600/hbm/diffusiondrive_r34_256x1024_s600.hbm --thread_num 1 --core_id 1 --input_file camera.bin,lidar.bin,status.bin,noise.bin
hrt_model_exec perf --model_file build/s600/hbm/diffusiondrive_r34_256x1024_s600.hbm --thread_num 2 --core_id 1 --input_file camera.bin,lidar.bin,status.bin,noise.bin
hrt_model_exec perf --model_file build/s100p/hbm/diffusiondrive_r34_256x1024_s100p.hbm --thread_num 1 --core_id 1 --input_file camera.bin,lidar.bin,status.bin,noise.bin
hrt_model_exec perf --model_file build/s100p/hbm/diffusiondrive_r34_256x1024_s100p.hbm --thread_num 2 --core_id 1 --input_file camera.bin,lidar.bin,status.bin,noise.bin
```

源记录：全 INT8 BEV 余弦为 0.370948；只将四个 BEV 末端节点改为 INT16，余弦仍为 0.371840，平均 IoU 为 0.143013，剩余失真归因于上游 `/_backbone/Add_6` 融合特征。INT16 优先/max 后，S600 case_000 的 BEV 余弦、一致率、平均 IoU 为 0.998918、0.944061、0.868425；S100P 为 0.998913、0.943726、0.865501，五案例均值为 0.998799、0.955664、0.819837。

这些是历史源记录，不是复测。源还记录所有分段在 BPU/CPU 推理 0.0 ms；case_017 下 S100P 单线程14.370 ms /69.375 FPS、双线程总吞吐71.109 FPS；S600 为7.215 ms /138.247 FPS、双线程143.767 FPS。[评估说明](../evaluator/README_cn.md#reference-results)保留完整精度/性能表，并区分 case_000 数值对照和 case_017 profiling。

<a id="artifacts"></a>
## 产物与身份

每个 HBM 保留在各自目标输出路径，同时保存编译日志以及 ONNX、校准、配置摘要清单。自定义转换模型不等于清单中的已发布资产。当前运行入口对外部路径也验证发布 SHA-256，因此新自定义资产需显式增加资产/绑定并验证，不能替换发布校验和来强行通过。

测试前检查实际名称、形状、类型、scale、zero point 和 axis。逻辑浮点参考不能证明物理 HBM 元数据。使用[离线评估器](../evaluator/README_cn.md)比较解码结果，同时保留原始张量与溯源记录。评估器只报告描述性指标，不提供发布门槛。

<a id="known-gaps"></a>
## 已知缺口

Checkpoint/导出版本、改写代码、原校准集与精确 profiling 输入二进制仍缺失。本次主机迁移未验证工具链可用性、完整算子落位、真实 HBM 元数据、OE 编译、板端输出对齐或性能。保留配置与文档是为了明确这些依赖，不将不完整源配方描述为已复现转换。
