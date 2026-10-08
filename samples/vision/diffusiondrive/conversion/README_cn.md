[English](README.md) | [简体中文](README_cn.md)

# DiffusionDrive 转换

两份源 PTQ YAML 逐字节保留，记录编译设置；执行前需按下文准备导出与校准输入。使用发布资产推理无需自行转换。

<a id="source-model"></a>
## 源模型

源文档引用官方 [DiffusionDrive 项目](https://github.com/hustvl/DiffusionDrive)及 NAVSIM checkpoint，未固定 checkpoint URL/摘要、上游提交或导出器。[论文](https://openaccess.thecvf.com/content/CVPR2025/html/Liao_DiffusionDrive_Truncated_Diffusion_Model_for_End-to-End_Autonomous_Driving_CVPR_2025_paper.html)说明算法本身。如需复现重建，先取得匹配的模型代码、权重和导出流程。

使用已发布资产推理时，直接按[模型准备](../model/README_cn.md)操作，无需自行转换。发布 HBM 的校验和认证这些发布文件；独立导出的模型需记录自己的摘要。

<a id="directory"></a>
## 目录结构

```text
conversion/
├── configs/  # configs 相关文件
├── README.md  # 英文说明
└── README_cn.md  # 中文说明
```

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

两者均请求全图 INT16 激活/max 校准、O2/latency、1 核、32 个编译任务、开启缓存且输入/输出无 padding。源记录称 v3.7.0 不支持 INT16 GridSample，因此该算子保持 INT8。该图为 INT16 优先配置；各算子与公开 IO 张量的实际类型以编译产物 metadata 为准。配置仅覆盖 S100P/S600，无 S100/X5 配置。

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

导出需完成上述改写，并提供相对本目录的 `build/diffusiondrive_navsim_bpu_clean_float.onnx`，与上文的输入/输出契约一致。PTQ 前应先将导出浮点输出与目标 checkpoint 验证对照。

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

这些是源编译命令。执行前必须准备 ONNX 和四份完整校准目录。保留编译器版本、日志、生成报告、算子落位报告和 HBM 摘要，并在编译后验证数值正确性与目标执行。工具链与导出/校准输入由使用者提供，本目录没有自动下载工具链或生成输入的脚本。

<a id="validation"></a>
## 验证

在各自匹配的板卡上先检查生成模型的实际 IO 元数据，再以有效输入执行运行入口。HRT 工具形式如下，工作目录仍为本目录；model-info/perf 工具属于板端依赖：

```bash
hrt_model_exec model_info --model_file build/s600/hbm/diffusiondrive_r34_256x1024_s600.hbm
hrt_model_exec model_info --model_file build/s100p/hbm/diffusiondrive_r34_256x1024_s100p.hbm
```

性能测试需提供有效量化 `camera.bin,lidar.bin,status.bin,noise.bin`。任意随机数据可能超出 GridSample 支持范围。这些文件须反映模型实际类型/量化参数，不能直接使用浮点 NPZ 字节。显式准备后，单/双线程形式如下：

```bash
hrt_model_exec perf --model_file build/s600/hbm/diffusiondrive_r34_256x1024_s600.hbm --thread_num 1 --core_id 1 --input_file camera.bin,lidar.bin,status.bin,noise.bin
hrt_model_exec perf --model_file build/s600/hbm/diffusiondrive_r34_256x1024_s600.hbm --thread_num 2 --core_id 1 --input_file camera.bin,lidar.bin,status.bin,noise.bin
hrt_model_exec perf --model_file build/s100p/hbm/diffusiondrive_r34_256x1024_s100p.hbm --thread_num 1 --core_id 1 --input_file camera.bin,lidar.bin,status.bin,noise.bin
hrt_model_exec perf --model_file build/s100p/hbm/diffusiondrive_r34_256x1024_s100p.hbm --thread_num 2 --core_id 1 --input_file camera.bin,lidar.bin,status.bin,noise.bin
```

源记录：全 INT8 BEV 余弦为 0.370948；只将四个 BEV 末端节点改为 INT16，余弦仍为 0.371840，平均 IoU 为 0.143013，剩余失真归因于上游 `/_backbone/Add_6` 融合特征。INT16 优先/max 后，S600 case_000 的 BEV 余弦、一致率、平均 IoU 为 0.998918、0.944061、0.868425；S100P 为 0.998913、0.943726、0.865501，五案例均值为 0.998799、0.955664、0.819837。

以下为源发布记录：所有分段在 BPU 执行、CPU 推理 0.0 ms；case_017 下 S100P 单线程14.370 ms /69.375 FPS、双线程总吞吐71.109 FPS；S600 为7.215 ms /138.247 FPS、双线程143.767 FPS。[评估说明](../evaluator/README_cn.md#reference-results)保留完整精度/性能表，并区分 case_000 数值对照和 case_017 profiling。

<a id="artifacts"></a>
## 产物与身份

每个 HBM 保留在各自目标输出路径，同时保存编译日志以及 ONNX、校准、配置摘要清单。自定义转换的模型使用独立的资产身份与绑定：注册新资产、记录其摘要并显式验证。运行入口对发布路径验证发布 SHA-256，自定义制品通过其注册绑定与记录摘要运行；发布校验和保持不变。

测试前检查实际名称、形状、类型、scale、zero point 和 axis。物理 HBM 元数据为权威依据，逻辑浮点参考仅用于数值对照。使用[离线评估器](../evaluator/README_cn.md)比较解码结果，同时保留原始张量与溯源记录。评估器报告描述性指标；发布门槛由你的发布流程确定。

<a id="known-gaps"></a>
## 补充准备

Checkpoint/导出版本、改写代码、原校准集与精确 profiling 输入二进制是本配方要求使用者准备的输入。工具链可用性、完整算子落位、真实 HBM 元数据、OE 编译、板端输出对齐与性能在实际执行时验证。保留的配置与文档完整列出这些依赖，便于执行前逐项准备与检查。
