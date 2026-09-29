[English](README.md) | [简体中文](README_cn.md)

# YOLO26 Depth 模型转换

本目录统一 X5 Mapper 与 S 系列 `hb_compile` 入口，同时保留各自的数据格式。
流程包括 ONNX 导出、确定性校准数据、解析后的编译配置、日志和制品摘要。
本轮已验证主机准备逻辑；**尚未执行真实 Torch 导出、OpenExplorer 编译或板测**。

<a id="source-model"></a>
## 源模型与计算图边界

请提供兼容 Ultralytics `Depth` 头的已训练
`yolo26{n,s,m,l,x}-depth.pt`。工具不附带或自动下载权重。
源分支记录 `ultralytics==8.4.105`，X5 另记录 `torch==1.13.0`；原始依赖分别保留在
`requirements-x5-source.txt` 与 `requirements-s-source.txt`。这些历史版本不代表本轮已验证
重新安装成功，也不代表支持任意 Python 版本。

| 目标与变体 | ONNX 输出 | 运行时前处理 | CPU 解码 |
|---|---|---|---|
| X5 n/s/m/l/x | 已校准 log-depth `[1,192,192,1]` | 768 方形线性 letterbox，填充值 114，NV12 | exp、放大到 768、去 padding、还原原图 |
| S100/S100P/S600 n/s/m | 同上 | 同上，传入打包 NV12 | 同上 |
| S100/S100P/S600 l/x | 原始 logit `[1,192,192,1]` | 768 方形线性拉伸，RGB float32 NCHW `/255` | clip `[-4,5]`、权重 scale/bias、exp、直接还原原图 |

`export.py` 保留源 attention、area-attention 和 Depth 细化逻辑。
log 边界止于 clip 与 scale/bias；**exp 和最终尺寸还原在 CPU 上执行**。
原 S 文档将其写成图内处理，与实际导出及运行代码不符。

导出报告会记录权重中的校准参数。已发布 S l/x 的运行时参数分别为
`(cal_a, cal_b)=(1,-0.2498779296875)` / `(1,-0.316650390625)`。
若重新训练的权重采用其他参数，需要显式更新对应运行时绑定。
结果是相对深度，不能直接声明为经过独立校准的米制距离。

<a id="toolchain-targets"></a>
## 工具链与目标配置

在 x86 Linux 主机的对应 OE 环境中转换。源 X5 记录为
OpenExplorer 1.2.8 / Mapper 1.24.3、O3 latency 优化、尾部卷积 int16 输出。
S 仅记录镜像名称 `ai_toolchain_ubuntu_22_s100_s600_gpu`，没有固定镜像摘要及编译器版本。
新一轮转换应记录实际工具版本。

源 X5 离线镜像地址为
`https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz`。
取得并加载镜像后，原容器启动方式如下：

```bash
# 从仓库根目录执行；将 /path/to/work 换成外部工作目录。
docker run -it --rm --network host --shm-size=15g \
  -v "$PWD":/workspace -v /path/to/work:/work --workdir /workspace \
  openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 /bin/bash
```

本轮没有重新验证镜像可用性、安装及兼容性。请在隔离的导出环境中使用源依赖记录。
本目录 `requirements.txt` 只列主机校准与配置依赖；工具不会隐式安装软件。
导出器的 `--help` 不依赖 Torch 或 Ultralytics。

29 份原始 YAML 逐字节保留在 `ptq_yamls/{x5,s}`：

| 目标 | march | 已发布制品对应配方 | 保留的实验配方 |
|---|---|---|---|
| X5 | bayes-e | n/s/m/l/x 共 5 份 NV12 | 无 |
| S100 | nash-e | n/s/m 共 3 份 NV12，l/x 共 2 份 lite | n/s/m 共 3 份 lite |
| S100P | nash-m | 同上 | 同上 |
| S600 | nash-p | 同上 | 同上 |

`compile.py` 读取对应模板，在外部目录生成 `config.yaml`，写入 ONNX、校准及工作目录的
绝对路径，不再假定模板中的历史相对文件名恰好等于导出结果。

<a id="export"></a>
## 导出

以下 Python 命令均从本 `conversion/` 目录执行。每次导出使用新的输出目录，已有路径会报错。

```bash
python export.py --target x5 --variant n \
  --weights /work/weights/yolo26n-depth.pt --output-dir /work/depth/export_x5_n
python export.py --target s100 --variant n \
  --weights /work/weights/yolo26n-depth.pt --output-dir /work/depth/export_s_n
python export.py --target s100 --variant l \
  --weights /work/weights/yolo26l-depth.pt --output-dir /work/depth/export_s_l
```

默认 opset 11，生成 `yolo26n-depth_op11_log.onnx` 或
`yolo26l-depth_op11_lite.onnx`、权重副本及 `export-report.json`。
输入尺寸固定为 768。可以修改 `--opset`，但只有源默认值有历史依据。
S 可通过 `--boundary` 显式导出其他实验边界；X5 没有 lite 源配方，因此拒绝该选项。

<a id="calibration"></a>
## 校准数据

两侧相同的 SUNRGBD ZIP 提取工具合并保留。请自行准备本地数据并遵守数据集条款，仓库不附带数据集。

```bash
python extract_sunrgbd_subset.py --archive /work/datasets/SUNRGBD.zip \
  --split train --count 100 --seed 20260725 --output /work/depth/train100
python prepare_calibration.py --target x5 --variant n \
  --images /work/depth/train100/images --output /work/depth/cal_x5 --count 100
python prepare_calibration.py --target s100 --variant n \
  --images /work/depth/train100/images --output /work/depth/cal_s_nv12 --count 100
python prepare_calibration.py --target s100 --variant l \
  --images /work/depth/train100/images --output /work/depth/cal_s_lite --count 100
```

X5 生成 RGB CHW uint8 `.bin`，由 Mapper 执行 `/255`；S 保留源流程，生成已 `/255` 的
RGB float32 NCHW `.npy`。S NV12 与 lite 分别采用 letterbox 和直接拉伸。
直接 RGB 校准与 NV12 色度采样后重建的 RGB 不保证逐字节相等。

每个数据目录旁生成同名 `.json` 和 `.md`，例如 `cal_s_nv12.json`。
可用 `--manifest`、`--report` 指定位置，但必须是校准目录之外的两个不同文件。
记录包含原图与张量 SHA-256、形状、类型和几何。默认种子 20260725，数量 100。
不可解码图像、零数量及已有输出均报错；失败后可能保留部分张量，请使用新目录重新准备。
不要混合两种 S 校准数据。

<a id="compile"></a>
## 编译

先仅检查准备阶段，不调用 OE。正式编译另用一个新目录，避免覆盖已有记录：

```bash
python compile.py --target s100 --variant n \
  --onnx /work/depth/export_s_n/yolo26n-depth_op11_log.onnx \
  --calibration-manifest /work/depth/cal_s_nv12.json \
  --output /work/depth/preflight_s_n --prepare-only
python compile.py --target s100 --variant n \
  --onnx /work/depth/export_s_n/yolo26n-depth_op11_log.onnx \
  --calibration-manifest /work/depth/cal_s_nv12.json \
  --output /work/depth/compile_s_n
python compile.py --target x5 --variant n \
  --onnx /work/depth/export_x5_n/yolo26n-depth_op11_log.onnx \
  --calibration-manifest /work/depth/cal_x5.json \
  --output /work/depth/compile_x5_n
```

S l/x 使用匹配的变体、lite ONNX 和 lite 校准清单。
`--target s100p` / `s600` 分别选择 nash-m / nash-p，这些 S 目标可共用同一种校准表示。
`--experimental-lite` 可选择保留的 S n/s/m lite 配方，但不改变已发布运行时方案。

编译前检查每个张量的摘要、形状、类型、值域及目录内容。
X5 调用 checker、makertbin 和 model-info；S 调用 `hb_compile`。
即使编译器返回零，没有预期制品也会失败。X5 还保留量化 ONNX、编译器 cosine、
延迟/FPS 与 DDR 估计。`mapper.py` 保留原直接 X5 流程，参数为
`--onnx`、`--variant`、`--calibration`、`--output`、`--size 768`、`--jobs`、
`--optimize-level`，不具备新增的校准清单校验。

<a id="validation"></a>
## 验证与结果解释

`--prepare-only` 只检查本地文件，不验证 ONNX 图、OE 兼容性、模型语义及精度。
真实编译后仍需核对输入约定、输出布局/类型及原始/已校准边界，再进行浮点对照和板测。
请同时保存输入、权重摘要、校准清单、编译器版本与日志。

```bash
hb_model_info /work/depth/compile_x5_n/artifacts/yolo26n_depth_bayese_768x768_nv12.bin
hrt_model_exec model_info --model_file /work/depth/compile_x5_n/artifacts/yolo26n_depth_bayese_768x768_nv12.bin
```

第二条需要匹配的目标环境。编译器估计、单图 cosine、数据集精度与实测延迟是不同指标。
离线深度评估位于 `../evaluator/`。重新编译的模型不会继承已发布制品的 SHA-256 或历史性能证据。

<a id="artifacts"></a>
## 输出制品

- `config.yaml`：对应源配方及本次实际输入路径。
- `preparation.json`：ONNX、校准清单摘要、身份信息和初始 not-run 状态。
- `reports/`：真实命令标准输出与错误输出；失败日志也保留。
- `working/`：编译中间文件。
- `artifacts/`：最终 BIN/HBM；X5 另复制量化 ONNX。
- `compile-report.json`：仅在必需制品和检查全部完成后生成。

失败时不会生成成功的最终报告。保留的部分工作目录用于诊断，下一次请换新目录。

<a id="known-gaps"></a>
## 已知缺口与源证据

本轮未运行 Torch 导出、两套 OE、已下载模型或 SUNRGBD 评估。
源记录缺少权重摘要、S 镜像/编译器精确版本以及 S HBM 发布方摘要。
导出器依赖预期的 Depth 模块，找不到唯一一个对应头时显式报错。

源 S 实验记录显示 NV12 l/x 存在截断，lite n/s/m 的 cosine 较低，所测 int16 调整也未改善。
这些记录用于解释混合发布方案，不是本轮验证结果。
源文档“全部达到 0.999”的说法与表中 0.9984 冲突，详见
[源审计](../../../../docs/releases/unified-migration/2026-09-26-b8-yolo26-depth-source-review.md)。
不得通过删除表格或降低门槛掩盖矛盾。

源可选调优记录中的具体对照为：NV12 l/x 约 17% 像素饱和，cosine 为
0.9938/0.9944；S100 lite n/s/m 为 0.9903/0.9854/0.9529。
max/0.9999 校准将 lite n 提高到 0.9975，仍未达到原门槛。
所述 S100 尾部卷积 int16 测试为 0.985449；全节点 int16 将模型约 13 MB 增至 25 MB、
约 5.8 ms 增至 23.0 ms，cosine 反而降至 0.982804。
这些是复现证据不完整的源观察。新实验应固定 ONNX、校准集、图像、板卡与前处理，
每次只改一个选项，分别记录原始输出与还原深度的对照。

代码遵循仓库根许可证；Ultralytics 权重和 SUNRGBD 数据保留各自上游条款。
本样例不额外授予权重导出或数据使用许可。
