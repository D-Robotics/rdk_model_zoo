[English](README.md) | 简体中文

# LiteAvatar 转换

[English](README.md) · [示例](../README_cn.md) · [Python 运行时](../runtime/python/README_cn.md)

本目录保存将上游 [HumanAIGC/lite-avatar](https://github.com/HumanAIGC/lite-avatar)
（音频驱动数字人）流水线转换为示例运行时所需的按目标划分的四件 HBM
产物的配方：ONNX 导出、两个无损图变换、校准数据生成（含运行时一致的
解码器校准集）、四个 `hb_compile` 配置以及余弦/上板验证脚本。

四个产物（全节点按配置 INT16 量化；march 按目标 —— S100 `nash-e`、
S100P `nash-m`、S600 `nash-p`）：

| 阶段 | 模型 | 输入 | 输出 | 配置 |
| --- | --- | --- | --- | --- |
| 音频编码器 | `encoder_avatar.onnx`（paraformer 编码器，50 层隐状态） | `feats` [1,400,560] | `hidden_states` [1,400,50,512] | `encoder_avatar_config.yaml` |
| audio2mouth | `model_1_folded.onnx` | `input_au` [1,30,50,512] | `output` [1,30,32]（+`feat`） | `audio2mouth_int16_config.yaml` |
| 人脸编码器 | `net_encode.onnx` | `image` [1,3,384,384] | `f0..f3` 特征金字塔 | `net_encode_int16_config.yaml` |
| 人脸解码器 | `net_decode_deconv_inorm.onnx` | `f0;f1;f2;f3;mouth_param` | `mouth_image` [1,3,384,384] | `net_decode_deconv_inorm_fullrange_config.yaml` |

解码器**不是**纯 INT16 BPU 模型：其归一化子图经配置中 21 节点的 `run_on_cpu`
列表保持 FP32（编译器节点报告显示这些节点为混合落位——部分融合、部分以
FP32 落在 BPU，`Pow` 方差算子在 CPU 上），主要卷积以 INT16 跑 BPU——个别
逐点卷积的编译精度随 march 不同；原因见
[ONNX 导出与解码器图变换](#export)。

<a id="source-model"></a>
## 源模型与许可

- 代码：`github.com/HumanAIGC/lite-avatar`，`main` 分支 @
  `5b7ec850945e03d56fb290b05fb68440c359fa86`
  （MIT——上游完整许可证文本以
  [`LICENSE-LiteAvatar`](../LICENSE-LiteAvatar) 随样例分发）。上游检出提供
  `lite_avatar.py`、`audio2mouth_cpu.py`、
  `extract_paraformer_feature.py`、内嵌的 `funasr_local/` FunASR 前端（MIT，见
  `LICENSE-FunASR`）、`data/sample_data.zip`（sha256
  `d1c77b2df48b7fd7f19e6eef52bc5107e3131dd9777162faf2576f9451f69d7b`）以及
  `weights/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/`
  （含 `am.mvn` 和 `example/asr_example.wav`，Apache-2.0，经 ModelScope
  paraformer 包分发）。
- 权重：上游仓库的 `download_model.sh` 从上游 ModelScope 仓库
  [HumanAIGC-Engineering/LiteAvatarGallery](https://modelscope.cn/models/HumanAIGC-Engineering/LiteAvatarGallery)
  拉取 `model_1.onnx`（audio2mouth）和 `model.pb`（paraformer）。
  **Gallery 模型卡未声明许可证**，这些产物仅作为用户自备的转换输入。
- `net_encode.pt` / `net_decode.pt`（TorchScript 人脸生成器）随
  `data/sample_data.zip` 发布，是必需的导出输入（见下文）。
- 模型输入音频：`asr_example.wav` 随 ModelScope paraformer 包发布
  （Apache-2.0，经 FunASR/ModelScope 分发）。

<a id="directory"></a>
## 目录结构

下列 Python 脚本均在上游 `lite-avatar` 检出根目录运行，且本目录已作为
`conversion/` 拷入其中：

```text
lite-avatar/                  # 上游检出 @5b7ec850
├── conversion/               # 本目录的拷贝（scripts/、配置）
│   ├── *_config.yaml         # 四个 hb_compile 配置
│   ├── scripts/              # 导出 / 变换 / 校准 / 验证脚本
│   └── calib_data/           # 校准脚本的生成物
├── data/preload/             # 解压后的 sample_data.zip（bg_video.mp4、ref_frames/、*.pt）
├── weights/                  # download_model.sh 输出（model_1.onnx、model.pb、am.mvn 等）
└── funasr_local/             # 上游内嵌前端（导出/校准脚本使用）
```

本目录内容：

```text
conversion/
├── README.md                            # 英文说明
├── README_cn.md                         # 中文说明
├── LICENSE-FunASR                       # 内嵌 FunASR 前端的 MIT 许可证
├── encoder_avatar_config.yaml           # hb_compile 配置（音频编码器）
├── audio2mouth_int16_config.yaml        # hb_compile 配置（audio2mouth）
├── net_encode_int16_config.yaml         # hb_compile 配置（人脸编码器）
├── net_decode_deconv_inorm_fullrange_config.yaml  # hb_compile 配置（人脸解码器）
└── scripts/
    ├── probe_paraformer_encoder.py      # 导出 encoder_avatar.onnx
    ├── fold_audio2mouth.py              # 常量输入折叠
    ├── export_facegen.py                # 导出 net_encode/net_decode ONNX
    ├── replace_resize_deconv.py         # Resize -> ConvTranspose
    ├── decompose_inorm.py               # InstanceNorm 预分解
    ├── gen_calib_encoder.py             # 编码器校准数据
    ├── gen_calib_audio2mouth.py         # audio2mouth 校准数据
    ├── gen_calib_facegen.py             # 人脸校准数据
    ├── gen_real_calib.py                # 板端运行时一致解码器校准
    ├── cosine_eval_ptq.py               # 量化 ONNX 余弦门槛
    ├── board_infer.py                   # 板端 HBM 推理转储
    └── board_compare.py                 # 主机端上板余弦门槛
```

一次性准备工作区：

```bash
git clone https://github.com/HumanAIGC/lite-avatar
cd lite-avatar && git checkout 5b7ec850
mkdir -p conversion && cp -r <model-zoo>/samples/vision/lite_avatar/conversion/. conversion/
bash download_model.sh                    # model_1.onnx + model.pb -> weights/
unzip data/sample_data.zip -d data/       # sha256 d1c77b2d...；得到 data/preload/
```

<a id="toolchain-targets"></a>
## 工具链与目标

使用两个互不混用的环境：

1. **导出/校准主机** —— CPU Python，按已验证的包集合安装（Torch 2.3.1+cpu /
   torchaudio 2.3.1+cpu —— 上游 TorchScript 权重与 `funasr_local` 固定的栈 ——
   以及 onnx、**onnxsim**（常量输入折叠与已验证的简化依赖它）、onnxruntime
   （导出验证与余弦检查）、`funasr_local` 所需的 `typeguard==2.13.3`、
   soundfile、librosa、opencv-python、numpy 1.26.4、scipy）。无需板端 SDK：

   ```bash
   python3 -m venv .venv-export && . .venv-export/bin/activate
   pip install torch==2.3.1+cpu torchaudio==2.3.1+cpu \
       --index-url https://download.pytorch.org/whl/cpu
   pip install onnx onnxsim onnxruntime typeguard==2.13.3 \
       soundfile librosa opencv-python numpy==1.26.4 scipy
   ```

2. **OE 工具链** —— Docker 镜像
   `registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0`
   （hbdk 4.7.5 / hmct 2.6.5 / `hb_compile` 3.5.3），用于四次 `hb_compile`；
   镜像自带 `nash-e`/`nash-m`/`nash-p` 三个 march。请**串行**执行（同一时刻
   只编译一个）：工具链按并发实例授权。解码器配置已设 `jobs: 8`，其余三个
   为 `jobs: 32`。

配方按目标各构建一套 —— S100（`nash-e`）、S100P（`nash-m`）、S600
（`nash-p`）—— 图、变换与校准数据完全相同；每个目标的配置副本改
`march`、输出前缀与工作目录（见[编译](#compile)）。X5 无构建。

<a id="export"></a>
## ONNX 导出与解码器图变换

在检出根目录运行（下列默认路径均相对根目录）：

```bash
# 1. 导出堆叠了全部 50 层块隐状态的 paraformer 编码器
#    （供 audio2mouth 使用；需要 weights/.../model.pb + funasr_local）
python conversion/scripts/probe_paraformer_encoder.py
#    -> conversion/encoder_avatar.onnx

# 2. 折叠 audio2mouth 的常量输入（input_ph 全零、input_sp=2、w=1.0）。
#    --dst 默认写到检出根目录的 ./model_1_folded.onnx；编译配置与人脸校准
#    脚本期望它位于 conversion/ 下，因此显式传入 --dst：
python conversion/scripts/fold_audio2mouth.py \
    --src weights/model_1.onnx --dst conversion/model_1_folded.onnx

# 3. 人脸生成器：data/preload（sample_data.zip）中的 TorchScript
#    net_encode.pt/net_decode.pt -> ONNX
python conversion/scripts/export_facegen.py --data_dir data/preload --out conversion
#    -> conversion/net_encode.onnx, conversion/net_decode.onnx
```

`net_decode.onnx` 在 PTQ 前还需要两个无损、顺序固定的变换；两个脚本都会对照
原图验证等价性：

```bash
# 4. Resize -> ConvTranspose（3 个双线性 x2 上采样）。否则 hb_compile 会让
#    Resize 回落 CPU，约 157 ms/帧（未变换解码器的实测均值）。
python conversion/scripts/replace_resize_deconv.py
#    -> conversion/net_decode_deconv.onnx

# 5. 把 3 个无仿射 InstanceNorm 分解为命名的普通算子，
#    使 run_on_cpu 能把方差路径保持在 FP32
python conversion/scripts/decompose_inorm.py --verify
#    -> conversion/net_decode_deconv_inorm.onnx（余弦 1.000，最大|差| ~1e-6）
```

第 5 步存在的原因（根因记录在配置头部）：`hb_compile` 在优化阶段分解
InstanceNormalization 并把方差路径量化为 BPU INT16；复合节点上的 `run_on_cpu`
标志在该分解中不会保留。预先分解为
命名的普通算子后，配置的 `run_on_cpu` 列表可以把全部 21 个 norm 节点（每个
解码器块的 mean/centered/sq/var/vareps/invstd/apply）挡在 BPU INT16 之外——
节点报告显示 `Pow` 方差算子以 FP32 在 CPU 执行，列表中其余节点的落位由编译器
混合（部分融合、部分以 FP32 落在 BPU——S600 上 `Sub` 即为 BPU FP32）。
用 `Pow(x,-0.5)` 替代 `Reciprocal(Sqrt())` 以避开没有
CPU 实现的 HzRsqrt 融合。卷积主体保持整数量化跑 BPU——主要卷积为 si16，
个别逐点卷积的编译精度随 march 不同（S100/S100P 上为 si32，S600 上以
FP32 落在 BPU）。

<a id="calibration"></a>
## 校准数据

在检出根目录运行；脚本会加载 `funasr_local` 与 `weights/` 下的 paraformer 包：

```bash
# 6. 编码器校准：16 kHz wav 的后 CMVN 特征 [1,400,560]。
#    默认来源：funasr_local/.../test_wavs（上游检出自带）以及 weights/ 下的
#    paraformer example 目录。
python conversion/scripts/gen_calib_encoder.py
#    -> conversion/calib_data/encoder_avatar

# 7. audio2mouth 校准：真实 input_au 窗口 [1,30,50,512]。wav 来源同上；
#    可用 --paraformer_example_wav <目录> 增补更多语音。
python conversion/scripts/gen_calib_audio2mouth.py
#    -> conversion/calib_data/audio2mouth

# 8. 人脸校准：net_encode 图像 [1,3,384,384] 与解码器五输入集合
#    （mouth 参数来自 FP32 折叠 audio2mouth ONNX 在相同窗口上的输出）
python conversion/scripts/gen_calib_facegen.py
#    -> conversion/calib_data/facegen_encode, conversion/calib_data/facegen_decode
```

**第 9 步 —— 运行时一致的解码器校准（已验证配方的一部分）。** 发布的 v3
解码器使用以运行时一致参数重新生成的校准集编译：真实 Butterworth 平滑后的
mouth 参数与 `net_encode.hbm` 参考特征，并把动画循环铺到参考帧上。
`gen_real_calib.py` 内嵌已验证的前端/参数循环代码，在具备三个已编译 HBM 与
上游检出的**板上**运行，然后把目录拷回 `conversion/calib_data/facegen_decode`
再编译解码器。v3 配方包含此步骤；未验证仅 FP32 校准能复现发布产物：

```bash
python3 conversion/scripts/gen_real_calib.py \
    --hbm-dir /path/to/hbm --data /path/to/preload \
    --mvn /path/to/am.mvn --wav /path/to/asr_example.wav \
    --out conversion/calib_data/facegen_decode \
    --upstream /path/to/lite-avatar --n-ref 30
```

<a id="compile"></a>
## 编译

在 OE 工具链容器内、**`conversion/` 目录下**串行执行（各配置的
`onnx_model`、`cal_data_dir`、`working_dir` 均相对该目录的 `./` 路径）：

```bash
cd conversion
hb_compile -c encoder_avatar_config.yaml
hb_compile -c audio2mouth_int16_config.yaml
hb_compile -c net_encode_int16_config.yaml
hb_compile -c net_decode_deconv_inorm_fullrange_config.yaml
```

配置默认 `march: nash-e`（S100）。每增加一个目标，把四个配置各复制一份
（如 `net_encode_int16_s600_config.yaml`），在副本中改四个设置：
`model_parameters` 下把 `march` 改为 `nash-m`（S100P）或 `nash-p`
（S600），并为副本设置自己的 `working_dir` 与
`output_model_file_prefix` 使各目标产物互不覆盖；`compiler_parameters`
下把 `jobs` 改为 `8`：

```yaml
# net_encode_int16_s600_config.yaml —— net_encode_int16_config.yaml 的副本
model_parameters:
  march: nash-p                   # nash-e -> nash-p（S600）
  working_dir: ./net_encode_int16_s600_output
  output_model_file_prefix: net_encode_int16_s600
compiler_parameters:
  jobs: 8
```

每个配置输出 `./<prefix>_output/<prefix>.hbm` 和 `<prefix>_ptq_model.onnx`
（量化 ONNX，供余弦检查）。解码器配置保持 `max_percentile: 1.0`、
`jobs: 8` 与 21 节点 `run_on_cpu` 列表原样；不要重排或删减该列表。

配置中记录的多输入要点：`input_name`/`input_shape`/`cal_data_dir`/
`cal_data_type` 用分号分隔（每个输入一项，无空格），featuremap 输入用
`norm_type: no_preprocess`，`mouth_param` 形状 [1,32] 可直接作为 featuremap，
无需 reshape 成 [1,32,1,1]。

<a id="validation"></a>
## 验证

余弦（量化 ONNX 对比 FP32 ONNX；门槛 0.95 只作用于**主输出**，
即运行时真正消费的输出——用 `--stage` 按本样例四个阶段的契约声明，或对其他
图用 `--primary` 显式声明；audio2mouth 的辅助输出 `feat` 照常测量并完整报告
统计值，但不参与门槛）。请在 **OE 3.7 工具链容器**内运行——量化图所需的
horizon 自定义算子运行时只有该环境提供。`--n` 为必填且不得超过现有样本数
（参考校准数据集：audio_encoder 3、audio2mouth 17、facegen_encode 30、
facegen_decoder 30——示例即采用这些数量；重新生成的数据集数量可以不同）：

```bash
python conversion/scripts/cosine_eval_ptq.py \
    --fp32 conversion/net_encode.onnx \
    --ptq  conversion/net_encode_int16_output/net_encode_int16_ptq_model.onnx \
    --calib conversion/calib_data/facegen_encode \
    --stage facegen_encoder --n 30 --gate 0.95
# 解码器的五子目录校准布局加 --multi
```

上板验证（真实 BPU 对比 FP32 ONNX）—— `board_infer.py` 在**板上**运行，
`board_compare.py` 在主机运行：

```bash
# 板上（hbm_runtime + numpy）
python3 conversion/scripts/board_infer.py \
    --hbm /path/on/board/net_decode_deconv_inorm_fullrange.hbm \
    --calib /path/on/board/calib_data/facegen_decode --multi --n 30 \
    --out /path/on/board/board_out/net_decode
# 取回 board_out/ 后在主机上
python conversion/scripts/board_compare.py \
    --fp32 conversion/net_decode_deconv_inorm.onnx \
    --board_out conversion/board_out/net_decode \
    --calib conversion/calib_data/facegen_decode --multi \
    --stage facegen_decoder --n 30 --gate 0.95
```

与发布解码器产物匹配的上板实测（`net_decode_deconv_inorm_fullrange.hbm`，
sha256 `27e46cf9d6514f587d54c16cfa813583d89e59c9d5b5d4138efdf51e4d20df3c`；
S100，UCP/DNN 3.13.6，OE 3.7.0，nash-e，2026-10-12；`asr_example.wav` 166 帧，
与相同输入上的 FP32 ONNX 对比）：

| 检查 | 结果 |
| --- | --- |
| 解码器输出对比 FP32 参考 | 平均 PSNR 48.94 dB，平均 MAE 0.0052（解码器归一化 float 输出尺度上的转换级张量指标——与 [evaluator/evaluate_frames.py](../evaluator/evaluate_frames.py) 的 uint8 帧级 PSNR/MAE 不是一个尺度），无 NaN/Inf 像素 |
| 解码器时延（含 run_on_cpu 归一化路径） | 平均 24.69 ms/帧 |

其余三个产物 —— 以及四个阶段的 S100P/S600 构建 —— 用相同的余弦/上板脚本
对照各自校准集验证；示例的配对上板验证会在同一目标上把每套发布四件套的
每个阶段与原流水线对比（见示例 README）。

<a id="artifacts"></a>
## 产物

每个配置输出 `./<prefix>_output/<prefix>.hbm`（按目标复制的配置输出
`./<prefix>_<目标>_output/<prefix>_<目标>.hbm`）。把四个 HBM 按发布名拷入示例
的 `model/<目标>/` 目录：

| 阶段（输出目录） | s100 | s100p | s600 |
| --- | --- | --- | --- |
| 音频编码器（`encoder_avatar_output/`） | `lite_avatar_encoder_400x560_s100.hbm` | `lite_avatar_encoder_400x560_s100p.hbm` | `lite_avatar_encoder_400x560_s600.hbm` |
| audio2mouth（`audio2mouth_int16_output/`） | `lite_avatar_audio2mouth_30x50x512_s100.hbm` | `lite_avatar_audio2mouth_30x50x512_s100p.hbm` | `lite_avatar_audio2mouth_30x50x512_s600.hbm` |
| 人脸编码器（`net_encode_int16_output/`） | `lite_avatar_net_encode_384x384_s100.hbm` | `lite_avatar_net_encode_384x384_s100p.hbm` | `lite_avatar_net_encode_384x384_s600.hbm` |
| 人脸解码器（`net_decode_deconv_inorm_fullrange_output/`） | `lite_avatar_net_decode_inorm_fullrange_384x384_s100.hbm` | `lite_avatar_net_decode_inorm_fullrange_384x384_s100p.hbm` | `lite_avatar_net_decode_inorm_fullrange_384x384_s600.hbm` |

然后按 [模型准备](../model/README_cn.md) 路线 B 运行：自编译集合把自身
SHA-256 记入本地重建清单，各发布四件套的摘要以
`docs/release/s/models.yaml` 为准。

<a id="known-gaps"></a>
## 已知缺口

- ModelScope `LiteAvatarGallery` 权重（`model_1.onnx`、`model.pb`）未声明
  许可证；它们是用户自备的转换输入。
- 未验证仅 FP32 的解码器校准（第 8 步而不做第 9 步）能复现发布的 v3 产物；
  运行时一致的解码器校准是配方的一部分。
- 未变换的解码器图会把 `Resize` 编到 CPU 上，约 157 ms/帧（实测均值）；
  Resize→ConvTranspose 变换正是为避开该路径而存在。
- X5 不是本示例的目标，不存在也不声明 X5 构建。OE 镜像按并发实例授权
  （每个目标的四次编译须串行）。
