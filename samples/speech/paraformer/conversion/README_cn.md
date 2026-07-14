[English](./README.md)

# Paraformer 模型量化与转换

本目录包含完整的 Paraformer 固定 shape INT16 量化与部署流程。所有命令均在本目录执行，生成的 ONNX、校准数据、日志和 HBM 为本地构建产物，不应提交到仓库。

> **模型**：[iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404](https://modelscope.cn/models/iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404)
> **目标平台**：地瓜机器人 **S100 (nash-e)** BPU
> **工具链**：`ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0` (HMCT / hbdk4 4.7.5)
> **量化方案**：全 INT16 (Encoder / Predictor / Decoder) + CPU CIF
> **板端 300 条 AISHELL dev CER**：**3.13%**（vs FP32 baseline 5.20%，实际优于 HMCT 主机仿真 5.02%）
> **板端端到端延迟**：Python `hbm_runtime` **45.6 ms/utt** · C++ UCP **40.8 ms/utt** · RTF ≈ **0.007**

---

## 1. 概述

Paraformer-large-contextual 是一款非自回归 ASR 模型，包含 4 个逻辑模块：Encoder、Predictor、CIF (Continuous Integrate-and-Fire)、Decoder。本文将说明如何将该模型部署到地瓜机器人 S100 BPU 平台。

**关键要点：**

| 模块 | 上 BPU？ | 原因 |
|---|---|---|
| Encoder | ✅ | 纯 Conv+MatMul+LayerNorm，标准 transformer |
| Predictor | ✅ | 简单 Conv+Linear+Sigmoid（12 节点） |
| **CIF** | ❌ CPU | 含 `GatherND(dynamic_index)`、boolean mask、`num_fires` 数据依赖 shape，hbdk4 无法编译 |
| Decoder | ✅ | 15 层 transformer + Contextual bias attention |

CIF 在 CPU 上仅需 1-2 毫秒，不是瓶颈。

**输入约束（固定 shape）：**

| 张量 | Shape | 说明 |
|---|---|---|
| `speech` | `[1, 400, 560]` fp32 | fbank+LFR (m=7, n=6)，400 帧 ≈ 24 秒音频上限 |
| `bias_embed` | `[1, 1, 512]` fp32 | Contextual 热词嵌入（0 = 无热词） |
| `max_label_len` | 100 | CIF 最多产出 100 个 token（monkey-patch 强制固定） |

---

## 2. 环境准备

### 2.1 拉取工具链 Docker（OpenExplorer）

```bash
docker pull ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0
```

启动容器（推荐）：

```bash
docker run --rm -it \
    -u $(id -u):$(id -g) \
    --entrypoint /bin/bash \
    -v /path/to/workspace:/workspace \
    -w /workspace \
    ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0
```

### 2.2 下载 Paraformer 模型

```bash
pip install modelscope funasr
python -c "
from funasr import AutoModel
AutoModel(model='iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404')
"
```

模型会被下载到 `~/.cache/modelscope/hub/models/`。将其复制到工作目录，例如 `./models/paraformer/`。

### 2.3 准备校准数据集

**关键：校准数据必须与部署时的输入分布一致。** 使用真实语音（如 AISHELL-1 dev 子集，50 条即可），提取 fbank+LFR 特征后保存：

```python
import numpy as np, torch, glob, os
from pathlib import Path
import soundfile as sf
from funasr import AutoModel

model = AutoModel(model="./models/paraformer", disable_update=True)
frontend = model.kwargs["frontend"]

CALIB_DIR = "./calib_data/speech"
os.makedirs(CALIB_DIR, exist_ok=True)

for i, wav_path in enumerate(sorted(glob.glob("./aishell_dev/**/*.wav", recursive=True))[:50]):
    wav_np, sr = sf.read(wav_path); assert sr == 16000
    wav_t = torch.from_numpy(wav_np).float().unsqueeze(0)
    feats, _ = frontend(wav_t, torch.tensor([wav_t.shape[1]]))
    feats_np = feats[0].detach().cpu().numpy()  # [T, 560]
    if feats_np.shape[0] >= 400:
        feats_400 = feats_np[:400]
    else:
        feats_400 = np.pad(feats_np, ((0, 400-feats_np.shape[0]), (0, 0)))
    feats_400 = feats_400[None, ...].astype(np.float32)
    np.save(f"{CALIB_DIR}/calib_{i:03d}.npy", feats_400)
```

---

## 3. 模型切分策略

```
speech ──► [Encoder hbm]           ──► enc_out
                                       │
                       ┌───────────────┼───────────────┐
                       ▼               ▼               ▼
              [Predictor hbm]     ┌─ CIF (CPU) ─┐   [Decoder hbm]
                       │          │             │        ▲
                    alphas ───────┤             │────────┤ pre_acoustic_embeds
                       │          │             │        │ token_num
                       └──────────┴─────────────┘        │
                                                    bias_embed
                                                         │
                                                         ▼
                                                       logits [1,100,8404]
```

三个 hbm 全部采用 **INT16 全量化 + `max` calibration**。CIF 用 numpy 在 CPU 上运行，接收 predictor 输出 (`alphas`, `Concat_5_output_0`) 并生成 decoder 所需的 `pre_acoustic_embeds` 和 `token_num`。

---

## 4. FunASR 导出改造

原始 FunASR 的 CIF 实现含有数据依赖的 shape，hbdk4 无法编译。需要对 CIF 进行 monkey-patch：

### 4.1 强制静态 max_label_len

原代码 `max_label_len = floor(alphas.sum(-1)).max()` 让下游张量 shape 数据依赖。改写：

```python
# reexport_fixed_shape.py（核心 patch）

MAX_LABEL_LEN = 100  # 固定 CIF 最大输出 token 数

def patched_cif_v1_export(hidden, alphas, threshold: float):
    """替换 FunASR 原始 cif_v1_export，去除动态 shape + 兼容 hbdk4。"""
    device = hidden.device; dtype = hidden.dtype
    batch_size, len_time, hidden_size = hidden.size()

    frames = torch.zeros(batch_size, len_time, hidden_size, dtype=dtype, device=device)
    fires = torch.zeros(batch_size, len_time, dtype=dtype, device=device)

    prefix_sum = torch.cumsum(alphas, dim=1, dtype=torch.float64).to(torch.float32)
    prefix_sum_floor = torch.floor(prefix_sum)
    disl_prefix_sum = torch.roll(prefix_sum, 1, dims=1)
    disl_prefix_sum_floor = torch.floor(disl_prefix_sum)
    disl_prefix_sum_floor[:, 0] = 0
    dislocation_diff = prefix_sum_floor - disl_prefix_sum_floor
    fire_idxs = dislocation_diff > 0
    fires = fires.masked_fill(fire_idxs, 1.0)
    fires = fires + prefix_sum - prefix_sum_floor

    # 关键：所有 .unsqueeze(-1) 改成 .reshape(...)，绕开 hbdk4 type_inf bug
    alphas_r = alphas.reshape(batch_size, len_time, 1)
    prefix_sum_hidden = torch.cumsum(alphas_r.repeat((1, 1, hidden_size)) * hidden, dim=1)
    frames = prefix_sum_hidden[fire_idxs]
    shift_frames = torch.roll(frames, 1, dims=0)
    batch_len = fire_idxs.sum(1)
    batch_idxs = torch.cumsum(batch_len, dim=0)
    shift_batch_idxs = torch.roll(batch_idxs, 1, dims=0)
    shift_batch_idxs[0] = 0
    shift_frames[shift_batch_idxs] = 0

    remains = fires - torch.floor(fires)
    remains_sel = remains[fire_idxs]
    hidden_sel = hidden[fire_idxs]
    remains_sel_2d = remains_sel.reshape(-1, 1)              # 关键：不用 unsqueeze
    remain_frames = remains_sel_2d.repeat((1, hidden_size)) * hidden_sel
    shift_remain_frames = torch.roll(remain_frames, 1, dims=0)
    shift_remain_frames[shift_batch_idxs] = 0
    frames = frames - shift_frames + shift_remain_frames - remain_frames

    max_label_len: int = MAX_LABEL_LEN
    frame_fires = torch.zeros(batch_size, max_label_len, hidden_size, dtype=dtype, device=device)
    indices = torch.arange(max_label_len, device=device).expand(batch_size, -1)
    batch_len_clamped = torch.clamp(batch_len, max=max_label_len)
    batch_len_2d = batch_len_clamped.reshape(batch_size, 1)  # 关键：不用 unsqueeze
    frame_fires_idxs = indices < batch_len_2d
    num_slots = int(frame_fires_idxs.sum().item())
    frame_fires[frame_fires_idxs] = frames[:num_slots]
    return frame_fires, fires


import funasr.models.paraformer.cif_predictor as cifp
cifp.cif_v1_export = torch.jit.script(patched_cif_v1_export)
```

### 4.2 强制固定 pre_token_length

```python
import funasr.models.contextual_paraformer.export_meta as cp_em

def patched_export_backbone_forward(self, speech, speech_lengths, bias_embed):
    batch = {"speech": speech, "speech_lengths": speech_lengths}
    enc, enc_len = self.encoder(**batch)
    mask = self.make_pad_mask(enc_len)[:, None, :]
    pre_acoustic_embeds, _pre_token_length, _, _ = self.predictor(enc, mask)
    B = pre_acoustic_embeds.shape[0]
    # 强制 pre_token_length = MAX_LABEL_LEN，让 decoder mask 尺寸静态
    pre_token_length_fixed = torch.full((B,), MAX_LABEL_LEN, dtype=torch.int32,
                                        device=pre_acoustic_embeds.device)
    decoder_out, _ = self.decoder(enc, enc_len, pre_acoustic_embeds,
                                  pre_token_length_fixed, bias_embed)
    decoder_out = torch.log_softmax(decoder_out, dim=-1)
    return decoder_out, pre_token_length_fixed

cp_em.export_backbone_forward = patched_export_backbone_forward
```

### 4.3 触发 ONNX 导出

```python
from funasr import AutoModel
m = AutoModel(model="./models/paraformer", disable_update=True)
m.export(quantize=False, opset_version=15)
# 产出 model.onnx + model_eb.onnx
```

**为什么 Unsqueeze 要改 Reshape？**
hbdk4 4.7.5 在 CIF 分支的 `remains[fire_idxs].unsqueeze(-1)` 上会触发 `type_inf` 异常。将所有 `.unsqueeze(...)` 改成等价的 `.reshape(...)` 后，输出 ONNX 用的是 Reshape 算子，hbdk4 可以正常导出。

---

## 5. ONNX 图手术流水线

从 FunASR 原始 ONNX 到 hbdk4-ready 4 输入 decoder-only，共 **7 步图手术**：

```
model.onnx (FunASR 导出)                         ← FunASR AutoModel.export()
        │
        ▼
extract_decoder_predictor.py                     ← 抽 /decoder/* 子图
        │  · 非 /decoder/ 全部作为 boundary
        │  · 5 输入 → decoder_only.onnx
        ▼
convert_gather_int64_to_int32.py                 ← HMCT 崩在 INT64 Gather → 转 INT32
        │
        ▼
topsort.py                                       ← onnx.checker 要求拓扑序
        │
        ▼
fold_range_ops.py                                ← hbdk4 拒绝 Range → 展开为 Constant
        │
        ▼
shape_freeze.py (onnxsim)                        ← 固定 shape 传播 + 常量折叠
        │  2903 → 1214 nodes
        ▼
axis_normalize + fold /predictor/Gather → Const(1)
        │  · 消除 dynamic-repeats 的 Tile
        │  · negative axis → positive
        ▼
decoder_only_final.onnx (1137 nodes) ✅ hbdk4 export 通过
```

同时需要抽出 `encoder_only.onnx` 和 `predictor_only.onnx`（步骤类似，边界不同）。

**常见问题一览：**

| 问题 | 症状 | 解决 |
|---|---|---|
| INT64 Gather | HMCT `adjust_multi_output_use_quant_info_pass_fail` | 145 个 Constant INT64→INT32 |
| Range 算子 | hbdk4: `Operator Range should be optimized` | ORT 探针取值 → Constant |
| Unsqueeze type_inf | hbdk4 type_inf 异常（CIF `remains.unsqueeze(-1)`） | 改写为 `.reshape(-1,1)`（见 §4.1） |
| GatherND 动态索引 | `size of last dimension of index cannot be dynamic` | CIF 整体切到 CPU；decoder 用 `pre_acoustic_embeds` 作 graph input |
| Tile 动态 repeats | `cannot create ArrayAttr from OpResult` | 折 `/predictor/Gather_output_0` (batch=1) 为 Constant |
| Split axis=-1 | hbir slice 推断 shape 错误 | 所有 negative axis 归一化到 positive |

---

## 6. 量化编译（三个 hbm）

### 6.1 统一编译参数

三个 hbm 都用同一套 `compiler_parameters` + `quant_config`：

```yaml
compiler_parameters:
  optimize_level: O2
  compile_mode: latency
  core_num: 1              # S100 单 BPU 核
  jobs: 32                 # 编译并行度，按机器 CPU 核数调整
  cache_mode: disable

# 通用 quant_config 片段
quant_config:
  model_config:
    all_node_type: int16
    activation:
      calibration_type: max
```

### 6.2 Encoder

**YAML (`encoder_int16.yaml`)：**

```yaml
model_parameters:
  onnx_model: ./encoder_nomask.onnx
  march: nash-e
  output_model_file_prefix: paraformer_encoder_int16
  working_dir: ./encoder_int16_output

input_parameters:
  input_name: speech
  input_shape: 1x400x560
  input_type_rt: featuremap
  input_type_train: featuremap
  input_layout_train: NCHW
  separate_batch: false

calibration_parameters:
  cal_data_dir: ./calib_data/speech        # 50 条真实 fbank+LFR 特征
  calibration_type: max
  quant_config:
    model_config:
      all_node_type: int16
      activation:
        calibration_type: max

compiler_parameters:
  optimize_level: O2
  compile_mode: latency
  core_num: 1
  jobs: 32
```

**编译：**

```bash
hb_compile -c configs/encoder_int16.yaml
```

**产物指标：**

| 项 | 值 |
|---|---|
| hbm 大小 | 211.5 MB |
| 静态延迟 | 32.52 ms (FPS 30.75) |
| 静态内存 | 211 MB |
| 动态内存 | 3.6 MB |
| DDR 占用 | 499 MB |
| 编译时间 | ~55 min (jobs=32) |

### 6.3 Predictor

Predictor 的校准数据是 encoder 的 **FP32 输出**（不是随机数据）。先在 FP32 pipeline 上跑 50 条真实音频，把 encoder 输出存到 `real_calib/encoder_after_norm_Add_1_output_0/`。

**YAML：**

```yaml
model_parameters:
  onnx_model: ./predictor_only.onnx
  march: nash-e
  output_model_file_prefix: predictor_int16
  working_dir: ./predictor_int16_output

input_parameters:
  input_name: /encoder/after_norm/Add_1_output_0
  input_shape: 1x400x512
  input_type_rt: featuremap
  input_type_train: featuremap
  input_layout_train: NCHW
  separate_batch: false

calibration_parameters:
  cal_data_dir: ./real_calib/encoder_after_norm_Add_1_output_0
  calibration_type: max
  quant_config:
    model_config:
      all_node_type: int16
      activation:
        calibration_type: max

compiler_parameters:
  optimize_level: O2
  compile_mode: latency
  core_num: 1
  jobs: 32
```

**产物指标：**

| 项 | 值 |
|---|---|
| hbm 大小 | ~4 MB |
| 静态延迟 | ~0.35 ms (FPS ~2877) |
| 输出 | `/predictor/Add_output_0` [1,401] (alphas) + `/predictor/Concat_5_output_0` [1,401,512] |

### 6.4 Decoder

Decoder 的校准数据必须包含 **4 个真实分布输入**（encoder 输出、frame_fires、token_num、bias_embed）。跑完整 FP32 pipeline 提取，存入 `real_calib/` 对应子目录。

**YAML：**

```yaml
model_parameters:
  onnx_model: ./decoder_only_final.onnx
  march: nash-e
  output_model_file_prefix: decoder_int16
  working_dir: ./decoder_int16_output

input_parameters:
  input_name: /encoder/after_norm/Add_1_output_0;token_num;bias_embed;onnx::Shape_8609
  input_shape: 1x400x512;1;1x1x512;1x100x512
  input_type_rt: featuremap;featuremap;featuremap;featuremap
  input_type_train: featuremap;featuremap;featuremap;featuremap
  input_layout_train: NCHW;NCHW;NCHW;NCHW
  separate_batch: false

calibration_parameters:
  cal_data_dir: ./real_calib/encoder_after_norm_Add_1_output_0;./real_calib/token_num;./real_calib/bias_embed;./real_calib/shape_8609
  calibration_type: max
  quant_config:
    model_config:
      all_node_type: int16
      activation:
        calibration_type: max

compiler_parameters:
  optimize_level: O2
  compile_mode: latency
  core_num: 1
  jobs: 32
```

**产物指标：**

| 项 | 值 |
|---|---|
| hbm 大小 | 73.5 MB |
| 静态延迟 | 5.77 ms (FPS 173) |
| DDR 占用 | 127 MB |
| 输入 | 4 个：encoder_out, token_num, bias_embed, pre_acoustic_embeds |
| 输出 | `logits [1, 100, 8404]` |

---

## 7. CIF (CPU numpy) 实现

CIF 在 CPU 用 numpy 实现，位于 encoder+predictor 与 decoder 之间：

```python
import numpy as np

MAX_LABEL_LEN = 100

def cif_numpy(alphas, concat5, real_T):
    """
    Args:
        alphas:  [1, 401] fp32 — predictor 输出（激活权重）
        concat5: [1, 401, 512] fp32 — predictor 拼接输出（decoder cross-attn 用）
        real_T:  int — 真实 LFR 帧数（用于 mask 掉 padding 区域的虚假 fire）
    Returns:
        frame_fires: [1, 100, 512] fp32 — pre_acoustic_embeds
        token_num:   [1] int32
    """
    alphas = alphas.copy()
    if real_T is not None and real_T < alphas.shape[1]:
        alphas[:, real_T:] = 0.0  # ★ 关键：不做这一步 CER 从 5% 涨到 44%

    B, T = alphas.shape
    H = concat5.shape[-1]

    prefix_sum = np.cumsum(alphas.astype(np.float64), axis=1).astype(np.float32)
    prefix_sum_floor = np.floor(prefix_sum)
    disl_ps_floor = np.floor(np.roll(prefix_sum, 1, axis=1))
    disl_ps_floor[:, 0] = 0
    fire_idxs = (prefix_sum_floor - disl_ps_floor) > 0

    fires = np.zeros_like(prefix_sum)
    fires[fire_idxs] = 1.0
    fires = fires + prefix_sum - prefix_sum_floor

    prefix_sum_hidden = np.cumsum(
        alphas[..., None].astype(np.float64) * concat5.astype(np.float64),
        axis=1).astype(np.float32)
    frames = prefix_sum_hidden[fire_idxs]
    shift_frames = np.roll(frames, 1, axis=0)

    batch_len = fire_idxs.sum(axis=1)
    batch_idxs = np.cumsum(batch_len)
    shift_batch_idxs = np.roll(batch_idxs, 1); shift_batch_idxs[0] = 0
    shift_frames[shift_batch_idxs] = 0

    remains = fires - np.floor(fires)
    remain_frames = remains[fire_idxs][:, None] * concat5[fire_idxs]
    shift_remain_frames = np.roll(remain_frames, 1, axis=0)
    shift_remain_frames[shift_batch_idxs] = 0

    frames = frames - shift_frames + shift_remain_frames - remain_frames

    frame_fires = np.zeros((B, MAX_LABEL_LEN, H), dtype=np.float32)
    indices = np.arange(MAX_LABEL_LEN)[None, :]
    batch_len_clamped = np.clip(batch_len, None, MAX_LABEL_LEN)
    slot_mask = indices < batch_len_clamped[:, None]
    num_slots = int(slot_mask.sum())
    frame_fires[slot_mask] = frames[:num_slots]

    return frame_fires, batch_len_clamped.astype(np.int32)
```

---

## 8. 端到端推理 pipeline

```python
import numpy as np
import soundfile as sf
import torch, json

# 加载 vocab
vocab = json.load(open("./models/paraformer/tokens.json", encoding="utf-8"))

# 加载 3 个 hbm 模型（示意；实际板端用 hrt/hbrt4 API）
encoder = load_hbm("./encoder_int16.hbm")
predictor = load_hbm("./predictor_int16.hbm")
decoder = load_hbm("./decoder_int16.hbm")

def paraformer_infer(wav_path):
    # 1. Frontend (CPU): fbank + LFR
    wav, sr = sf.read(wav_path); assert sr == 16000
    feats = extract_fbank_lfr(wav)              # [T, 560]
    real_T = feats.shape[0]                     # 真实帧数（用于 CIF mask）
    speech = pad_or_truncate(feats, 400)[None, ...].astype(np.float32)  # [1, 400, 560]

    # 2. Encoder (BPU INT16)
    enc_out = encoder.run(speech)               # [1, 400, 512]

    # 3. Predictor (BPU INT16)
    alphas, concat5 = predictor.run(enc_out)    # [1,401], [1,401,512]

    # 4. CIF (CPU numpy)
    pre_acoustic, token_num = cif_numpy(alphas, concat5, real_T)  # [1,100,512], [1]

    # 5. Decoder (BPU INT16)
    bias_embed = np.zeros((1, 1, 512), dtype=np.float32)  # 无热词
    logits = decoder.run(enc_out, token_num, bias_embed, pre_acoustic)  # [1,100,8404]

    # 6. Postprocess: argmax + tokenizer decode
    tn = int(token_num[0])
    ids = np.argmax(logits[0, :tn], axis=-1)
    return "".join(vocab[int(i)] for i in ids
                   if not (vocab[int(i)].startswith("<") and vocab[int(i)].endswith(">")))
```

---

## 9. WER 实测结果

### 9.1 评估配置

- **数据集**：AISHELL dev（`speech_asr_aishell_devsets`）300 条样本，覆盖 40 speakers
- **参考文本**：AISHELL 官方转写
- **指标**：CER（字级 Levenshtein / 总参考字数）
- **Pipeline**：fbank+LFR → 3 个 hbm（板端 hbm_runtime / UCP）/ 对应 ptq_model.onnx（主机仿真）+ CPU CIF

### 9.2 结果

| 环境 | 采样 | CER | 相对 FP32 |
|---|---|---|---|
| **FP32 baseline**（原始 ONNX） | 300 条 | **5.20%** | — |
| **主机仿真 INT16**（HMCT ptq_model） | 300 条 | **5.02%** | **-0.18 pp** ✅ |
| **板端实测 INT16 · Python hbm_runtime** | 300 条 | **3.13%** ✅ | **-2.07 pp** |
| **板端实测 INT16 · C++ UCP** | 300 条 | **3.13%** ✅ | **-2.07 pp** |

**结论：INT16 全量化不但没损失精度，板端硬件路径反而比主机 HMCT 仿真更精确（3.13% vs 5.02%）**——原因是 HMCT 的 QDQ 仿真节点引入了额外的舍入路径，而板端 hbm 直接用真实 BPU 定点算子实现。Python `hbm_runtime` 与 C++ UCP 输出 **逐 bit 一致**。

### 9.3 样本对比（板端前 5 条）

| # | GT | 板端 INT16 输出 |
|---|---|---|
| 1 | 广州市房地产中介协会分析 | ✓ 完全正确 |
| 2 | 新地王的诞生迅速搅热南沙土地市场 | 新地王的诞生迅速**绞**热南沙土地市场（1 字替换） |
| 3 | 证监会已依法移送公安机关处理 | ✓ 完全正确 |
| 4 | 公开表示汽车是终极移动设备 | ✓ 完全正确 |
| 5 | 医用的有天然的合成的等 | 医用的**由**天然**岛**合成的**岛**（3 字替换） |

主要错误：CIF 边界字符漏掉、少量同音替换。均属正常量化误差范围。

---

## 10. 板端部署与实测延迟

### 10.1 hbm 上板

将编译好的 3 个 hbm + 词表拷贝到板端（S100，Ubuntu 22.04 aarch64）：

```bash
cd model && bash download_model.sh
```

### 10.2 板端单模型延迟测试（`hrt_model_exec perf`）

板端自带命令行工具，可直接测每个 hbm 的纯 BPU 延迟：

```bash
# thread=1 → 单帧纯 BPU 延迟
hrt_model_exec perf --model_file encoder_int16.hbm  --thread_num 1 --frame_count 200
hrt_model_exec perf --model_file predictor_int16.hbm --thread_num 1 --frame_count 500
hrt_model_exec perf --model_file decoder_int16.hbm  --thread_num 1 --frame_count 200
```

**板端 perf 单帧延迟（S100，单 BPU 核）：**

| 模块 | perf 延迟 | 编译静态估算 | FPS |
|---|---|---|---|
| Encoder INT16 | **33.11 ms** | 32.52 ms | 30.18 |
| Predictor INT16 | **0.67 ms** | 0.35 ms | 1462 |
| Decoder INT16 | **6.12 ms** | 5.77 ms | 162.8 |

S100 单核 BPU 已被饱和；`--thread_num 8` 只会并发排队，单帧延迟不变。

### 10.3 板端端到端 pipeline 常驻推理

板端目录布局：

```
<rdk_model_zoo>/samples/speech/paraformer/
├── model/                     # 执行 download_model.sh，安装三段 HBM、tokens 和前端资源
├── runtime/python/            # hbm_runtime 实现：paraformer.py + main.py + run.sh
├── runtime/cpp/               # UCP/hbDNN 实现：CMakeLists.txt + src/paraformer.cpp + run.sh
└── <AUDIO_DATA_DIR>/
    ├── audio/*.wav            # 16 kHz 原始语音
    └── manifest.json          # [{utt_id, text}, ...]
```

#### 10.3.1 Python 常驻（推荐做原型验证）

用地瓜官方 `hbm_runtime` Python 包（详见 [Python 接口手册](https://developer.d-robotics.cc/rdk_s_doc/Algorithm_Application/python-api)）：

```bash
# 板端已预装 hobot-dnn，可直接 import
pip3 list | grep hbm-runtime
# hbm-runtime  0.1.0.post1

cd runtime/python
bash run.sh <AUDIO_DATA_DIR>                   # 默认全 manifest
N_UTT=30 bash run.sh <AUDIO_DATA_DIR>          # 或限定 30 条
```

当前 Python Runtime 仅包含两个代码文件：`paraformer.py` 负责 FunASR WAV 前处理、三段 HBM、CPU CIF 和解码；`main.py` 负责 manifest 调度、时延统计，并提供给 C++ Runtime 使用的内部特征生成模式。用户直接执行 `run.sh` 即可，无需自行调用 `hbm_runtime` 接口或准备 `.npy` 特征。

```python
# main.py 的正常模式：WAV → 前处理 → Encoder → Predictor → CIF → Decoder
python3 main.py \
  --manifest <AUDIO_DATA_DIR>/manifest.json \
  --audio-dir <AUDIO_DATA_DIR>/audio

# 仅供 runtime/cpp/run.sh 内部调用：WAV → build/eval/feats/*.npy
python3 main.py \
  --manifest build/eval/manifest.json \
  --audio-dir <AUDIO_DATA_DIR>/audio \
  --output-dir build/eval/feats \
  --preprocess-only
```

`--preprocess-only` 仅用于 C++ Runtime bridge：它在临时目录写入特征和 `feat_length`，C++ 可执行程序仍保持原有 hbUCP/hbDNN pipeline。普通用户不需要手动使用该模式。

**Python 300 条验证记录（实现见 `runtime/python/paraformer.py` 与 `runtime/python/main.py`）：**

| 阶段 | 延迟 |
|---|---|
| Encoder (BPU) | 33.63 ms |
| Predictor (BPU) | 1.44 ms |
| CIF (CPU numpy) | 3.41 ms |
| Decoder (BPU) | 7.12 ms |
| **端到端** | **45.61 ms/utt** |
| **RTF** | **~0.008** |

**wall-clock**：300 条 14.0 秒（46.7 ms/utt 含 numpy I/O）

上述 `45.61 ms/utt` 是历史验证的 HBM pipeline 指标（Encoder、Predictor、CPU CIF、Decoder），不包含 WAV 前处理。当前 `main.py` 会单独输出 `frontend_ms`、`Average HBM pipeline` 和 `Average end-to-end`，因此从 WAV 直接运行时的端到端时间会额外包含前处理耗时，不应与历史 HBM pipeline 指标直接混用。

#### 10.3.2 C++ UCP 常驻（推荐上产）

用底层 `hbUCP` + `hbDNN` C API 直接对接 `libhbucp.so` / `libdnn.so`。见 `runtime/cpp/src/paraformer.cpp` 完整实现，编译：

```bash
cd runtime/cpp
N_UTT=300 bash run.sh <AUDIO_DATA_DIR>       # 跑 300 条 CER
```

`runtime/cpp/run.sh` 会先调用 `../python/main.py --preprocess-only` 将 WAV 写入 `build/eval/feats/`，再启动 C++ 可执行程序。该 `.npy` 目录为运行时临时 bridge，用户只需要提供 WAV 和 manifest。

关键 API 调用序列：

```cpp
#include <hobot/dnn/hb_dnn.h>
#include <hobot/hb_ucp.h>
#include <hobot/hb_ucp_sys.h>

// 一次加载 3 个 hbm
hbDNNPackedHandle_t packed;
const char* files[] = {"/opt/hobot/model/s100/basic/paraformer/paraformer_large_encoder_400x560_s100.hbm",
                       "/opt/hobot/model/s100/basic/paraformer/paraformer_large_predictor_400x512_s100.hbm",
                       "/opt/hobot/model/s100/basic/paraformer/paraformer_large_decoder_400x512_s100.hbm"};
hbDNNInitializeFromFiles(&packed, files, 3);

// 每个模型的 handle + 张量属性 + 预分配 BPU 内存（hbUCPMalloc）
hbDNNHandle_t enc_h; hbDNNGetModelHandle(&enc_h, packed, "paraformer_encoder_int16");
hbDNNTensor enc_in, enc_out;
hbDNNGetInputTensorProperties(&enc_in.properties, enc_h, 0);
hbUCPMalloc(&enc_in.sysMem, enc_in.properties.alignedByteSize, 0);
// ... 类似给 predictor / decoder 的所有输入输出预分配

// 每次推理循环
memcpy(enc_in.sysMem.virAddr, speech_data, N);   // 需考虑 aligned stride
hbUCPMemFlush(&enc_in.sysMem, HB_SYS_MEM_CACHE_CLEAN);
hbUCPTaskHandle_t task;
hbDNNInferV2(&task, &enc_out, &enc_in, enc_h);
hbUCPSchedParam sched;
HB_UCP_INITIALIZE_SCHED_PARAM(&sched);
sched.backend = HB_UCP_BPU_CORE_ANY;
hbUCPSubmitTask(task, &sched);
hbUCPWaitTaskDone(task, 0);
hbUCPReleaseTask(task);
hbUCPMemFlush(&enc_out.sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
// 读出 enc_out.sysMem.virAddr，同样处理 predictor / CIF / decoder
```

**⚠️ Aligned stride 处理**：BPU 内存按 16 字节对齐（例如 [1,400,560] fp32 的 stride 是 [921600, 2304, 4] 而非 [896000, 2240, 4]）。`memcpy` 平摊拷贝会数据错位，必须按行 padding；示例代码里的 `Model::copy_tensor()` 处理了 fast/slow 两种路径。

**C++ UCP 300 条验证记录（实现见 `runtime/cpp/src/paraformer.cpp`）：**

| 阶段 | Python | **C++ UCP** | 加速 |
|---|---|---|---|
| Encoder (BPU) | 33.63 | **33.15** | ≈ |
| Predictor (BPU) | 1.44 | **1.00** | ≈ |
| CIF (CPU) | 3.41 | **0.38** | **9x** |
| Decoder (BPU) | 7.12 | **6.29** | ≈ |
| **端到端** | 45.61 | **40.81 ms/utt** | **1.12x** |
| **CER** | 3.13% | **3.13%** | 一致 |

**wall-clock**：300 条 13.4 秒（44.7 ms/utt）；一次模型加载 ~1.85 s

C++ 版本主要优势：CPU 侧 CIF 快 9x（numpy 开销比手写循环大），BPU 部分相同（都通过同一底层库）。

### 10.4 结论

**部署选型：**
- **原型 / 调试**：Python `hbm_runtime`（`runtime/python/paraformer.py` 与 `runtime/python/main.py`）— 40+ 行代码即可上手
- **生产 / 追求极致延迟**：C++ UCP（`runtime/cpp/src/paraformer.cpp`）— 40.8 ms/utt，RTF 0.007

---

## 11. 常见问题与经验

### 11.1 校准数据必须匹配真实分布

**症状**：INT16 pipeline CER = 100%（decoder 所有位置 argmax 到 `</s>`），FP16 输出全 NaN。

**根因**：使用 `np.random.randn` 生成校准数据，encoder 输出在 N(0,1) 空间，但实际部署时 encoder 输出范围只在 [-0.4, 0.3]，INT16 量化 scale 完全错位。

**修复**：跑 FP32 pipeline 在 50 条真实音频上，捕获真实的 `enc_out / concat5 / alphas / frame_fires / token_num / bias_embed`，用作校准数据。

**教训**：**校准数据的分布必须严格匹配部署时的输入分布。用随机数据是最常见的量化失败原因。**

### 11.2 Padding 会产生虚假 CIF fire

**症状**：FP32 pipeline CER = 44.4%，输出前 N 字正确，后面全是垃圾字符。

**根因**：encoder 的 `make_pad_mask` 被常量折叠成全 1（fixed shape 编译要求），padding 区域的 encoder 输出未被 mask，predictor 在这些位置也生成非零 alphas → CIF 虚假 fire。

**修复**：CIF 前 `alphas[:, real_T:] = 0`，其中 `real_T` 是真实 LFR 帧数。**必须做这一步**，否则 CER 从 5% 升到 44%。

### 11.3 生成真实校准数据（脚本示例）

```python
# gen_real_calib.py
import os, glob
import numpy as np
import onnxruntime as ort

enc = ort.InferenceSession("encoder_nomask.onnx", providers=["CPUExecutionProvider"])
pred = ort.InferenceSession("predictor_only.onnx", providers=["CPUExecutionProvider"])

BASE = "./real_calib"
for name in ["encoder_after_norm_Add_1_output_0", "predictor_Concat_5_output_0",
             "predictor_Add_output_0", "shape_8609", "token_num", "bias_embed"]:
    os.makedirs(f"{BASE}/{name}", exist_ok=True)

for i, feat in enumerate(sorted(glob.glob("./calib_data/speech/*.npy"))[:50]):
    speech = np.load(feat)  # [1, 400, 560]
    enc_out = enc.run(None, {"speech": speech})[0]
    pr = dict(zip([o.name for o in pred.get_outputs()],
                  pred.run(None, {"/encoder/after_norm/Add_1_output_0": enc_out})))
    alphas = pr["/predictor/Add_output_0"]
    concat5 = pr["/predictor/Concat_5_output_0"]
    frame_fires, token_num = cif_numpy(alphas, concat5, real_T=None)  # 校准用完整长度
    bias = np.zeros((1, 1, 512), dtype=np.float32)

    np.save(f"{BASE}/encoder_after_norm_Add_1_output_0/{i:03d}.npy", enc_out)
    np.save(f"{BASE}/predictor_Concat_5_output_0/{i:03d}.npy", concat5)
    np.save(f"{BASE}/predictor_Add_output_0/{i:03d}.npy", alphas)
    np.save(f"{BASE}/shape_8609/{i:03d}.npy", frame_fires)
    np.save(f"{BASE}/token_num/{i:03d}.npy", token_num)
    np.save(f"{BASE}/bias_embed/{i:03d}.npy", bias)
```

---

## 12. 已知限制

1. **Contextual embedder**：热词编码器 (`bias_encoder`) 仍是 FP32 ONNX，未上 BPU。本文示例走全 zeros bias（等同于关闭 Contextual）。启用热词需单独编 embedder hbm。
2. **音频长度上限**：当前固定 T=400 帧（≈24 秒）。<24s 音频用 padding + `alphas[real_T:]=0`；>24s 需分段处理。

---

## 13. 最终配置汇总

| 模块 | 量化 | hbm 大小 | 板端 perf 单帧 | Python 常驻 | C++ UCP 常驻 |
|---|---|---|---|---|---|
| Encoder | INT16 all | 211.5 MB | 33.11 ms | 33.63 ms | 33.15 ms |
| Predictor | INT16 all | ~4 MB | 0.67 ms | 1.44 ms | 1.00 ms |
| CIF | CPU numpy | — | — | 3.41 ms | 0.38 ms |
| Decoder | INT16 all | 73.5 MB | 6.12 ms | 7.12 ms | 6.29 ms |
| **端到端** | — | **~289 MB** | ~41 ms | **45.61 ms** | **40.81 ms** |
| **CER** | — | — | — | **3.13%** | **3.13%** |

**精度总览：**

| 环境 | AISHELL dev 300 条 CER |
|---|---|
| FP32 baseline (原始 ONNX) | 5.20% |
| INT16 主机仿真 (HMCT ptq_model) | 5.02% |
| **INT16 板端实测 (Python / C++ UCP)** | **3.13%** ✅ |

**部署路径：**
- 原型验证 → `runtime/python/paraformer.py` 与 `runtime/python/main.py`（Python `hbm_runtime`）
- 生产部署 → `runtime/cpp/src/paraformer.cpp`（`hbUCP` + `hbDNN` C API）
