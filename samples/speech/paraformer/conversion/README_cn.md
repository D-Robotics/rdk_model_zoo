# Paraformer 模型转换

[English](README.md) · [Sample 入口](../README_cn.md) · [Python 运行](../runtime/python/README_cn.md)

本目录现在能从固定 FunASR 架构的真实权重直接导出 encoder、predictor、decoder，
固定部署形状并与 Torch 做数值检查。CIF 继续使用共享 CPU 实现。
**真实音频校准和显式 OE 编译编排已实现。** 实际 OE/SDK/板端验证及专用 evaluator
仍待完成；FP32 导出通过不能证明 HBM 或板端行为。

源模型是 `iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404`。
S 源实现将其拆成 encoder、predictor、decoder，CPU CIF 连接 predictor 和
 decoder。已有发布制品的准备方式见[模型说明](../model/README_cn.md)。
转换自己的权重是另一项操作；这里的图变换不会下载权重、生成 HBM，
也不构成对已有发布制品的认证。

## 导出环境与快速开始

使用独立的 Python 3.12 环境。本次实际验证的组合是 Torch/torchaudio 2.6.0、
FunASR 1.3.14、NumPy 1.26.4、ONNX 1.17.0、ONNX Runtime 1.20.1、
protobuf 4.23.0、ModelScope 1.40.1。导出仅用 CPU，不需要板端 SDK、
ONNX Simplifier 或 OE。架构和前处理配套文件固定；本地 `model.pt` 必须严格
匹配全部参数，缺失参数不会由随机初始权重代替。

从仓库根目录执行：

```bash
python3.12 -m venv .venv-paraformer-export
. .venv-paraformer-export/bin/activate
python -m pip install -r samples/speech/paraformer/conversion/requirements-export.txt
python samples/speech/paraformer/conversion/export.py --help
```

若尚无源权重，请先显式下载。`model.pt` 约 913 MB，另有元数据；
还需为原始及变换后的 ONNX 图预留空间。以下示例只写入
`models/paraformer-source`。下载的是 hub 的 `master`，导出报告记录实际
本地文件摘要，**不把可变分支名当作不可变的发布权重版本**。

```python
from modelscope import snapshot_download
snapshot_download(
    "iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404",
    revision="master",
    local_dir="models/paraformer-source",
    allow_file_pattern=["model.pt", "config.yaml", "tokens.json", "am.mvn"],
)
```

导出到新目录：

```bash
python samples/speech/paraformer/conversion/export.py \
  --model-dir models/paraformer-source \
  --output-dir outputs/paraformer_export
```

若要用自己的音频检查，请先用 [Python 前端](../runtime/python/README_cn.md)
准备特征 NPY，再以一个或多个 `--feature` 参数开始新的导出：

```bash
python samples/speech/paraformer/conversion/export.py \
  --model-dir models/paraformer-source \
  --output-dir outputs/paraformer_export_with_audio \
  --feature outputs/paraformer_features/feats/BAC009S0724W0121.npy
```

特征路径必须已经存在，它不是 WAV；文件必须是有限值 float32 `[1,400,560]`
数组。导出检查使用不按有效帧屏蔽的 CIF 来验证模型边界，不使用单条音频的
有效帧长度，也不计算 CER。

| 参数 | 含义 |
| --- | --- |
| `--model-dir` | 必填，本地目录须包含 `model.pt`、`config.yaml`、`tokens.json`、`am.mvn`；不会隐式下载。后三个配套文件必须匹配固定源摘要。 |
| `--output-dir` | 必填且必须不存在；不覆盖源权重或已有导出。 |
| `--feature` | 可重复指定的准备后 NPY；可省略，但零/随机特征和 decoder count=0、1、17、100 的检查仍执行。 |
| `--threads` | 正整数 CPU 线程数，默认 4。 |

输出为 `encoder.onnx`、`predictor.onnx`、`decoder.onnx`、对应 `*.raw.onnx`
诊断图，以及 `export-report.json`。只有最终文件完成物理名称、形状及图变换
检查；原始图可能保留 Torch 自动命名，不能作为部署接口。

成功报告的 `status` 为 `completed`，含源文件/特征/模型 SHA-256、依赖版本、
节点数和逐用例最大绝对差值。所有完整输出须满足 `rtol=1e-4, atol=1e-4`，
类型和形状不变且数值有限。参数/预检失败返回 2 且不创建输出；之后发生错误
则返回 2，保留部分文件和 `failed` 报告。部分制品不是可验收导出，强制中断
进程也可能留下不完整状态。

## 准备真实校准数据与编译配置

导出后，生成新的独立工作目录。下面使用两条自带 WAV 验证操作流程，
**不构成代表性校准集或精度验收**：

```bash
python samples/speech/paraformer/conversion/prepare.py \
  --export-dir outputs/paraformer_export \
  --wav-dir samples/speech/paraformer/test_data \
  --sample-count 2 \
  --output-dir outputs/paraformer_calibration
```

实际转换应提供能代表业务分布的 16 kHz WAV 集合。递归查找小写 `*.wav`，
排序后取前 N 条，默认 50，对齐源配方的参考数量；不足时如实记录。
50 条本身也不能证明覆盖充分。空集合、采样率不符、音频损坏或阶段输出不合规
都会使本轮失败，不静默跳过选中的文件，不隐式重采样，也不生成随机校准数据。

统一前端执行多声道平均、fbank/LFR/CMVN，CPU 种子固定为 191009，再填充或
截断至 400 帧；直接复用运行时前端，不为取特征而构造整套 AutoModel。
源校准脚本依赖的全局随机状态不是可复现契约；当前种子、依赖和输入摘要
写入报告，每条记录保留原始/有效帧数与截断状态。

encoder、predictor 由 CPU ONNX Runtime 实际执行，之后调用同一 CPU CIF，
显式传 **`real_T=None`**，保留源校准不按有效帧屏蔽的分布。运行时推理才使用
音频的有效帧数。零 token 样本仍保存，不静默移除。

| 校准子目录 | 形状 | dtype | 用途 |
| --- | --- | --- | --- |
| `speech` | `[1,400,560]` | float32 | encoder 输入 |
| `encoder_after_norm_Add_1_output_0` | `[1,400,512]` | float32 | predictor/decoder 的 context |
| `predictor_Add_output_0` | `[1,401]` | float32 | predictor 权重与 CIF 来源记录 |
| `predictor_Concat_5_output_0` | `[1,401,512]` | float32 | predictor 隐藏向量与 CIF 来源记录 |
| `shape_8609` | `[1,100,512]` | float32 | decoder 声学嵌入 |
| `token_num` | `[1]` | int32 | decoder 有效 token 数 |
| `bias_embed` | `[1,1,512]` | float32 | 全零上下文偏置 |

各目录使用对齐文件名，例如 `000000.npy`。工作目录还包含
`source/{encoder,predictor,decoder}.onnx`、原导出报告和 CMVN 快照、
`configs/{encoder,predictor,decoder}.yaml`，以及 `preparation.json`。
报告给每个源输入和派生数组记录摘要、形状、类型和值域。配置内的模型/校准
路径相对于工作目录，若直接运行原配置须以工作目录为 cwd；下述编译入口
会显式重映射路径。

| 准备参数 | 默认值/行为 |
| --- | --- |
| `--export-dir` | 必填，已完成的导出目录；核验报告摘要和真实 ONNX 签名，拒绝外部数据文件形式的 ONNX。 |
| `--wav-dir` | 必填，真实 WAV 目录；读取和哈希使用同一份字节。 |
| `--output-dir` | 必填且必须不存在；不覆盖或在部分结果中续写。 |
| `--cmvn-file` | 默认 Sample 的 `model/am.mvn`，必须符合固定摘要。 |
| `--sample-count` | 正整数，默认 50；另记实际选中数量。 |
| `--threads` | 正整数 ONNX Runtime CPU 线程数，默认 4。 |
| `--jobs` | 写入 YAML 的正整数 OE 编译并行数，默认与源相同的 32。 |

`status: prepared` 只表示校准和配置准备完成。开始后的准备失败会写入
`status: preparation_failed`，保留当前音频、已完成记录和部分文件，退出码为 2；
编译入口拒绝这些部分工作目录。依赖/前置检查也可能在创建目录之前失败。

## 显式 S100 / nash-e 编译

配方只支持 **S100 / nash-e**，保留源 max 校准、内部 INT16、NCHW featuremap、
O2 latency 优化、单 BPU 核和关闭编译缓存的设置。token 数输入保持 int32。
内部 INT16 不等于最终物理 I/O 精度保证，运行前仍需通过 SDK 核验真实 HBM 签名。

应使用匹配的、已安装 `hb_compile` 的 S OE 工具链。源记录为
`ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0`、hbdk4 4.7.5；此处未验证镜像仓库
及可用性，不虚构拉取地址。FP32 导出环境本身不提供 OE 编译器。在 OE 环境中也须能访问本仓库，
从仓库根目录运行入口，并安装 NumPy、PyYAML；编译步骤不导入 Torch。

```bash
python samples/speech/paraformer/conversion/compile.py \
  --workspace outputs/paraformer_calibration \
  --output-dir outputs/paraformer_compiled \
  --compiler hb_compile
```

`--workspace` 和**新的** `--output-dir` 必填；`--compiler` 默认 `hb_compile`，
也可指定可执行文件。工作目录路径不能含 OE 校准目录分隔符 `;`。
先把整个准备目录移动或挂载到工具链环境。入口核验全部快照/配置/NPY 摘要，
拒绝额外加入的校准文件，再生成本轮绝对路径配置，保持原准备目录不变。
逐阶段执行 `hb_compile -c <stage.yaml>`，全部结束后再次核验准备目录。

每阶段记录精确 argv/cwd、UTC 时间、配置摘要、完整且分开的 stdout/stderr
及退出码。进程无法启动时退出码为空；非零退出，或退出为零但没有预期的非空
HBM，均判失败并停止后续阶段。`compile-report.json` 保留此前完成的阶段。
重试请使用新输出目录，先前日志和部分制品仍保留。

相对于本轮编译目录的预期输出：

| 阶段 | HBM | 可选量化 ONNX，供后续评估 |
| --- | --- | --- |
| encoder | `encoder/paraformer_encoder_int16.hbm` | `encoder/paraformer_encoder_int16_ptq_model.onnx` |
| predictor | `predictor/predictor_int16.hbm` | `predictor/predictor_int16_ptq_model.onnx` |
| decoder | `decoder/decoder_int16.hbm` | `decoder/decoder_int16_ptq_model.onnx` |

即使三个非空 HBM 都生成且退出码为零，也只记 **`compiled_unverified`**，
不是板端/SDK/精度验收。不会自动发布、改名成官方资产或拷入运行时模型目录。
本机没有 OE，仅做了编排夹具测试及真实的“编译器缺失”拒绝检查。
校准准备不保证量化质量；没有 PTQ ONNX 时明确记录缺失，不伪造文件。
详见[校准与编译准备证据](../../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-calibration-review.md)。

## 固定部署语义

encoder 始终处理 400 帧，推理时真实音频长度由 CPU CIF 应用。predictor 输出
401 个权重和 401 个隐藏向量，包含源实现的 0.45 尾部及零隐藏帧。decoder
物理宽度固定为 100，`token_num` 改变有效前缀 mask，而不是输出形状。
保留发布接口的名称，包括 `onnx::Shape_8609`；显式输出绑定避免依赖某次 Torch
生成的内部张量编号后缀。

decoder 编排参考 FunASR，并保留 [MIT 许可](LICENSE-FunASR)。固定宽度 mask
直接表达既有部署契约，代替旧流程按一次输入探测后固化 Range 的方式。
上游通用 decoder 直接接受**未填充** token 序列时，短序列结果不能简单视为
数值等价；此项比较失败已保留。部署对照采用未修改的上游 decoder 导出器，
再执行历史源固定 100 宽度的 Range 处理。不能由这些导出检查推断原始变长
模型的精度或 HBM 的精度。

## 图测试

```bash
python -m unittest discover -s samples/speech/paraformer/tests -v
```

此前独立图工具还在 Python 3.14.7、NumPy 2.5.3、ONNX 1.23.0、
ORT 1.30.0 上测试过；这不代表 FunASR 导出兼容该环境。
可选依赖缺失时部分测试会跳过，请使用上述完整导出环境并核对实际执行数量，
不要把 skip 当作验证通过。

## 已提供的变换

所有函数接收内存中的 ONNX `ModelProto`，在副本上处理并返回新模型，
不读写文件、不下载内容、不调用 OE。失败时抛出异常，不修改调用者的模型。
当前支持普通的平面张量图，显式拒绝嵌套控制流；常量求值器不支持稀疏初始化器
和自定义 domain 算子。

| 函数 | 行为及拒绝条件 |
| --- | --- |
| `topological_sort(model)` | 按张量依赖排序，保留模型元数据和重复的节点名称。遇到环、缺失输入/输出或重复张量生产者时拒绝。只做依赖检查，之后还需调用 `onnx.checker.check_model`。 |
| `gather_indices_int32(model)` | 只为 Gather 的索引生成 INT32 表示，保留其他节点使用的原常量。常量溢出、类型无法确定、动态 INT64 索引均默认拒绝。 |
| `gather_indices_int32(model, allow_dynamic=True)` | 只有调用者已证明运行时索引处于 INT32 范围时才能启用。插入 Cast，但不插入运行时边界检查，超范围值可能回绕。超出求值规模限制的直接常量会拒绝，不会自动按动态输入处理。 |
| `fold_constant_ranges(model)` | 只折叠由受支持的确定性常量运算或固定形状信息证明的 Range。数据相关、无法求值或超规模时拒绝，不使用一次示例推理的结果固化图。 |
| `normalize_axes(model)` | 对受支持的输入秩语义算子归一化负轴；共享轴常量按各消费者分别复制。需要归一化而输入秩未知时拒绝。Unsqueeze 使用输出秩，故不包含在此变换中。 |

后三个函数返回前会运行 ONNX checker。常量传播只支持有限算子白名单，
接受的常量和结果最多一百万个元素。这是求值支持边界，不是安全沙箱，
也不是整个进程的内存用量保证。输入形状元数据必须真实描述模型，
不能靠一次推理观察来建立此契约。

## 可执行 API 示例

从仓库根目录、安装了 ONNX 的环境中运行。以下只是小图示例，
不是 Paraformer 权重的转换命令；也展示了排序后显式调用 checker 的方式。

```python
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from samples.speech.paraformer.conversion.graph_ops import (
    gather_indices_int32,
    topological_sort,
)

model = helper.make_model(
    helper.make_graph(
        [helper.make_node("Gather", ["features", "index"], ["selected"])],
        "gather-example",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("selected", TensorProto.FLOAT, [1])],
        initializer=[numpy_helper.from_array(np.array([1], np.int64), "index")],
    ),
    opset_imports=[helper.make_opsetid("", 15)],
    ir_version=10,
)
original = model.SerializeToString()
rewritten = gather_indices_int32(topological_sort(model))
onnx.checker.check_model(rewritten)
assert model.SerializeToString() == original
assert rewritten.graph.node[0].input[1] != "index"
print("Gather-only rewrite validated; input model preserved")
```

将这些函数接入文件流程时，应先检查新图，并用代表性输入比较原图与新图输出，
再保存为新文件；不要覆盖源模型。动态索引的 Cast 还需要模型契约提供范围依据，
几个测试输入通过不足以证明这一点。API 不自动替调用者作出这两个判断。

## 源流程与待迁移部分

S 源提交 `380e1a2bf42041af54be6f34935e50197cfadff9` 的
[完整原始中文说明](../../../../platforms/s/samples/speech/paraformer/conversion/README_cn.md)
作为历史资料保留。各步骤当前边界如下：

| 源步骤 | 用途 | 统一实现状态 |
| --- | --- | --- |
| `01_reexport_fixed_shape.py` | 导出固定形状模型 | 由 `export.py` 直接导出三个阶段替代；不生成整套 CIF 图、不使用全局补丁或覆盖源输出。 |
| `02_extract_decoder.py`、`07_extract_predictor.py`、`08_extract_encoder.py` | 依赖内部名称切出三个阶段 | 由显式 Torch 阶段边界替代，保持部署名称和形状。 |
| `03_convert_gather_int64_to_int32.py` 至 `06_shape_freeze.py` | 处理 Gather、依赖顺序、Range 和轴 | 共享工具已接入真实权重导出。不调用简化器，故不接受未经核验的简化器结果。 |
| `09_gen_calib_features.py` | 从代表性真实音频生成特征 | 已由 `prepare.py` 接入统一、确定性前端。 |
| `10_gen_real_calib.py`、`cif_numpy.py` | 运行阶段模型，生成 predictor/decoder 校准数据 | 已实现真实 encoder/predictor 执行及共享不屏蔽 CIF（`real_T=None`），区别于运行时有效帧屏蔽。 |
| 三份 `*_int16.yaml` | 为 `nash-e` 编译三个阶段 | 已按源参数生成路径一致的配置并提供显式 OE 调用；真实编译/SDK 验证未执行。 |
| `11_eval_pipeline.py` | 比较三阶段语音流程 | 待迁入专用 evaluator。 |

源配方采用 max 校准、内部 INT16、O2 latency 优化及单 BPU 核。
这些配置不能确定最终物理输入输出类型，也不能证明新 HBM 的运行兼容性；
运行时仍须核验真实阶段签名。源脚本的 `out/` 默认路径与 YAML 相对根目录
的路径存在不一致，直接照搬执行不能当作已验证的端到端流程。
源性能数字只保留历史意义，本次图测试不证明 CER、延迟或数据集精度。

## 当前验证范围

十项图测试覆盖共享 Gather 常量、常量溢出和求值规模限制、动态 Cast 显式启用、
中间张量命名唯一性、依赖排序、残缺图拒绝、静态/动态 Range，以及不同秩
消费者共享负轴的处理。数值对照实际使用 ONNX Runtime 运行原图和新图，
要求输出类型和值一致。

真实权重阶段导出及两条完整示例流程单独验证；OE 编译、SDK 执行、板测和数据集 CER
仍未验证。详见[主机证据记录](../../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-graph-ops-review.md)。

真实权重结果、初次失败和复现方式见[导出记录](../../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-export-review.md)。

16 项导出检查使用真实 encoder 生成的 context（零/随机特征及两条真实音频特征）。
另用任意随机隐藏向量做压力检查时，Torch/ORT 差异明显扩大，**不满足**导出容差。
相同 ORT 设置下，旧、新固定宽度 ONNX 图在这些压力用例中一致；这证明源部署
行为得到保留，不代表全输入域 Torch/ONNX 等价。两条示例转录与参考文本也存在
识别错误，不声明数据集 CER 或精度提升。
