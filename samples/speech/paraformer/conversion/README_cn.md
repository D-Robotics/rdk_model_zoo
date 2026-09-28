# Paraformer 模型转换

[English](README.md) · [Sample 入口](../README_cn.md) · [Python 运行](../runtime/python/README_cn.md)

本目录目前提供**主机 ONNX 图变换工具**，还不是完整的 Paraformer 导出器或
OE 编译流程。统一导出、三个阶段的切图、校准和编译编排仍在迁移。
图测试通过不能视为语音模型转换成功。

源模型是 `iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404`。
S 源实现将其拆成 encoder、predictor、decoder，CPU CIF 连接 predictor 和
 decoder。已有发布制品的准备方式见[模型说明](../model/README_cn.md)。
转换自己的权重是另一项操作；这里的图变换不会下载权重、生成 HBM，
也不构成对已有发布制品的认证。

## 依赖与验证命令

图模块依赖 NumPy 和 ONNX；数值测试还需要 ONNX Runtime 的 CPU provider。
本次实际使用 Python 3.14.7、NumPy 2.5.3、ONNX 1.23.0、ONNX Runtime 1.30.0。
这些版本描述的是**本次图测试环境**，不代表 FunASR 导出和 OE 工具链的兼容组合。
源 Torch/FunASR 导出器应使用独立环境，尚未验证它与上述版本的兼容性。

在仓库根目录、已安装上述依赖的环境中执行：

```bash
python -c 'import numpy, onnx, onnxruntime; print(numpy.__version__, onnx.__version__, onnxruntime.__version__)'
python -m unittest samples.speech.paraformer.tests.test_conversion_graph_ops -v
```

缺少 ONNX 或 ORT 时测试模块会跳过。请检查测试实际运行数量，不能把 skip 当作通过。

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
| `01_reexport_fixed_shape.py` | 导出固定 400 特征帧、最多 100 token 的整图 | 待迁移；源脚本会修改补丁和已有输出，不能假设可安全重复执行。 |
| `02_extract_decoder.py`、`07_extract_predictor.py`、`08_extract_encoder.py` | 提取三个 ONNX 阶段 | 待迁移；单次输入探测的边界值不能视为常量证明。 |
| `03_convert_gather_int64_to_int32.py` 至 `06_shape_freeze.py` | 处理 Gather、依赖顺序、Range 和轴 | 上述共享工具已实现并完成小图测试；真实整图集成与简化器结果核验仍待完成。 |
| `09_gen_calib_features.py` | 从代表性真实音频生成特征 | 待接入统一可复现前端。 |
| `10_gen_real_calib.py`、`cif_numpy.py` | 运行阶段模型，生成 predictor/decoder 校准数据 | 待迁移；源校准明确使用无有效帧屏蔽的 CIF（`real_T=None`），与运行时按有效帧屏蔽不同。 |
| 三份 `*_int16.yaml` | 为 `nash-e` 编译三个阶段 | 仅有历史源配方，尚无统一 OE 调用及新编译模型验证。 |
| `11_eval_pipeline.py` | 比较三阶段语音流程 | 待迁入专用 evaluator。 |

源配方采用 max 校准、内部 INT16、O2 latency 优化及单 BPU 核。
这些配置不能确定最终物理输入输出类型，也不能证明新 HBM 的运行兼容性；
运行时仍须核验真实阶段签名。源脚本的 `out/` 默认路径与 YAML 相对根目录
的路径存在不一致，直接照搬执行不能当作已验证的端到端流程。
源性能数字只保留历史意义，本次图测试不证明 CER、延迟或数据集精度。

## 当前验证范围

九项图测试覆盖共享 Gather 常量、常量溢出和求值规模限制、动态 Cast 显式启用、
中间张量命名唯一性、依赖排序、残缺图拒绝、静态/动态 Range，以及不同秩
消费者共享负轴的处理。数值对照实际使用 ONNX Runtime 运行原图和新图，
要求输出类型和值一致。

Sample 主机回归与真实权重、OE 编译、SDK 执行和板测分别记录；本次转换工作
尚未验证后四项。详见[主机证据记录](../../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-graph-ops-review.md)。
