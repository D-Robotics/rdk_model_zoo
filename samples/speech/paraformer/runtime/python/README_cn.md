# Paraformer Python CPU 中间处理

[English](README.md)

当前目录已提供 CPU CIF（连续积分触发）实现。统一音频前端、三模型运行流程、
命令行入口及 C++ 桥接仍在迁移，当前还不是可直接上板运行的完整 Sample。
下述主机检查不代表模型推理或精度验证。归档的
[S 运行说明](../../../../../platforms/s/samples/speech/paraformer/runtime/python/README_cn.md)
保留作历史参考，不是统一入口。

## 在流程中的位置

源流程为：音频 → FunASR 前端 → encoder → predictor → CPU CIF → decoder → 文本。
CIF 接收 predictor 的权重和隐藏状态，生成 decoder 使用的固定形状声学嵌入与
有效 token 数。它不执行模型、不读写文件、不解码词表，也不选择板卡。
独立的 [cif.py](cif.py) 让运行与校准复用同一份数值逻辑，避免把辅助函数塞进模型推理类。

## 依赖与可执行主机示例

需要 Python 和 NumPy；仅运行此数值模块不需要 Torch、FunASR、厂商 SDK 或板卡。
在仓库根目录执行：

```bash
python - <<'PYCODE'
import numpy as np
from samples.speech.paraformer.runtime.python.cif import cif_numpy

weights = np.zeros((1, 401), dtype=np.float32)
hidden = np.zeros((1, 401, 512), dtype=np.float32)
weights[0, :3] = [0.75, 0.75, 0.5]
hidden[0, :3] = np.array([2, 6, 10], dtype=np.float32)[:, None]
embeddings, token_count = cif_numpy(weights, hidden, real_T=3)
print(embeddings.shape, token_count.tolist(), embeddings[0, :2, 0].tolist())
PYCODE
```

预期输出：`(1, 100, 512) [2] [3.0, 8.0]`。两次累计权重跨越整数边界，分别积分
得到对应的声学嵌入。这是合成数据示例，不是语音识别结果。

## 接口契约

| 参数／返回值 | 契约 |
| --- | --- |
| `alphas` | 有限且非负的 `float32 [1,401]` 权重 |
| `concat5` | 有限的 `float32 [1,401,512]` 隐藏状态 |
| `real_T` | 必须显式指定；推理使用 `0…400` 整数，仅无屏蔽校准使用 `None` |
| 声学嵌入 | 独立持有的 `float32 [1,100,512]`，未使用的行补零 |
| token 数 | 独立持有的 `int32 [1]`，最大为 100 |

推理时先将 `real_T` 及之后的权重置零，再累计。有效帧为零或总权重不足 1 时，
返回零嵌入和零计数；调用方须处理此计数，本函数不决定是否执行 decoder。
不足下一整数的残余权重不产生 token；超过 100 个嵌入时保留前 100 个，与源契约一致。

计算保留源实现的 float64 累加后转 float32，以及每帧最多触发一次的规则；它不是
面向权重大于 1 的通用多次触发积分器。输入不会被修改。形状、类型、有限性、
权重非负性及有效帧数不符合契约时直接报错，不自动转换类型或扩展 batch。
`real_T=None` 保留源校准流程的无屏蔽分布，不能用来替代推理时的 padding 屏蔽。

## 验证与剩余工作

```bash
python -m unittest discover -s samples/speech/paraformer/tests -v
```

7 项主机测试覆盖空输出、手算分数边界、padding 与校准模式、100 token 截断、
24 组源实现对照、输入所有权与非法契约。对照使用归档的 S 提交
`380e1a2bf42041af54be6f34935e50197cfadff9`；源实现无 token 时抛出 `IndexError`，
统一实现已修复。详见[评审记录与证据](../../../../../docs/releases/unified-migration/2026-09-28-b10-paraformer-cif-review.md)。

源清单仅发布 S100 模型，本次不声明 X5、S100P 或 S600 适配。真实前端等价性、
SDK 绑定、三段推理、原生 C++、完整转换和评测流程以及各层双语 Sample 文档
仍待完成。板端推理、OE 编译、数据集 CER 与延迟均未执行。
