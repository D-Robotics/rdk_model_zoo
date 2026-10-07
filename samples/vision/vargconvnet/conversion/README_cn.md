# VargConvNet 转换

<a id="source-model"></a>
## 源模型

推理时使用模型指南提供的已发布 X5 部署 bin 与 wrapper。重新构建需先
准备 ONNX 图、权重修订、框架版本及 PTQ YAML。

<a id="toolchain-targets"></a>
## 工具链与目标

在匹配目标的 x86 Linux OE 环境中构建 X5 模型，并按所用 OE 版本配置编译工具链。


工具链资源:

- [OE Docker environment](https://forum.d-robotics.cc/t/topic/35229)

<a id="export"></a>
## ONNX 导出

导出的 ONNX 图需满足 wrapper 的运行时 I/O 契约：NV12 打包前为
RGB/NCHW 224×224，输出为 1000 类分数。

<a id="calibration"></a>
## 校准

取得所选模型图和匹配量化配方后，按该图的预处理准备校准数据。

<a id="compile"></a>
## 编译

为模型图与校准数据创建匹配的 PTQ 配置，再使用 X5 OE 工具链编译。
推理时可使用 model/README 中的已发布制品。

<a id="validation"></a>
## 转换后验证

核对实际 metadata、224×224 packed NV12 与 squeeze 后 1000 分数的单 F32 输出。重建模型可用准确契约引用和外部路径选择，其哈希/来源必须与发布制品分开记录。

```bash
# cwd: repository root on X5; published-artifact smoke
bash samples/vision/vargconvnet/model/download.sh x5 vargconvnet
python3 samples/vision/vargconvnet/runtime/python/main.py --target x5 --variant vargconvnet
```

<a id="artifacts"></a>
## 产物

发布制品及落地路径见[模型准备](../model/README_cn.md#artifacts)。

<a id="known-gaps"></a>
## 补充准备

推理时按 [model/README_cn.md](../model/README_cn.md) 准备已发布 X5 制品。构建替代模型需提供符合 wrapper 契约的 ONNX 图（名义 RGB/NCHW 224×224 输入、1,000 类分类分数）、匹配的 PTQ 配置，以及按模型归一化处理的校准数据。使用 X5 OE 工具链，并通过样例运行时检查编译结果。
