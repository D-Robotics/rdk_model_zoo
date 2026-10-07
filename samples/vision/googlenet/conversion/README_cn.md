# GoogLeNet 转换

<a id="source-model"></a>
## 源模型

推理时按模型指南准备已发布的 X5 部署 bin
（`googlenet_224x224_nv12.bin`）。重新构建时先准备匹配的 ONNX 图、
权重修订与 PTQ YAML。

<a id="toolchain-targets"></a>
## 工具链与目标

在 x86 Linux 主机的 OpenExplorer Docker 或对应 OE 包编译环境中为
X5 构建模型。先准备 ONNX 图与 PTQ 配置，再使用 OE 包提供的
`hb_mapper checker`、`hb_mapper makertbin`、`hb_perf`、`hrt_model_exec`。
离线 Docker 镜像可从地瓜机器人开发者
论坛（[topic 35229](https://forum.d-robotics.cc/t/topic/35229)）获取。

<a id="export"></a>
## ONNX 导出

导出的 GoogLeNet ONNX 图需满足 wrapper 的运行时契约：NV12 打包前的
RGB/NCHW 224×224 输入，以及 1000 类分数输出。

<a id="calibration"></a>
## 校准

针对所选模型图和量化配方准备校准图像及预处理数值，再运行 PTQ。

<a id="compile"></a>
## 编译

准备符合运行时契约的 ONNX 图与 PTQ 配置（见
[补充准备](#known-gaps)），再使用上文 OE 编译命令。推理请使用
[model/README_cn.md](../model/README_cn.md) 的发布制品路线。

<a id="validation"></a>
## 转换后验证

重建模型的运行时协议为：NV12 打包前 `1x3x224x224` 输入、
ImageNet-1k 分类 logits 输出（squeeze 后 1000 分数的单 F32 向量）。
重建模型可用准确契约引用加外部 `--model-path` 选择；其哈希/来源必须
与发布制品分开记录。发布制品的功能检查：

```bash
# cwd: repository root on X5
bash samples/vision/googlenet/model/download.sh x5 googlenet
python3 samples/vision/googlenet/runtime/python/main.py --target x5 --variant googlenet
```

<a id="artifacts"></a>
## 产物

发布制品及落地路径见[模型准备](../model/README_cn.md#artifacts)。

<a id="known-gaps"></a>
## 补充准备

推理时按 [model/download.sh](../model/README_cn.md) 准备已发布的 X5 模型。构建替代制品时，先准备符合[源模型](#source-model)运行时契约的 ONNX 图、匹配的校准数据与归一化配置，并为 X5 OE 工具链配置 PTQ YAML。编译后使用样例运行时运行该制品。
