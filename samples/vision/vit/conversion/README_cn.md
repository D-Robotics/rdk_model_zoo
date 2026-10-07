# ViT 转换

<a id="source-model"></a>
## 源模型

源说明使用 PyTorch CIFAR-10 训练与 `vit_cifar10_batch1.onnx`，引用
[ViT_PyTorch](https://github.com/xiongqi123123/ViT_PyTorch.git)。未交付
固定权重版本或导出脚本。原 YAML 与 hb_compile.log 逐字节保留。

<a id="toolchain-targets"></a>
## 工具链与目标

原日志记录的构建环境为 hbdk 4.2.11 / hmct 2.4.1 / hb_compile 3.3.22。
YAML march=nash-e 面向 S100；没有 S100P/S600/X5 配置。需匹配的 x86
Linux OE 环境（[OE 资源](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview)、
[工具链手册](https://toolchain.d-robotics.cc/)）。

<a id="export"></a>
## 导出

不含导出脚本或固定权重版本。重建时需补齐训练权重、模型定义与兼容
导出环境（训练流程参考
[ViT_PyTorch](https://github.com/xiongqi123123/ViT_PyTorch.git)），并把
得到的 ONNX 放到 `conversion/vit_cifar10_batch1.onnx`。

<a id="calibration"></a>
## 校准

配方使用 50 张 CIFAR-10 校准图片，以 float32 RGB NCHW 数据放在
`calibration_data_rgb/`（目录请自行准备）。YAML mean=0.4914/0.4822/
0.4465，scale=4.943153707865546/5.014042553191489/4.975124378109453；
softmax qtype=int32。生成数据前请核对编码（NumPy header 还是原始
缓冲）、值域与 OE loader 路径：原日志从另一个绝对目录读取 raw
calibration_*.npy。

<a id="compile"></a>
## 编译

在 OE 环境内、ONNX 图与校准数据前提补齐后执行：

```bash
cd samples/vision/vit/conversion
hb_compile --config config_vit_nv12.yaml
```

配置输出：`vit_cifar10_batch1/vit_cifar10_batch1.hbm`；jobs=8、latency、O2。须检查完整日志和成功退出。

<a id="validation"></a>
## 验证

原日志记录：Y [1,224,224,1] U8、UV [1,112,112,2] U8、output [1,10]
F32。重新生成后需核对实际 metadata，再做同板数值对照和 CIFAR-10
精度评估。

<a id="artifacts"></a>
## 产物

唯一 YAML 产生不带量化后缀的 HBM；发布 int8/int16 是 [model](../model/README_cn.md) 的独立制品。该配方不能证明二者各自构建过程，改名不构成 int16 构建证据。

<a id="known-gaps"></a>
## 补充准备

重新构建 ViT 时，从链接的 CIFAR-10 训练流程与固定权重修订开始，再导出 S100 YAML 使用的 ONNX 图。准备校准值前，先核对原 YAML 与日志所要求的编码和值域。将 int8 与 int16 制品分别构建、验证为独立 S100 输出；上文 YAML 命令说明其配置输出，每个部署文件应对应自己的已发布变体身份。使用发布文件推理时，通过模型下载器准备制品。
