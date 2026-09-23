# ViT 转换

<a id="source-model"></a>
## 源模型

源说明使用 PyTorch CIFAR-10 训练与 `vit_cifar10_batch1.onnx`，引用 [ViT_PyTorch](https://github.com/xiongqi123123/ViT_PyTorch.git)。未交付固定权重版本或导出脚本。原 YAML 与 hb_compile.log 逐字节保留。

<a id="toolchain-targets"></a>
## 工具链与目标

历史日志记录 hbdk 4.2.11 / hmct 2.4.1 / hb_compile 3.3.22，并非新验证容器。YAML march=nash-e 面向 S100；没有 S100P/S600/X5 配置。需匹配的 x86 Linux OE 环境；[OE 资源](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview)。

<a id="export"></a>
## 导出

仅凭已交付文件无法重现：需先补齐准确训练权重、模型定义、兼容导出环境及 ONNX，放到 `conversion/vit_cifar10_batch1.onnx`。不把未验证通用命令当可运行导出配方。

<a id="calibration"></a>
## 校准

源记载 50 张 CIFAR-10 校准图片，float32 RGB NCHW 数据放 `calibration_data_rgb/`，但不含准备脚本和数据。YAML mean=0.4914/0.4822/0.4465，scale=4.943153707865546/5.014042553191489/4.975124378109453；softmax qtype=int32。生成前需核对编码、值域和 OE loader：历史日志从另一个绝对目录读取 raw calibration_*.npy，扩展名不足以证明带 NumPy header 还是原始缓冲。

<a id="compile"></a>
## 编译

以下命令以导出、校准缺失前提补齐及匹配 OE 环境为条件；本轮未执行。

```bash
cd samples/vision/vit/conversion
hb_compile --config config_vit_nv12.yaml
```

配置输出：`vit_cifar10_batch1/vit_cifar10_batch1.hbm`；jobs=8、latency、O2。须检查完整日志和成功退出。

<a id="validation"></a>
## 验证

历史日志：Y [1,224,224,1] U8、UV [1,112,112,2] U8、output [1,10] F32。重新生成后需核对实际 metadata，再做同板源/统一数值对照和 CIFAR-10 精度；本轮转换及板测 not-run。

<a id="artifacts"></a>
## 产物

唯一 YAML 产生不带量化后缀的 HBM；发布 int8/int16 是 [model](../model/README_cn.md) 的独立制品。该配方不能证明二者各自构建过程，改名不构成 int16 构建证据。

<a id="known-gaps"></a>
## 缺失项

缺固定权重/导出版本、校准准备/数据编码、int8/int16 配方对应关系、当前工具链/容器验证及重新生成精度。历史日志路径差异保留，不重写证据。现阶段使用发布制品推理，板端验证待补。
