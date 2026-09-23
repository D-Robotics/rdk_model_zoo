# GoogLeNet 转换

<a id="source-model"></a>
## 源模型

源提供已发布 X5 部署 bin 及其 wrapper，没有 ONNX 图、固定权重修订、框架版本、导出脚本或 PTQ YAML。

<a id="toolchain-targets"></a>
## 工具链与目标

发布目标为 X5，未提供编译配置与 OE 版本，也无 S 目标配方。

<a id="export"></a>
## ONNX 导出

源没有可运行导出命令。wrapper 预期 NV12 打包前为名义 RGB/NCHW 224×224、输出为 1000 类分数；该运行假设不等于训练/导出配方。

<a id="calibration"></a>
## 校准

未交付校准准备、数据选择/数量、归一化配方或数据文件。准备校准数据前须取得真实模型图与匹配量化配方。

<a id="compile"></a>
## 编译

缺少 ONNX 与 PTQ 配置，无法给出已验证编译命令。当前使用 model/README 中的发布制品路线，不提供占位 YAML 或虚构编译成功。

<a id="validation"></a>
## 转换后验证

板端验证 not-run。核对实际 metadata、224×224 packed NV12 与 squeeze 后 1000 分数的单 F32 输出。重建模型可用准确契约引用和外部路径选择，其哈希/来源必须与发布制品分开记录。

```bash
# cwd: repository root on X5; published-artifact smoke
bash samples/vision/googlenet/model/download.sh x5 googlenet
python3 samples/vision/googlenet/runtime/python/main.py --target x5 --variant googlenet
```

<a id="artifacts"></a>
## 产物

发布制品及落地路径见[模型准备](../model/README_cn.md#artifacts)。本次迁移不新增 ONNX/权重/编译产物。

<a id="known-gaps"></a>
## 已知缺口

没有产生转换/精度/板端证据。导出与校准可用性按上述真实源材料声明，不由模型名称推断。重建前固定版本、权重与数据集输入。
