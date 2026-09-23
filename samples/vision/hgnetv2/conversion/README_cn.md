# HGNetV2 转换

<a id="source-model"></a>
## 源模型

五份原始导出脚本加载 timm hgnetv2_b0…b4.ssld_stage2_ft_in1k 预训练权重。源指定 torch 1.13 与 OE Docker v1.2.8，脚本注释提及 hb_mapper 1.24.3/opset 11；timm 与远程权重修订/哈希未固定。首次执行可能下载权重，推理不会执行导出。

<a id="toolchain-targets"></a>
## 工具链与目标

X5 march 为 bayes-e，源文件逐字节保留。本次未重跑源 v1.2.8 环境。

| YAML | ONNX path (conversion cwd) | Output path |
| --- | --- | --- |
| `hgnetv2_b0.yaml` | `./onnx_export/hgnetv2_b0.onnx` | `hgnetv2_b0_224x224_nv12/hgnetv2_b0_224x224_nv12.bin` |
| `hgnetv2_b1.yaml` | `./onnx_export/hgnetv2_b1.onnx` | `hgnetv2_b1_224x224_nv12/hgnetv2_b1_224x224_nv12.bin` |
| `hgnetv2_b2.yaml` | `./onnx_export/hgnetv2_b2.onnx` | `hgnetv2_b2_224x224_nv12/hgnetv2_b2_224x224_nv12.bin` |
| `hgnetv2_b3.yaml` | `./onnx_export/hgnetv2_b3.onnx` | `hgnetv2_b3_224x224_nv12/hgnetv2_b3_224x224_nv12.bin` |
| `hgnetv2_b4.yaml` | `./onnx_export/hgnetv2_b4.onnx` | `hgnetv2_b4_224x224_nv12/hgnetv2_b4_224x224_nv12.bin` |

<a id="export"></a>
## ONNX 导出

在源 OE/PyTorch 环境中的条件导出命令；从 onnx_export 执行使产物匹配 YAML 路径。其他脚本将 b0 换为 b1/b2/b3/b4。输入名 input、输出名 output、shape 1×3×224×224、opset 11。本次迁移未执行。

```bash
# cwd: repository root
cd samples/vision/hgnetv2/conversion/onnx_export
python3 export_hgnetv2_b0_bpu.py
# output: hgnetv2_b0.onnx in this directory
```

<a id="calibration"></a>
## 校准

YAML 设置 cal_data_dir ../cal_data、cal_data_type float32、preprocess_on true；训练输入 RGB/NCHW，mean 123.675/116.28/103.53、scale 0.01712475/0.017507/0.01742919。旧 README 建议复制 20–50 张 JPEG，但没有准备脚本或转换收据证明 JPEG 如何满足 float32/preprocess_on 组合。该处作为未解决前提：校准前核对准确 OE 加载与归一化语义，不能声称复制或改名图片就完成了校准。conversion cwd 下路径解析为 samples/vision/hgnetv2/cal_data。

<a id="compile"></a>
## 编译

仅在图与校准前提验证后于 OE 执行（本次未运行）：

```bash
# cwd: repository root
cd samples/vision/hgnetv2/conversion
hb_mapper checker --model-type onnx --march bayes-e --model ./onnx_export/hgnetv2_b0.onnx
hb_mapper makertbin --model-type onnx --config hgnetv2_b0.yaml
```

保留 latency/O3。输出基名与发布文件名一致，但名称一致不能证明图、字节或精度相同。

<a id="validation"></a>
## 转换后验证

板端验证 not-run。核对实际 metadata、224×224 packed NV12 与 squeeze 后 1000 分数的单 F32 输出。重建模型可用准确契约引用和外部路径选择，其哈希/来源必须与发布制品分开记录。

```bash
# cwd: repository root on X5; published-artifact smoke
bash samples/vision/hgnetv2/model/download.sh x5 b0
python3 samples/vision/hgnetv2/runtime/python/main.py --target x5 --variant b0
```

<a id="artifacts"></a>
## 产物

发布制品及落地路径见[模型准备](../model/README_cn.md#artifacts)。本次迁移不新增 ONNX/权重/编译产物。

<a id="known-gaps"></a>
## 已知缺口

没有产生转换/精度/板端证据。导出与校准可用性按上述真实源材料声明，不由模型名称推断。重建前固定版本、权重与数据集输入。
