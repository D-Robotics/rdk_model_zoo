# ResNeXt 转换

<a id="source-model"></a>
## 源模型

ResNeXt 源引用 timm resnext50_32x4d 与 ONNX 简化，但没有可执行导出、固定包/权重版本或摘要。

<a id="toolchain-targets"></a>
## 工具链与目标

X5 march 为 bayes-e，源文件逐字节保留。OE 版本未指定。

| YAML | ONNX path (conversion cwd) | Output path |
| --- | --- | --- |
| `ResNeXt50_32x4d_config.yaml` | `./ResNeXt50_32x4d.onnx` | `ResNeXt50_32x4d_224x224_nv12/ResNeXt50_32x4d_224x224_nv12.bin` |

<a id="export"></a>
## ONNX 导出

未交付可执行导出脚本。需在 YAML 旁准备 `ResNeXt50_32x4d.onnx`，名义输入 RGB NCHW 1×3×224×224；input_shape/name 为空，从图读取。

<a id="calibration"></a>
## 校准

YAML 训练输入 RGB/NCHW、运行输入 NV12，mean 123.675/116.28/103.53，scale 0.01712475/0.017507/0.01742919。ResNeXt 使用 ./calibration_data_rgb_f32、float32、校准 default，但没有准备脚本/数量/数据。

<a id="compile"></a>
## 编译

仅在图与校准前提验证后于 OE 执行（本次未运行）：

```bash
# cwd: repository root
cd samples/vision/resnext/conversion
hb_mapper checker --model-type onnx --march bayes-e --model ./ResNeXt50_32x4d.onnx
hb_mapper makertbin --model-type onnx --config ResNeXt50_32x4d_config.yaml
```

保留 latency/O3。输出基名与发布文件名一致，但名称一致不能证明图、字节或精度相同。

<a id="validation"></a>
## 转换后验证

板端验证 not-run。核对实际 metadata、224×224 packed NV12 与 squeeze 后 1000 分数的单 F32 输出。重建模型可用准确契约引用和外部路径选择，其哈希/来源必须与发布制品分开记录。

```bash
# cwd: repository root on X5; published-artifact smoke
bash samples/vision/resnext/model/download.sh x5 50_32x4d
python3 samples/vision/resnext/runtime/python/main.py --target x5 --variant 50_32x4d
```

<a id="artifacts"></a>
## 产物

发布制品及落地路径见[模型准备](../model/README_cn.md#artifacts)。本次迁移不新增 ONNX/权重/编译产物。

<a id="known-gaps"></a>
## 已知缺口

没有产生转换/精度/板端证据。导出与校准可用性按上述真实源材料声明，不由模型名称推断。重建前固定版本、权重与数据集输入。
