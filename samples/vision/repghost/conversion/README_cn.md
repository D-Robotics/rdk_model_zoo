# RepGhost 转换

<a id="source-model"></a>
## 源模型

源说明基于 PyTorch/timm RepGhost 变体，但未固定 timm/torch 版本、权重修订或权重摘要。未交付权重与可执行 ONNX 导出脚本；未重建验证与发布 bin 的对应关系。

<a id="toolchain-targets"></a>
## 工具链与目标

五份原样源 YAML 均面向 X5 `march: bayes-e`。源未指定 OE 版本，重建时必须记录实际版本。没有 S 配方。

| Variant | Config | Required ONNX | Published filename |
| --- | --- | --- | --- |
| `100` | `RepGhost_100.yaml` | `./repghostnet_100.onnx` | `RepGhost_100_224x224_nv12.bin` |
| `111` | `RepGhost_111.yaml` | `./repghostnet_111.onnx` | `RepGhost_111_224x224_nv12.bin` |
| `130` | `RepGhost_130.yaml` | `./repghostnet_130.onnx` | `RepGhost_130_224x224_nv12.bin` |
| `150` | `RepGhost_150.yaml` | `./repghostnet_150.onnx` | `RepGhost_150_224x224_nv12.bin` |
| `200` | `RepGhost_200.yaml` | `./repghostnet_200.onnx` | `RepGhost_200_224x224_nv12.bin` |

<a id="export"></a>
## ONNX 导出

检入材料不足以给出经过验证的导出命令。需准备匹配变体的 RGB NCHW ONNX（名义输入 1×3×224×224），先核对真实输入输出。YAML 的 `input_shape` 和 `input_name` 为空，实际从图读取，并未强制限定 224。按表格把图放在对应 YAML 旁。

<a id="calibration"></a>
## 校准

所有配置读取 `./calibration_data_rgb_f32`，类型 float32，校准算法 default，训练输入 RGB/NCHW、运行输入 NV12。mean 为 123.675/116.28/103.53，scale 为 0.01712475/0.017507/0.01742919。缺少校准图选择、数量、预处理脚本和产物；这属于前提缺口，不是可执行准备配方。须确认浮点数据与图和 YAML 归一化一致，不能把 NV12 目录改名冒充 RGB 校准数据。

<a id="compile"></a>
## 编译

仅在 OE 环境、补齐 ONNX 与 RGB float32 校准数据后可执行以下条件命令。本次迁移未执行；cwd 为本 conversion 目录。以 100 为例：

```bash
cd samples/vision/repghost/conversion
hb_mapper checker --model-type onnx --march bayes-e --model ./repghostnet_100.onnx
hb_mapper makertbin --model-type onnx --config RepGhost_100.yaml
```

所有配置的 working_dir 为 `RepGhost_224x224_nv12`，输出前缀也为 `RepGhost_224x224_nv12`，均不含变体。预期 bin 为 `RepGhost_224x224_nv12/RepGhost_224x224_nv12.bin`。各变体必须隔离工作目录或在下一次构建前保留输出。源编译选项为 latency/O3 并启用 dump_calibration_data，不能让多个变体并发写同一目录。

<a id="validation"></a>
## 转换后验证

板端验证 not-run。先核对 224 几何、packed NV12、squeeze 后 (1000,) 的单 F32 输出及分数语义，随后可用准确引用与独立路径运行重建 100 制品：

```bash
# cwd: repository root on X5
python3 samples/vision/repghost/runtime/python/main.py --target x5 \
  --asset-id x5:repghost:RepGhost_100_224x224_nv12.bin \
  --model-path samples/vision/repghost/conversion/RepGhost_224x224_nv12/RepGhost_224x224_nv12.bin \
  --test-img samples/vision/repghost/test_data/ibex.JPEG
```

引用只选择契约，不证明重建字节等于发布字节。单独记录重建哈希与来源，对照源实现并评估量化精度后再交付。

<a id="artifacts"></a>
## 产物

逐变体发布文件名见上表，下载落在 `samples/vision/repghost/model/`。编译在工作目录生成共同基名；移动已验证构建时须保留变体身份，单纯改名不能证明图等价或精度一致。

<a id="known-gaps"></a>
## 已知缺口

缺少：固定框架/OE 版本、权重修订/哈希、可运行导出、校准数据选择/数量/准备及转换和板端精度证据。五份 YAML 与源许可声明逐字节保留。当前可用发布模型下载路线，不声明端到端转换可复现。
