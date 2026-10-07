# HGNetV2 转换

<a id="source-model"></a>
## 源模型

五个导出脚本加载 `timm` 的 PP-HGNetV2 预训练权重
（`hgnetv2_b0.ssld_stage2_ft_in1k` … `hgnetv2_b4.ssld_stage2_ft_in1k`）。
交付脚本标注 torch 1.13 与 OE Docker v1.2.8；脚本注释标注
`hb_mapper` 1.24.3、opset 11。timm 与权重版本未固定。首次运行会从
Hugging Face 下载权重；推理本身不执行任何导出。

| 变体 | timm 模型 id | 输出 `.bin` |
| --- | --- | --- |
| b0 | `hgnetv2_b0.ssld_stage2_ft_in1k` | `hgnetv2_b0_224x224_nv12.bin` |
| b1 | `hgnetv2_b1.ssld_stage2_ft_in1k` | `hgnetv2_b1_224x224_nv12.bin` |
| b2 | `hgnetv2_b2.ssld_stage2_ft_in1k` | `hgnetv2_b2_224x224_nv12.bin` |
| b3 | `hgnetv2_b3.ssld_stage2_ft_in1k` | `hgnetv2_b3_224x224_nv12.bin` |
| b4 | `hgnetv2_b4.ssld_stage2_ft_in1k` | `hgnetv2_b4_224x224_nv12.bin` |

<a id="toolchain-targets"></a>
## 工具链与目标

模型转换在 x86 Linux 主机上执行（X5，march `bayes-e`），从不在板卡上
运行。请安装 RDK X5 OpenExplorer 工具链 **1.2.8**：

```bash
# 下载并加载离线 Docker 镜像
wget https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
docker load -i docker_openexplorer_ubuntu_20_x5_cpu_v1.2.8.tar.gz
```

或前往地瓜开发者社区获取离线 Docker 镜像
（[topic 35229](https://forum.d-robotics.cc/t/topic/35229)）。启动容器时
挂载仓库以共享工作目录：

```bash
# 将 /path/to/rdk_model_zoo 替换为你的检出路径
docker run -it --rm \
  -v /path/to/rdk_model_zoo:/data \
  openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 /bin/bash
```

在容器内（或任意带 PyTorch ≥ 1.13 的 Python 3 环境）安装导出依赖：
`pip install timm`。网络受限时，在启动导出脚本前设置国内镜像：
`export HF_ENDPOINT=https://hf-mirror.com`。

| YAML | ONNX 路径（conversion cwd） | 输出路径 |
| --- | --- | --- |
| `hgnetv2_b0.yaml` | `./onnx_export/hgnetv2_b0.onnx` | `hgnetv2_b0_224x224_nv12/hgnetv2_b0_224x224_nv12.bin` |
| `hgnetv2_b1.yaml` | `./onnx_export/hgnetv2_b1.onnx` | `hgnetv2_b1_224x224_nv12/hgnetv2_b1_224x224_nv12.bin` |
| `hgnetv2_b2.yaml` | `./onnx_export/hgnetv2_b2.onnx` | `hgnetv2_b2_224x224_nv12/hgnetv2_b2_224x224_nv12.bin` |
| `hgnetv2_b3.yaml` | `./onnx_export/hgnetv2_b3.onnx` | `hgnetv2_b3_224x224_nv12/hgnetv2_b3_224x224_nv12.bin` |
| `hgnetv2_b4.yaml` | `./onnx_export/hgnetv2_b4.onnx` | `hgnetv2_b4_224x224_nv12/hgnetv2_b4_224x224_nv12.bin` |

<a id="export"></a>
## ONNX 导出

在 OE/PyTorch 环境内执行逐变体导出脚本。请在 `onnx_export/` 下运行，
产物路径才能与 YAML 匹配；其他变体将 `b0` 换为 `b1`/`b2`/`b3`/`b4`。
脚本使用输入名 `input`、输出名 `output`、形状 1×3×224×224、opset 11。

```bash
# cwd：仓库根目录
cd samples/vision/hgnetv2/conversion/onnx_export
python3 export_hgnetv2_b0_bpu.py
# 输出：本目录下的 hgnetv2_b0.onnx
```

<a id="calibration"></a>
## 校准

`hb_mapper` 需要 20–50 张代表性的 ImageNet 风格图片做 INT8 量化校准。
YAML 设置 `cal_data_dir: '../cal_data'`、`cal_data_type: float32`、
`preprocess_on: true`——加载器会按 YAML 归一化（训练输入 RGB/NCHW，
mean `123.675/116.28/103.53`，scale `0.01712475/0.017507/0.01742919`）
自行处理这些 JPEG。不含准备脚本；请在 `conversion/` 同级自建目录并
放入图片：

```bash
# cwd：samples/vision/hgnetv2/conversion
mkdir -p ../cal_data
# 拷贝 20–50 张 ImageNet val 风格的 JPEG 图片到该目录
```

<a id="compile"></a>
## 编译

在 OE 环境内、ONNX 图与校准数据就绪后执行：

```bash
# cwd：仓库根目录
cd samples/vision/hgnetv2/conversion
hb_mapper checker --model-type onnx --march bayes-e --model ./onnx_export/hgnetv2_b0.onnx
hb_mapper makertbin --model-type onnx --config hgnetv2_b0.yaml
```

YAML 保持 `compile_mode: latency` / `optimize_level: O3`。产物
`hgnetv2_b0_224x224_nv12.bin` 写入 `hgnetv2_b0_224x224_nv12/` 子目录；
把它拷贝或软链到 `../model/`，运行时即可直接使用：

```bash
# cwd：samples/vision/hgnetv2/conversion
cp hgnetv2_b0_224x224_nv12/hgnetv2_b0_224x224_nv12.bin ../model/
```

输出文件名与已发布文件名一致。

<a id="validation"></a>
## 转换后验证

运行时协议为 224×224 packed NV12 输入、squeeze 后 1000 分数的单 F32
输出。重建模型可用准确契约引用加外部路径选择；其哈希/来源必须与
发布制品分开记录。发布制品的功能检查：

```bash
# cwd：仓库根目录（X5 板卡）
bash samples/vision/hgnetv2/model/download.sh x5 b0
python3 samples/vision/hgnetv2/runtime/python/main.py --target x5 --variant b0
```

<a id="artifacts"></a>
## 产物

发布制品及落地路径见[模型准备](../model/README_cn.md#artifacts)。

<a id="known-gaps"></a>
## 补充准备

导出脚本运行时会获取 `hgnetv2_b*.ssld_stage2_ft_in1k`。构建时记录 timm 版本与权重哈希。准备 20–50 张 ImageNet 风格 JPEG 作为校准数据，再按上文逐制品 YAML 和编译命令操作。保持各已发布输出文件名、目标与配置一一对应。
