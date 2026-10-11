[English](README.md) | 简体中文

# MobileNetV4 模型转换

本流程从固定版本的 timm 检查点重建已发布的 MobileNetV4 部署模型：导出 FP32
ONNX、INT8 训练后量化（PTQ），并为每个 RDK 目标编译。它在带 Docker 的 x86
Linux 主机上执行，不是板卡操作。导出、校准和配置步骤使用公共
[主机流程](../../../../utils/tools/mobilenet/README_cn.md)；每次编译在对应的
OpenExplorer（OE）工具链容器中运行。

<a id="source-model"></a>
## 源模型

- 框架：PyTorch 2.8.0 与 timm 1.0.20（版本固定在
  `utils/tools/mobilenet/requirements.lock.txt`）。
- 权重，按 Hugging Face revision 与文件 SHA-256 固定在
  `utils/tools/mobilenet/checkpoints.json`：
  - `small`：`timm/mobilenetv4_conv_small.e2400_r224_in1k`，revision
    `331fb803779522b685cf942e15f914fb6741c1eb`
  - `medium`：`timm/mobilenetv4_conv_medium.e500_r224_in1k`，revision
    `02a09fbfb82b289e871ba8255f9da58c056fb13b`
  - `large`：`timm/mobilenetv4_conv_large.e500_r256_in1k`，revision
    `033de0bd74b5261f7339c5de242a2e4785808aa3`
- 对应关系：检查点即 timm 发布的 MobileNetV4-Conv-Small、MobileNetV4-Conv-Medium、MobileNetV4-Conv-Large
  ImageNet-1k 分类模型，权重不做修改。许可证：Apache-2.0（检查点 model card）。

<a id="preprocessing"></a>
### 预处理合同

校准、评测与 runtime 使用同一几何，跟随检查点自带的评测设置（`crop_pct`）：

| 变体 | `crop_pct` | 短边缩放到 | 裁剪 |
| --- | --- | --- | --- |
| small | 0.875 | 256（`int(224 / 0.875)`） | 224x224 中心裁剪 |
| medium | 0.95 | 235（`int(224 / 0.95)`） | 224x224 中心裁剪 |
| large | 0.95 | 269（`int(256 / 0.95)`） | 256x256 中心裁剪 |

缩放为抗锯齿双三次（Pillow）。像素为 RGB；ONNX 输入 `data` 是按 ImageNet
均值/标准差归一化的 float32 `[1,3,S,S]`（S = 224 或 256），输出 `logits` 为 `[1,1000]`。
部署模型把归一化编译进去（`mean_value` = 均值 x 255，`scale_value` =
1 / (255 x 标准差)），因此 runtime 送入由 SxS 裁剪图转换的 NV12。NV12
转换使用 OpenCV 的有限范围 BT.601 变换，所以两种配置都设置
`input_space_and_range: bt601_video`。

<a id="directory"></a>
## 目录结构

```text
conversion/
├── README.md  # 英文说明
├── README_cn.md  # 中文说明
├── export.py  # 固定版本 timm 检查点 -> 已校验的 FP32 ONNX
├── ptq_s.yaml  # S100 / S100P / S600 编译配置（含占位符）
└── ptq_x5.yaml  # X5 编译配置（含占位符）
```

校准输入与 OE 配置由 `utils/tools/mobilenet/workflow.py` 生成；`ptq_*.yaml`
是已发布制品使用的参考编译配置。

<a id="toolchain-targets"></a>
## 工具链与目标

使用与目标匹配的 OE 容器编译，并把镜像 tag 与 digest 和该次构建一起保存。
已发布制品所用版本：

| 目标 | march | 工具链镜像 | 工具 | 配置 |
| --- | --- | --- | --- | --- |
| x5 | bayes-e | `registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_20_x5_gpu:v1.2.8` | hb_mapper 1.24.3、HBDK 3.49.15、horizon_nn 1.1.0 | `ptq_x5.yaml` |
| s100 | nash-e | `registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0` | hb_compile 3.5.3、HMCT 2.6.5、HBDK 4.7.5 | `ptq_s.yaml` |
| s100p | nash-m | 与 s100 相同的镜像 | 同 s100 | `ptq_s.yaml` |
| s600 | nash-p | 与 s100 相同的镜像 | 同 s100 | `ptq_s.yaml` |

文档：[RDK S 工具链概览](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview)、
[D-Robotics 工具链下载](https://toolchain.d-robotics.cc/)。

<a id="export"></a>
## 导出（ONNX）

下面的命令构建 `small`；构建 `medium`、`large` 时使用 `--model v4-medium-224`、`--model v4-large-256` 和单独的
工作目录。`WORK` 是仓库之外的绝对路径目录。

```bash
# cwd：仓库根目录；Python 3.10，已安装锁定的依赖
export REPO=$(pwd)
export WORK=/abs/path/mobilenetv4-small
python3 utils/tools/mobilenet/workflow.py fetch --model v4-small \
  --output $WORK/weights
python3 samples/vision/mobilenetv4/conversion/export.py --model v4-small \
  --source-dir $WORK/weights \
  --images samples/vision/mobilenetv4/test_data/great_grey_owl.JPEG \
           samples/vision/mobilenetv4/test_data/zebra_cls.jpg \
  --opset 11 --simplify --output $WORK/export
# 预期：$WORK/export/model.onnx（data float32 [1,3,224,224] -> logits [1,1000]；
#       large 为 [1,3,256,256]）
#       以及 status 为 "passed" 的 export.json
```

`export.py` 校验检查点哈希，严格加载权重，简化计算图，并用真实图像把 logits
与 Top-5 同 PyTorch 对比。`export.json` 记录每张图像的最大绝对误差，并把 ONNX
的 SHA-256 与检查点、预处理合同和依赖版本绑定。

<a id="calibration"></a>
## 校准

- 数据集：COCO 2017 train2017 中固定的 200 张图像子集（选取种子 42），
  在 manifest 中逐张记录 SHA-256。该集合与评测集没有共同图像，
  `workflow.py calibrate` 会校验这一点。
- 预处理：上述合同，并按目标写出：X5 使用 [0,255] 的原始 RGB float32
  （`.rgb`，无头小端；RGB/NV12 转换由 X5 加载器完成）；S 系列使用归一化后的
  ONNX 输入，保存为 float32 `.npy`。

```bash
# cwd：仓库根目录；对每个目标各执行一次：x5、s100、s100p、s600
python3 utils/tools/mobilenet/workflow.py calibrate --model v4-small \
  --platform x5 \
  --manifest /abs/calibration/manifest.json \
  --images-root /abs/calibration \
  --expected-images 200 \
  --evaluation-manifest /abs/imagenetv2/manifest.json \
  --output $WORK/x5/calibration
# 预期：$WORK/x5/calibration/data（200 个文件）与 calibration.json
```

<a id="compile"></a>
## 编译

填充参考配置中的占位符（`SIZE` 为模型输入尺寸：small 为 224，medium 为 224，large 为 256）；编译参数（优化级别、
latency 模式、单核、NV12 输入）由该配置固定：

```bash
# cwd：$WORK
export SIZE=224
sed -e "s#@WORK@#$WORK#g" -e "s#@SIZE@#$SIZE#g" \
  $REPO/samples/vision/mobilenetv4/conversion/ptq_x5.yaml > x5/config.yaml
for pair in s100:nash-e s100p:nash-m s600:nash-p; do
  target=${pair%%:*}; march=${pair##*:}
  sed -e "s#@WORK@#$WORK#g" -e "s#@TARGET@#$target#g" -e "s#@MARCH@#$march#g" \
      -e "s#@SIZE@#$SIZE#g" \
    $REPO/samples/vision/mobilenetv4/conversion/ptq_s.yaml > $target/config.yaml
done
```

把 `$WORK` 以相同的绝对路径挂载进容器，并在容器内运行工具链，例如 X5：

```bash
# cwd：$WORK/x5；已发布构建使用 --shm-size 15g（X5），
# S 系列镜像另外使用 --gpus all
docker run --rm --user "$(id -u):$(id -g)" --shm-size 15g \
  -v "$WORK:$WORK" -w "$WORK/x5" \
  registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_20_x5_gpu:v1.2.8 \
  bash -c "hb_mapper checker --model-type onnx --march bayes-e \
             --model $WORK/export/model.onnx --input-shape data 1x3x${SIZE}x${SIZE} && \
           hb_mapper makertbin --config config.yaml --model-type onnx"

# S100、S100P、S600：cwd 为 $WORK/<target>，镜像 ..._s100_s600_gpu:v3.7.0
#   hb_compile --config config.yaml
# 预期：<target>/build/mobilenetv4_<target>.bin（x5）或 .hbm（S）
```

把每个输出重命名为发布名称（见[产物](#artifacts)）；名称中带有 march 标记：
`bayese`、`nashe`、`nashm`、`nashp`。

<a id="validation"></a>
## 转换后验证

1. 在匹配的板卡上用 `hrt_model_exec model_info --model_file <file>` 检查制品。
   输入 shape 按目标与变体区分：

   | 目标 / 变体 | metadata 暴露的输入 | 输出 |
   | --- | --- | --- |
   | x5，small 与 medium | 一个 packed NV12 输入，224x224（`mobilenetv4_{conv_small,conv_medium}_bayese_224x224_nv12.bin`） | F32 `[1,1000,1,1]` |
   | x5，large | 一个 packed NV12 输入，256x256（`mobilenetv4_conv_large_bayese_256x256_nv12.bin`） | F32 `[1,1000,1,1]` |
   | s100/s100p/s600，small | Y `[1,224,224,1]`、UV `[1,112,112,2]`（`mobilenetv4_conv_small_nash{e,m,p}_224x224_nv12.hbm`） | F32 `[1,1000]` |
   | s100/s100p/s600，medium | Y `[1,224,224,1]`、UV `[1,112,112,2]`（`mobilenetv4_conv_medium_nash{e,m,p}_224x224_nv12.hbm`） | F32 `[1,1000]` |
   | s100/s100p/s600，large | Y `[1,256,256,1]`、UV `[1,128,128,2]`（`mobilenetv4_conv_large_nash{e,m,p}_256x256_nv12.hbm`） | F32 `[1,1000]` |

   输出为原始 logits，由 runtime 任务施加 softmax。
2. 用新制品执行 sample 快速开始做冒烟测试：退出码 0 并打印 Top-5 列表
   （见[预期结果](../README_cn.md#expected-results)）。
3. 用[评测器](../evaluator/README_cn.md)测量全量精度，并与同一检查点的 FP32
   ONNX 基线比较；已发布结果与验收规则见
   [sample README](../README_cn.md#performance)。

<a id="artifacts"></a>
## 产物

| 产物 | 目标 | 落盘位置 |
| --- | --- | --- |
| `mobilenetv4_conv_small_bayese_224x224_nv12.bin` | x5 | `model/` |
| `mobilenetv4_conv_small_nashe_224x224_nv12.hbm` | s100 | `model/s100/` |
| `mobilenetv4_conv_small_nashm_224x224_nv12.hbm` | s100p | `model/s100p/` |
| `mobilenetv4_conv_small_nashp_224x224_nv12.hbm` | s600 | `model/s600/` |
| `mobilenetv4_conv_medium_bayese_224x224_nv12.bin` | x5 | `model/` |
| `mobilenetv4_conv_medium_nashe_224x224_nv12.hbm` | s100 | `model/s100/` |
| `mobilenetv4_conv_medium_nashm_224x224_nv12.hbm` | s100p | `model/s100p/` |
| `mobilenetv4_conv_medium_nashp_224x224_nv12.hbm` | s600 | `model/s600/` |
| `mobilenetv4_conv_large_bayese_256x256_nv12.bin` | x5 | `model/` |
| `mobilenetv4_conv_large_nashe_256x256_nv12.hbm` | s100 | `model/s100/` |
| `mobilenetv4_conv_large_nashm_256x256_nv12.hbm` | s100p | `model/s100p/` |
| `mobilenetv4_conv_large_nashp_256x256_nv12.hbm` | s600 | `model/s600/` |

清单与 [model/README_cn.md](../model/README_cn.md) 一致。

<a id="known-gaps"></a>
## 补充准备

- 校准与评测数据是外部输入。COCO train2017 子集和 ImageNetV2 MatchedFrequency
  集合（10,000 张）需向其所有者获取；manifest 格式及其检查见
  [主机流程](../../../../utils/tools/mobilenet/README_cn.md)。
- OE 工具链镜像需要访问 `registry.d-robotics.cc`；请用自己的凭据执行
  `docker login`。凭据不得写入这些文件。
- 编译某个目标不需要板卡，但验证需要：使用与目标匹配的板卡（S100P 与 S600
  是不同于 S100 的独立板卡）。
- 本流程之前发布的模型（权重不同，且 S medium 为 256x256 输入）不再随本 sample
  分发。Conv-Large 只以其 256x256 训练分辨率发布。
