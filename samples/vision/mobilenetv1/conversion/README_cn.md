[English](README.md) | 简体中文

# MobileNetV1 模型转换

本流程从固定版本的 timm 检查点重建已发布的 MobileNetV1 部署模型：导出 FP32
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
  - `100`：`timm/mobilenetv1_100.ra4_e3600_r224_in1k`，revision
    `6e88b523abded44469eafcd52f7832d5e0300b1b`
  - `125`：`timm/mobilenetv1_125.ra4_e3600_r224_in1k`，revision
    `9e2b4f9a089f39eaf59cb38fb5d595f4d78c8064`
- 对应关系：检查点即 timm 发布的 MobileNetV1-100、MobileNetV1-125
  ImageNet-1k 分类模型，权重不做修改。许可证：Apache-2.0（检查点 model card）。

<a id="preprocessing"></a>
### 预处理合同

校准、评测与 runtime 使用同一几何，跟随检查点自带的评测设置（`crop_pct`）：

| 变体 | `crop_pct` | 短边缩放到 | 裁剪 |
| --- | --- | --- | --- |
| 100 | 0.875 | 256（`int(224 / 0.875)`） | 224x224 中心裁剪 |
| 125 | 0.9 | 248（`int(224 / 0.9)`） | 224x224 中心裁剪 |

缩放为抗锯齿双三次（Pillow）。像素为 RGB；ONNX 输入 `data` 是按均值 0.5 / 标准差 0.5（MobileNetV1
检查点自带的归一化）归一化的 float32 `[1,3,224,224]`，输出 `logits` 为 `[1,1000]`。
部署模型把归一化编译进去（`mean_value` = 均值 x 255，`scale_value` =
1 / (255 x 标准差)），因此 runtime 送入由 224x224 裁剪图转换的 NV12。NV12
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
├── ptq_s_100.yaml  # 变体 100 的 S 配置（另加权重偏差校正）
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
| s100 | nash-e | `registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0` | hb_compile 3.5.3、HMCT 2.6.5、HBDK 4.7.5 | `ptq_s.yaml`（100 用 `ptq_s_100.yaml`） |
| s100p | nash-m | 与 s100 相同的镜像 | 同 s100 | `ptq_s.yaml`（100 用 `ptq_s_100.yaml`） |
| s600 | nash-p | 与 s100 相同的镜像 | 同 s100 | `ptq_s.yaml`（100 用 `ptq_s_100.yaml`） |

文档：[RDK S 工具链概览](https://developer.d-robotics.cc/rdk_doc/rdk_s/Advanced_development/toolchain_development/overview)、
[D-Robotics 工具链下载](https://toolchain.d-robotics.cc/)。

<a id="export"></a>
## 导出（ONNX）

下面的命令构建 `100`；构建 `125` 时使用 `--model v1-125` 和单独的
工作目录。`WORK` 是仓库之外的绝对路径目录。

```bash
# cwd：仓库根目录；Python 3.10，已安装锁定的依赖
export REPO=$(pwd)
export WORK=/abs/path/mobilenetv1-100
python3 utils/tools/mobilenet/workflow.py fetch --model v1-100 \
  --output $WORK/weights
python3 samples/vision/mobilenetv1/conversion/export.py --model v1-100 \
  --source-dir $WORK/weights \
  --images samples/vision/mobilenetv1/test_data/bulbul.JPEG \
           samples/vision/mobilenetv1/test_data/zebra_cls.jpg \
  --opset 11 --simplify --output $WORK/export
# 预期：$WORK/export/model.onnx（data float32 [1,3,224,224] -> logits [1,1000]）
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
python3 utils/tools/mobilenet/workflow.py calibrate --model v1-100 \
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

填充参考配置中的占位符（`SIZE` 为模型输入尺寸：224）；编译参数（优化级别、
latency 模式、单核、NV12 输入）由该配置固定：

```bash
# cwd：$WORK
export SIZE=224
sed -e "s#@WORK@#$WORK#g" -e "s#@SIZE@#$SIZE#g" \
  $REPO/samples/vision/mobilenetv1/conversion/ptq_x5.yaml > x5/config.yaml
for pair in s100:nash-e s100p:nash-m s600:nash-p; do
  target=${pair%%:*}; march=${pair##*:}
  sed -e "s#@WORK@#$WORK#g" -e "s#@TARGET@#$target#g" -e "s#@MARCH@#$march#g" \
      -e "s#@SIZE@#$SIZE#g" \
    $REPO/samples/vision/mobilenetv1/conversion/ptq_s.yaml > $target/config.yaml
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
# 预期：<target>/build/mobilenetv1_<target>.bin（x5）或 .hbm（S）
```

把每个输出重命名为发布名称（见[产物](#artifacts)）；名称中带有 march 标记：
`bayese`、`nashe`、`nashm`、`nashp`。

变体 `100` 在 S100、S100P、S600 上改用 `ptq_s_100.yaml`：它增加工具链的权重偏差校正
（`quant_config.model_config.weight.bias_correction`），构建仍为 INT8，Top-1 损失从
5.3% 降到 1.3-1.4%。X5 工具链的对应选项（`optimization: bias_correction`）反而使 X5
构建变差，因此 `100` 的已发布 X5 构建原样使用 `ptq_x5.yaml`。

<a id="validation"></a>
## 转换后验证

1. 在匹配的板卡上用 `hrt_model_exec model_info --model_file <file>` 检查制品。
   输入 shape 按目标与变体区分：

   | 目标 / 变体 | metadata 暴露的输入 | 输出 |
   | --- | --- | --- |
   | x5，100 与 125 | 一个 packed NV12 输入，224x224（`mobilenetv1_{100,125}_bayese_224x224_nv12.bin`） | F32 `[1,1000,1,1]` |
   | s100/s100p/s600，100 | Y `[1,224,224,1]`、UV `[1,112,112,2]`（`mobilenetv1_100_nash{e,m,p}_224x224_nv12.hbm`） | F32 `[1,1000]` |
   | s100/s100p/s600，125 | Y `[1,224,224,1]`、UV `[1,112,112,2]`（`mobilenetv1_125_nash{e,m,p}_224x224_nv12.hbm`） | F32 `[1,1000]` |

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
| `mobilenetv1_100_bayese_224x224_nv12.bin` | x5 | `model/` |
| `mobilenetv1_100_nashe_224x224_nv12.hbm` | s100 | `model/s100/` |
| `mobilenetv1_100_nashm_224x224_nv12.hbm` | s100p | `model/s100p/` |
| `mobilenetv1_100_nashp_224x224_nv12.hbm` | s600 | `model/s600/` |
| `mobilenetv1_125_bayese_224x224_nv12.bin` | x5 | `model/` |
| `mobilenetv1_125_nashe_224x224_nv12.hbm` | s100 | `model/s100/` |
| `mobilenetv1_125_nashm_224x224_nv12.hbm` | s100p | `model/s100p/` |
| `mobilenetv1_125_nashp_224x224_nv12.hbm` | s600 | `model/s600/` |

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
- 本流程之前发布的模型（权重、前处理不同，且输出为概率）不再随本 sample 分发。
