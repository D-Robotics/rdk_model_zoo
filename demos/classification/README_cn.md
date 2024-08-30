[English](./README.md) | 简体中文

# Classification

- [Classification](#classification)
  - [1. 模型分类简介](#1-模型分类简介)
  - [2. 模型下载地址](#2-模型下载地址)
  - [3. 输入输出数据](#3-输入输出数据)
  - [4. PTQ量化流程](#4-PTQ量化流程)
    - [4.1. 模型准备](#41-模型准备)
    - [4.2. 模型检查](#42-模型检查)
    - [4.3. 校准集构建](#43-校准集构建)
    - [4.4. 模型量化与编译](#44-模型量化与编译)
    - [4.5. 模型推理](#45-模型推理)
  - [5. 模型性能数据](#5-模型性能数据)

## 1. 模型分类简介

模型分类指的是将机器学习模型对**输入数据进行分类**的过程。具体来说，分类模型的任务是将输入数据（例如图像、文本、音频等）分配到预定义的类别中。这些类别通常是离散的标签，例如将图像分为“猫”或“狗”，或将电子邮件分类为“垃圾邮件”或“非垃圾邮件”。

在机器学习中，分类模型的训练过程包括以下几个步骤：

- **数据采集**：此过程包含图片采集和对应标签的标注。
- **数据预处理**：对数据进行清理和转换，以适应模型的输入要求（如图像大小裁剪缩放）。
- **特征提取**：从原始数据中提取有用的特征，通常是为了将数据转换为适合模型处理的格式（如将模型从 bgr 转换为 rgb，NCHW 转换为 NHWC 的通道排序）。
- **模型训练**：使用标记的数据训练分类模型，调整模型训练参数提高模型的检测精度和置信度。
- **模型验证**：使用验证集数据来评估模型的性能，并对模型进行微调和优化。
- **模型测试**：在测试集上评估模型的最终性能，确定其泛化能力。
- **部署和应用**：将训练好的模型进行模型量化和边缘侧板端部署。

深度学习的分类模型一般应用于将图片按其标签进行类别分类，最著名的图片分类挑战是 ImageNet Classification。 ImageNet Classification 是深度学习领域中的重要图像分类挑战，该数据集由斯坦福大学教授李飞飞及其团队于2007年构建，并于2009年在CVPR上发布。ImageNet 数据集广泛用于图像分类、检测和定位任务。

ILSVRC（ImageNet Large-Scale Visual Recognition Challenge）是基于 ImageNet 数据集的比赛，首次举办于2010年，直到2017年结束。该比赛包括图像分类、目标定位、目标检测、视频目标检测和场景分类。历届优胜者包括著名的深度学习模型，如 AlexNet、VGG、GoogLeNet 和 ResNet。ILSVRC2012 数据集是最常用的子集，包含1000个分类，每个分类约有1000张图片，总计约120万张训练图片，另外还有5万张验证图片和10万张测试图片（测试集没有标签）。

ImageNet 官方下载地址：https://image-net.org/

![](../../data/ImageNet.png)

由于该仓库提供的模型均是预训练模型进行模型转换后得到的 onnx/bin 文件，故无需再进行模型训练操作（缺乏训练资源也是一个很大的问题），由于数据集非常大，使用 ImageNet 数据集是作为后续模型量化操作中校准数据集的构建，下表是 ImageNet ILSVRC2012 数据集的大小情况：


| 数据集类型          | 类别   | 图片数量    |
| -------------- | ---- | ------- |
| ILSVRC2012 训练集 | 1000 | 120万张图片 |
| ILSVRC2012 验证集 | 1000 | 5万张图片   |
| ILSVRC2012 测试集 | 1000 | 10万张图片  |

ILSVRC2012 是ImageNet的子集，而ImageNet本身有超过1400多万张图片，超过2万多的分类。其中有超过100万张图片有明确类别标注和物体位置标注。

对于基于ImageNet的图像识别的结果评估，往往用到两个准确率的指标，一个是top-1准确率，一个是top-5准确率。**Top-1准确率指的是输出概率中最大的那一个对应的是正确类别的概率；top-5准确率指的是输出概率中最大的5个对应的5个类别中包含了正确类别的概率**。本仓库提供 Top-5 准确率类别预测，可以更明显的将模型的输出结果进行对比。

## 2. 模型下载地址

分类模型的模型转换文件已经上传至云服务器中，可通过 wget 命令在服务器网站中下载：

服务器网站地址：https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/

下载示例：

```shell
wget https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/EdgeNeXt_x_small_224x224_nv12.bin
```

模型下载shell脚本在各个分类模型的 model 文件夹中，可以执行 `sh download_bin.sh` 或 `sh download_onnx.sh` 命令下载 onnx/bin 文件。

## 3. 输入输出数据

- 输入数据

| 模型类别 | 输入数据  | 数据类型    | 数据形状              | 数据排布格式 |
| ---- | ----- | ------- | ----------------- | ------ |
| onnx | image | FLOAT32 | 1 x 3 x 224 x 224 | NCHW   |
| bin  | image | NV12    | 1 x 224 x 224 x 3 | NHWC   |

- 输出数据


| 模型类别 | 输出数据 | 数据类型 | 数据形状 | 数据排布格式 |
| ---- | ------- | ------- | -------- | ---- |
| onnx | classes | FLOAT32 | 1 x 1000 | NC   |
| bin  | classes | FLOAT32 | 1 x 1000 | NC   |


模型预训练的 pth 模型输入数据形状都为 `1x3x224x224` ，数据输出形状都为 `1x1000` ，符合 ImageNet 分类模型格式输出要求。具体到量化步骤，模型量化的 bin 文件输入数据格式都统一为 `nv12` ，模型预处理部分 `bpu_libdnn` 已内置格式转换函数，可直接调用改函数对模型数据进行处理。


## 4. PTQ量化流程

模型量化部署是边缘侧上板运行重要的一部分，此文档参照天工开物工具链量化部署部分，对部分内容进行了精简，方便使用者快速上手开发。也可参照一下文档链接，对工具链可以有更深的理解与应用：

工具链文档链接：https://developer.d-robotics.cc/rdk_doc/04_toolchain_development

以模型 [RepVGG](./RepVGG/README_cn.md) 为例，对 onnx 模型进行量化部署流程

### 4.1. 模型准备

首先使用git命令，下载RepVGG源代码：

```shell
git clone https://github.com/DingXiaoH/RepVGG.git
```

权重文件链接：https://drive.google.com/drive/folders/1Avome4KvNp0Lqh2QwhXO6L5URQjzCjUq

由于该模型在 timm 库和 pytorch 自带的模型框架中都未录入，而 [RepVGG](./RepVGG/README_cn.md) 又是部署端非常友好的模型，故需要自己下载模型代码进行模型转换操作。根据 [环境安装 | RDK DOC (d-robotics.cc)](https://developer.d-robotics.cc/rdk_doc/Advanced_development/toolchain_development/intermediate/environment_config) 中环境部署一节，安装好开发机的 Docker 环境。在开发机安装好 Docker 环境后，使能 Docker 环境，并在自己的开发机中 git 下来的 [RepVGG](./RepVGG/README_cn.md) 源码内新建 makeonnx.py Python 文件，内容如下：

```python
#!/usr/bin/env python3

"""
 Copyright (c) 2021-2024 D-Robotics Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from repvgg import *
import torch
import onnx
import torch.onnx

if __name__ == "__main__":
    model_path = "RepVGG-A2-train.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_RepVGG_A2(deploy=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model = repvgg_model_convert(model) # 重参数化

    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
    onnx_file_path = "RepVGG-A2.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        opset_version=11,
        verbose=True,
        input_names=["data"],  # 输入名
        output_names=["output"],  # 输出名
    )
```

运行此代码，生成 RepVGG 推理模型的 onnx 文件

### 4.2. 模型检查

在导出 RepVGG-A2.onnx onnx 模型后，需要对模型进行检查，主要是看模型是否存在算子中数据形状不匹配、输入数据不固定的情况（分类模型暂时不支持动态多Batch和多Shape输入）。新建 01_check.sh 并添加以下 Shell 脚本代码并执行 `sh 01_check.sh` 指令：

```shell
#!/usr/bin/env bash
# Copyright (c) 2024 D-Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of D-Robotics Inc. This is proprietary information owned by
# D-Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of D-Robotics Inc.

set -ex
cd $(dirname $0) || exit

model_type="onnx"
onnx_model="RepVGG-A2.onnx"
march="bayes-e"

hb_mapper checker --model-type ${model_type} \
                  --model ${onnx_model} \
                  --march ${march}
```

生成部分的文件日志 hb_mapper_checker.log 如下：

```shell
============================================================================================
Node                         ON   Subgraph  Type                           In/Out DataType  
--------------------------------------------------------------------------------------------
/stage0/rbr_reparam/Conv     BPU  id(0)     HzSQuantizedConv               int8/int8        
/stage1.0/rbr_reparam/Conv   BPU  id(0)     HzSQuantizedConv               int8/int8        
/stage1.1/rbr_reparam/Conv   BPU  id(0)     HzSQuantizedConv               int8/int8        
/stage1.2/rbr_reparam/Conv   BPU  id(0)     HzSQuantizedConv               int8/int8        
/stage1.3/rbr_reparam/Conv   BPU  id(0)     HzSQuantizedConv               int8/int8        
/stage2.0/rbr_reparam/Conv   BPU  id(0)     HzSQuantizedConv               int8/int8        
/stage2.1/rbr_reparam/Conv   BPU  id(0)     HzSQuantizedConv               int8/int8        
/stage2.2/rbr_reparam/Conv   BPU  id(0)     HzSQuantizedConv               int8/int8        
/stage2.3/rbr_reparam/Conv   BPU  id(0)     HzSQuantizedConv               int8/int8        
/stage2.4/rbr_reparam/Conv   BPU  id(0)     HzSQuantizedConv               int8/int8        
/stage2.5/rbr_reparam/Conv   BPU  id(0)     HzSQuantizedConv               int8/int8        
/stage3.0/rbr_reparam/Conv   BPU  id(0)     HzSQuantizedConv               int8/int8        
/stage3.1/rbr_reparam/Conv   BPU  id(0)     HzSQuantizedConv               int8/int8        
/stage3.2/rbr_reparam/Conv   BPU  id(0)     HzSQuantizedConv               int8/int8        
/stage3.3/rbr_reparam/Conv   BPU  id(0)     HzSQuantizedConv               int8/int8        
/stage3.4/rbr_reparam/Conv   BPU  id(0)     HzSQuantizedConv               int8/int8        
/stage3.5/rbr_reparam/Conv   BPU  id(0)     HzSQuantizedConv               int8/int8        
/stage3.6/rbr_reparam/Conv   BPU  id(0)     HzSQuantizedConv               int8/int8        
/stage3.7/rbr_reparam/Conv   BPU  id(0)     HzSQuantizedConv               int8/int8        
/stage3.8/rbr_reparam/Conv   BPU  id(0)     HzSQuantizedConv               int8/int8        
/stage3.9/rbr_reparam/Conv   BPU  id(0)     HzSQuantizedConv               int8/int8        
/stage3.10/rbr_reparam/Conv  BPU  id(0)     HzSQuantizedConv               int8/int8        
/stage3.11/rbr_reparam/Conv  BPU  id(0)     HzSQuantizedConv               int8/int8        
/stage3.12/rbr_reparam/Conv  BPU  id(0)     HzSQuantizedConv               int8/int8        
/stage3.13/rbr_reparam/Conv  BPU  id(0)     HzSQuantizedConv               int8/int8        
/stage3.14/rbr_reparam/Conv  BPU  id(0)     HzSQuantizedConv               int8/int8        
/stage3.15/rbr_reparam/Conv  BPU  id(0)     HzSQuantizedConv               int8/int8        
/stage4.0/rbr_reparam/Conv   BPU  id(0)     HzSQuantizedConv               int8/int8        
/gap/GlobalAveragePool       BPU  id(0)     HzSQuantizedGlobalAveragePool  int8/int8        
/linear/Gemm                 BPU  id(0)     HzSQuantizedConv               int8/int32       
/linear/Gemm_reshape_output  CPU  --        Reshape                        float/float
```

可以看到，[RepVGG](./RepVGG/README_cn.md) 全部算子都是可以跑在BPU上的，且模型是 int8 量化，不需要量化为 int16 格式，对部署推理更加友好。


### 4.3. 校准集构建

前文提到模型是使用 ImageNet 数据集进行训练的，那么在校准集步骤则需要准备 100~300 张左右的数据集对模型进行校准操作。模型数据集可以使用天工开物工具链提供的校准数据集，也可以直接下载 ILSVRC2012 测试集后随机选取 100 张左右的图片（10G 左右）。新建 preprocess.py Python 前处理文件，加入以下代码：

```Python
#!/usr/bin/env python3

"""
 Copyright (c) 2021-2024 D-Robotics Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from horizon_tc_ui.data.dataloader import (ImageNetDataLoader,
                                           SingleImageDataLoader)
from horizon_tc_ui.data.transformer import (
    BGR2NV12Transformer, BGR2RGBTransformer, CenterCropTransformer,
    HWC2CHWTransformer, NV12ToYUV444Transformer, ShortSideResizeTransformer)

short_size = 256
crop_size = 224
model_input_height = 224
model_input_width = 224

def calibration_transformers():
    """
    step：
        1、short size resize to 256
        2、crop size 224 * 224 from center
        3、NHWC to NCHW
        4、bgr to rgb
    """
    transformers = [
        ShortSideResizeTransformer(short_size=short_size),
        CenterCropTransformer(crop_size=crop_size),
        HWC2CHWTransformer(),
        BGR2RGBTransformer()
    ]
    return transformers

def infer_transformers(input_layout="NHWC"):
    """
    step：
        1、short size resize to 256
        2、crop size 224 * 224 from center
        3、bgr to nv12
        4、nv12 to yuv444
    :param input_layout: input layout
    """
    transformers = [
        ShortSideResizeTransformer(short_size=short_size),
        CenterCropTransformer(crop_size=crop_size),
        BGR2NV12Transformer(data_format="HWC"),
        NV12ToYUV444Transformer((model_input_height, model_input_width),
                                yuv444_output_layout=input_layout[1:]),
    ]
    return transformers

def infer_image_preprocess(image_file, input_layout):
    """
    image for single image inference
    note: imread_mode [skimage / opencv]
        opencv read image as 8-bit unsigned integers BGR in range [0, 255]
        skimage read image as float32 RGB in range [0, 1]
        make sure to use the same imread_mode as the model training
    :param image_file: image path
    :param input_layout: NCHW / NHWC
    :return: processed image (uint8, 0-255)
    """
    transformers = infer_transformers(input_layout)
    image = SingleImageDataLoader(transformers,
                                  image_file,
                                  imread_mode="opencv")
    return image

def eval_image_preprocess(image_path, label_path, input_layout):
    """
    image for full scale evaluation
    note: imread_mode [skimage / opencv]
        opencv read image as 8-bit unsigned integers BGR in range [0, 255]
        skimage read image as float32 RGB in range [0, 1]
        make sure to use the same imread_mode as the model training
    :param image_path: image path
    :param label_path: label path
    :param input_layout: input layout
    :return: data_loader, evaluation lable size
    """
    transformers = infer_transformers(input_layout)
    data_loader = ImageNetDataLoader(transformers,
                                     image_path,
                                     label_path,
                                     imread_mode='opencv',
                                     batch_size=1,
                                     return_img_name=True)
    loader_size = sum(1 for line in open(label_path))
    return data_loader, loader_size
```

新建 `02_preprocess.sh` 脚本文件，加入以下代码并执行 `sh 02_preprocess.sh` :

```shell
#!/usr/bin/env bash
# Copyright (c) 2024 D-Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of D-Robotics Inc. This is proprietary information owned by
# D-Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of D-Robotics Inc.

set -e -v

cd $(dirname $0) || exit

python3 data_preprocess.py \
  --src_dir calibration_data/imagenet \
  --dst_dir ./calibration_data_rgb_f32 \
  --pic_ext .rgb \
  --read_mode opencv \
  --saved_data_type float32 
```

此脚本文件将调用 Python 代码中的 calibration_transformers函数，函数将图像的短边调整为256像素，保持长宽比不变，从图像中心裁剪一个大小为224 x 224的图像块，并将图像从 NHWC（通道在最后）格式转换为NCHW（通道在前）格式（根据工具链的文档，量化模型的输入必须是通道在前格式），最后将图像从 opencv 的BGR 格式转换为模型输入的 RGB 格式。转换后的输出如下:

```shell
write:./calibration_data_rgb_f32/ILSVRC2012_val_00000092.rgb
write:./calibration_data_rgb_f32/ILSVRC2012_val_00000096.rgb
write:./calibration_data_rgb_f32/ILSVRC2012_val_00000079.rgb
write:./calibration_data_rgb_f32/ILSVRC2012_val_00000099.rgb
write:./calibration_data_rgb_f32/ILSVRC2012_val_00000100.rgb
write:./calibration_data_rgb_f32/ILSVRC2012_val_00000098.rgb
write:./calibration_data_rgb_f32/ILSVRC2012_val_00000097.rgb
```

### 4.4. 模型量化与编译 

在模型准备、模型检查和校准集构建步骤后，就可以进入到最重要的步骤 —— 模型量化与编译。模型转换需要配置模型的 yaml 配置文件。文件中的配置参数可参照 [模型转换yaml配置参数说明](https://developer.d-robotics.cc/rdk_doc/Advanced_development/toolchain_development/intermediate/ptq_process#model_conversion) 具体参数信息 ，新建 `RepVGG_deploy_config.yaml`， 写出如下配置：

```yaml
# Copyright (c) 2024 D-Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of D-Robotics Inc. This is proprietary information owned by
# D-Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of D-Robotics Inc.

model_parameters:
  onnx_model: './RepVGG-A2.onnx'
  march: "bayes-e"
  layer_out_dump: False
  working_dir: 'RepVGG_224x224_nv12'
  output_model_file_prefix: 'RepVGG_224x224_nv12'
  debug_mode: "dump_calibration_data"
  remove_node_type: 'Quantize;Dequantize;Transpose;Cast;Reshape'

input_parameters:
  input_name: ""
  input_type_rt: 'nv12'
  input_type_train: 'rgb'
  input_layout_train: 'NCHW'
  input_shape: ''
  norm_type: 'data_mean_and_scale'
  mean_value: 123.675 116.28 103.53
  scale_value: 0.01712475 0.017507 0.01742919

calibration_parameters:
  cal_data_dir: './calibration_data_rgb_f32'
  cal_data_type: 'float32'
  calibration_type: 'default'

compiler_parameters:
  compile_mode: 'latency'
  debug: False
  optimize_level: 'O3'
```

新建 03_build.sh 脚本文件，加入以下代码并执行 `sh 03_build.sh`：

```shell
#!/usr/bin/env bash
# Copyright (c) 2024 D-Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of D-Robotics Inc. This is proprietary information owned by
# D-Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of D-Robotics Inc.

set -e -v

cd $(dirname $0) || exit

config_file="./RepVGG_deploy_config.yaml"
model_type="onnx"
# build model
hb_mapper makertbin --config ${config_file}  \
                    --model-type  ${model_type}
```

量化过程中生成部分的文件日志 hb_mapper_makerbin.log 如下：

```shell
======================================================================================================================================
Node                                                ON   Subgraph  Type               Cosine Similarity  Threshold   In/Out DataType  
--------------------------------------------------------------------------------------------------------------------------------------
HZ_PREPROCESS_FOR_data                              BPU  id(0)     HzPreprocess       0.999968           127.000000  int8/int8        
/stage0/rbr_reparam/Conv                            BPU  id(0)     Conv               0.999229           4.882881    int8/int8        
/stage0/nonlinearity/Relu_output_0_calibrated_pad   BPU  id(0)     HzPad              0.999229           31.343239   int8/int8        
/stage1.0/rbr_reparam/Conv                          BPU  id(0)     Conv               0.996237           31.343239   int8/int8        
...ge1.0/nonlinearity/Relu_output_0_calibrated_pad  BPU  id(0)     HzPad              0.996237           43.586590   int8/int8        
/stage1.1/rbr_reparam/Conv                          BPU  id(0)     Conv               0.990095           43.586590   int8/int8        
...ge1.1/nonlinearity/Relu_output_0_calibrated_pad  BPU  id(0)     HzPad              0.990095           28.410471   int8/int8        
/stage2.0/rbr_reparam/Conv                          BPU  id(0)     Conv               0.993131           28.410471   int8/int8        
...ge2.0/nonlinearity/Relu_output_0_calibrated_pad  BPU  id(0)     HzPad              0.993131           13.251609   int8/int8        
/stage2.1/rbr_reparam/Conv                          BPU  id(0)     Conv               0.992364           13.251609   int8/int8        
...ge2.1/nonlinearity/Relu_output_0_calibrated_pad  BPU  id(0)     HzPad              0.992364           12.763056   int8/int8        
/stage2.2/rbr_reparam/Conv                          BPU  id(0)     Conv               0.982343           12.763056   int8/int8        
...ge2.2/nonlinearity/Relu_output_0_calibrated_pad  BPU  id(0)     HzPad              0.982343           11.392269   int8/int8        
/stage2.3/rbr_reparam/Conv                          BPU  id(0)     Conv               0.975596           11.392269   int8/int8        
...ge2.3/nonlinearity/Relu_output_0_calibrated_pad  BPU  id(0)     HzPad              0.975596           15.545307   int8/int8        
/stage3.0/rbr_reparam/Conv                          BPU  id(0)     Conv               0.955964           15.545307   int8/int8        
...ge3.0/nonlinearity/Relu_output_0_calibrated_pad  BPU  id(0)     HzPad              0.955964           5.586124    int8/int8        
/stage3.1/rbr_reparam/Conv                          BPU  id(0)     Conv               0.963102           5.586124    int8/int8        
...ge3.1/nonlinearity/Relu_output_0_calibrated_pad  BPU  id(0)     HzPad              0.963102           5.237350    int8/int8        
/stage3.2/rbr_reparam/Conv                          BPU  id(0)     Conv               0.971733           5.237350    int8/int8        
...ge3.2/nonlinearity/Relu_output_0_calibrated_pad  BPU  id(0)     HzPad              0.971733           8.646875    int8/int8        
/stage3.3/rbr_reparam/Conv                          BPU  id(0)     Conv               0.968726           8.646875    int8/int8        
...ge3.3/nonlinearity/Relu_output_0_calibrated_pad  BPU  id(0)     HzPad              0.968726           5.870739    int8/int8        
/stage3.4/rbr_reparam/Conv                          BPU  id(0)     Conv               0.961378           5.870739    int8/int8        
...ge3.4/nonlinearity/Relu_output_0_calibrated_pad  BPU  id(0)     HzPad              0.961378           6.652196    int8/int8        
/stage3.5/rbr_reparam/Conv                          BPU  id(0)     Conv               0.956340           6.652196    int8/int8        
...ge3.5/nonlinearity/Relu_output_0_calibrated_pad  BPU  id(0)     HzPad              0.956340           5.533033    int8/int8        
/stage3.6/rbr_reparam/Conv                          BPU  id(0)     Conv               0.954281           5.533033    int8/int8        
...ge3.6/nonlinearity/Relu_output_0_calibrated_pad  BPU  id(0)     HzPad              0.954281           8.499337    int8/int8        
/stage3.7/rbr_reparam/Conv                          BPU  id(0)     Conv               0.946763           8.499337    int8/int8        
...ge3.7/nonlinearity/Relu_output_0_calibrated_pad  BPU  id(0)     HzPad              0.946763           7.030814    int8/int8        
/stage3.8/rbr_reparam/Conv                          BPU  id(0)     Conv               0.942780           7.030814    int8/int8        
...ge3.8/nonlinearity/Relu_output_0_calibrated_pad  BPU  id(0)     HzPad              0.942780           4.679632    int8/int8        
/stage3.9/rbr_reparam/Conv                          BPU  id(0)     Conv               0.930315           4.679632    int8/int8        
...ge3.9/nonlinearity/Relu_output_0_calibrated_pad  BPU  id(0)     HzPad              0.930315           6.504673    int8/int8        
/stage3.10/rbr_reparam/Conv                         BPU  id(0)     Conv               0.908927           6.504673    int8/int8        
...e3.10/nonlinearity/Relu_output_0_calibrated_pad  BPU  id(0)     HzPad              0.908927           5.817932    int8/int8        
/stage3.11/rbr_reparam/Conv                         BPU  id(0)     Conv               0.879031           5.817932    int8/int8        
...e3.11/nonlinearity/Relu_output_0_calibrated_pad  BPU  id(0)     HzPad              0.879031           3.217116    int8/int8        
/stage3.12/rbr_reparam/Conv                         BPU  id(0)     Conv               0.817807           3.217116    int8/int8        
...e3.12/nonlinearity/Relu_output_0_calibrated_pad  BPU  id(0)     HzPad              0.817807           5.921410    int8/int8        
/stage3.13/rbr_reparam/Conv                         BPU  id(0)     Conv               0.742658           5.921410    int8/int8        
...e3.13/nonlinearity/Relu_output_0_calibrated_pad  BPU  id(0)     HzPad              0.742658           6.380197    int8/int8        
/stage4.0/rbr_reparam/Conv                          BPU  id(0)     Conv               0.860627           6.380197    int8/int8        
/gap/GlobalAveragePool                              BPU  id(0)     GlobalAveragePool  0.700676           314.158783  int8/int8        
/linear/Gemm                                        BPU  id(0)     Conv               0.768926           21.596825   int8/int32       
/linear/Gemm_reshape_output                         CPU  --        Reshape            0.768926           --          float/float
2024-08-22 15:44:04,038 file: build.py func: build line No: 408 The quantify model output:
===============================================================================
Node          Cosine Similarity  L1 Distance  L2 Distance  Chebyshev Distance  
-------------------------------------------------------------------------------
/linear/Gemm  0.768926           1.503200     0.060809     8.068202
```

由于RepVGG是串行的VGG like结构，**不需要进行int16量化，所以量化调优测量倾向于更改模型内部融合测量和量化矫正方法，公版模型量化余弦相似度较低**。可以根据工具链参考文档下载参考模型进一步对模型进行调优。

当然，得到的 model_output/RepVGG_224x224_nv12.bin 文件如果想进一步优化，可以执行以下指令：

```shell
hb_model_modifier RepVGG_224x224_nv12/RepVGG_224x224_nv12.bin -r /linear/Gemm_reshape_output
``` 

这条指令将去除模型最后的 Reshape 算子，使模型以 INT32 输出，**实现所有算子都在 BPU 上运行**。

### 4.5. 模型推理

新建 postprocess.py Python 后处理文件，加入以下代码和图片：

![](./RepVGG/data/gooze.JPEG)

```Python
#!/usr/bin/env python3

"""
 Copyright (c) 2021-2024 D-Robotics Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import numpy as np
import logging
from multiprocessing.pool import ApplyResult
from horizon_tc_ui.data.imagenet_val import imagenet_val
from horizon_tc_ui.utils import tool_utils

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def postprocess(model_output):
    logits = np.squeeze(model_output[0])
    prob = softmax(logits)
    idx = np.argsort(-prob)
    top_five_label_probs = [(idx[i], prob[idx[i]], imagenet_val[idx[i]])
                            for i in range(5)]
    return top_five_label_probs

def eval_postprocess(model_output, label):
    predict_label, _, _ = postprocess(model_output)[0]
    if predict_label == label:
        sample_correction = 1
    else:
        sample_correction = 0
    return predict_label, sample_correction

def calc_accuracy(accuracy, total_batch):
    if isinstance(accuracy[0], ApplyResult):
        accuracy = [i.get() for i in accuracy]
        batch_result = sorted(list(accuracy), key=lambda e: e[0])
    else:
        batch_result = sorted(list(accuracy), key=lambda e: e[0])
    total_correct_samples = 0
    total_samples = 0
    acc_all = 0
    with open("RepVGG_eval.log", "w") as eval_log_handle:
        for data_id, predict_label, sample_correction, filename in batch_result:
            total_correct_samples += sample_correction
            total_samples += 1
            if total_samples % 10 == 0:
                record = 'Batch:{}/{}; accuracy(all):{:.4f}'.format(
                    data_id + 1, total_batch,
                    total_correct_samples / total_samples)
                logging.info(record)
            acc_all = total_correct_samples / total_samples
            eval_log_handle.write(
                f"input_image_name: {filename[0]} class_id: {predict_label:3d} class_name: {imagenet_val[predict_label][0]} \n"
            )
    return acc_all

def gen_report(acc):
    tool_utils.report_flag_start('MAPPER-EVAL')
    logging.info('{:.4f}'.format(acc))
    tool_utils.report_flag_end('MAPPER-EVAL')
```

新建 `cls_inference.py`，加入以下内容：

```Python
# Copyright (c) 2021 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import click
import logging

from horizon_tc_ui import HB_ONNXRuntime
from horizon_tc_ui.utils.tool_utils import init_root_logger, on_exception_exit

from preprocess import infer_image_preprocess
from postprocess import postprocess


def inference(sess, image_name, input_layout):
    if input_layout is None:
        logging.warning(f"input_layout not provided. Using {sess.layout[0]}")
        input_layout = sess.layout[0]
    image_data = infer_image_preprocess(image_name, input_layout)
    input_name = sess.input_names[0]
    output_names = sess.output_names
    output = sess.run(output_names, {input_name: image_data})
    top_five_label_probs = postprocess(output)
    logging.info("The input picture is classified to be:")
    for label_item, prob_item, class_item in top_five_label_probs:
        logging.info(
            f"label {label_item:3d}, prob {prob_item:.5f}, class {class_item}")


@click.version_option(version="1.0.0")
@click.command()
@click.option('-m', '--model', type=str, help='Input onnx model(.onnx) file')
@click.option('-i', '--image', type=str, help='Input image file.')
@click.option('-y',
              '--input_layout',
              type=str,
              default="",
              help='Model input layout')
@click.option('-o',
              '--input_offset',
              type=object,
              default=None,
              help='input inference offset.')
@click.option('-c',
              '--color_sequence',
              type=str,
              default=None,
              help='Color sequence')
@on_exception_exit
def main(model, image, input_layout, input_offset, color_sequence):
    init_root_logger("inference",
                     console_level=logging.INFO,
                     file_level=logging.DEBUG)
    if color_sequence:
        logging.warning("option color_sequence is deprecated.")
    if input_offset:
        logging.warning("option input_offset is deprecated.")

    sess = HB_ONNXRuntime(model_file=model)
    sess.set_dim_param(0, 0, '?')
    inference(sess, image, input_layout)


if __name__ == '__main__':
    main()
```


新建 `04_inference.sh` 脚本文件，加入以下代码并执行 `sh 04_inference.sh`，最终输出结果如下：

```shell
# Copyright (c) 2024 D-Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of D-Robotics Inc. This is proprietary information owned by
# D-Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of D-Robotics Inc.

set -e -v
cd $(dirname $0) || exit

#for converted quanti model inference
quanti_model_file="./RepVGG_224x224_nv12/RepVGG_224x224_nv12_quantized_model.onnx"
quanti_input_layout="NHWC"

#for original float model inference
original_model_file="./RepVGG_224x224_nv12/RepVGG_224x224_nv12_original_float_model.onnx"
original_input_layout="NCHW"

if [[ $1 =~ "origin" ]];  then
  model=$original_model_file
  layout=$original_input_layout
else
  model=$quanti_model_file
  layout=$quanti_input_layout
fi

infer_image="gooze.JPEG"

# -----------------------------------------------------------------------------------------------------
# shell command "sh 04_inference.sh" runs quanti inference by default 
# If quanti model infer is intended, please run the shell via command "sh 04_inference.sh quanti"
# If float  model infer is intended, please run the shell via command "sh 04_inference.sh origin"
# -----------------------------------------------------------------------------------------------------

python3 -u cls_inference.py \
        --model ${model} \
        --image ${infer_image} \
        --input_layout ${layout} 
```

```shell
2024-08-22 15:58:25,627 file: tool_utils.py func: tool_utils line No: 77 log will be stored in /open_explorer/samples/ai_toolchain/horizon_model_convert_sample/03_classification/RepVGG_A2/mapper/inference.log
2024-08-22 15:58:31,209 file: hb_onnxruntime.py func: hb_onnxruntime line No: 252 input[data] model input type is int8, input data type is uint8, will be convert.
2024-08-22 15:58:32,502 file: cls_inference.py func: cls_inference line No: 28 The input picture is classified to be:
2024-08-22 15:58:32,503 file: cls_inference.py func: cls_inference line No: 30 label  99, prob 0.99894, class ['goose']
2024-08-22 15:58:32,503 file: cls_inference.py func: cls_inference line No: 30 label  97, prob 0.00092, class ['drake']
2024-08-22 15:58:32,503 file: cls_inference.py func: cls_inference line No: 30 label  98, prob 0.00003, class ['red-breasted merganser, Mergus serrator']
2024-08-22 15:58:32,503 file: cls_inference.py func: cls_inference line No: 30 label 100, prob 0.00002, class ['black swan, Cygnus atratus']
2024-08-22 15:58:32,503 file: cls_inference.py func: cls_inference line No: 30 label 146, prob 0.00002, class ['albatross, mollymawk']
```

可以看到，输出的置信度为 0.9989，单张图片推理置信度还是很高的。

新建 `cls_evaluate.py`，加入以下代码：

```Python
# Copyright (c) 2021 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import os
import logging
import multiprocessing
import click

from horizon_tc_ui import HB_ONNXRuntime
from horizon_tc_ui.utils.tool_utils import init_root_logger, on_exception_exit

from preprocess import eval_image_preprocess
from postprocess import eval_postprocess, calc_accuracy, gen_report

sess = None

MULTI_PROCESS_WARNING_DICT = {
    "CPUExecutionProvider": {
        "origin":
            """
            onnxruntime infering float model does not work well with
            multiprocess, the program may be blocked.
            It is recommended to use single process operation
            """,
        "quanti":
            "",
    },
    "CUDAExecutionProvider": {
        "origin":
            """
            GPU does not work well with multiprocess, the program may prompt
            errors. It is recommended to use single process operation
            """,
        "quanti":
            """
            GPU does not work well with multiprocess, the program may prompt
            errors. It is recommended to use single process operation
            """,
    }
}
SINGLE_PROCESS_WARNING_DICT = {
    "CPUExecutionProvider": {
        "origin":
            "",
        "quanti":
            """
            Infering with single process may take a long time.
            It is recommended to use multi process running
            """
    },
    "CUDAExecutionProvider": {
        "origin": "",
        "quanti": ""
    }
}

DEFAULT_PARALLEL_NUM = {
    "CPUExecutionProvider": {
        "origin": 1,
        "quanti": int(os.environ.get('PARALLEL_PROCESS_NUM', 10)),
    },
    "CUDAExecutionProvider": {
        "origin": 1,
        "quanti": 1,
    }
}


def init_sess(model):
    global sess
    sess = HB_ONNXRuntime(model_file=model)
    sess.set_dim_param(0, 0, '?')


class ParallelExector(object):
    def __init__(self, parallel_num, input_layout):
        self._results = []
        self.input_layout = input_layout

        self.parallel_num = parallel_num
        self.validate()

        if self.parallel_num != 1:
            logging.info(f"Init {self.parallel_num} processes")
            self._pool = multiprocessing.Pool(processes=self.parallel_num)
            self._queue = multiprocessing.Manager().Queue(self.parallel_num)

    def get_accuracy(self, total_batch):
        return calc_accuracy(self._results, total_batch)

    def infer(self, val_data, batch_id, total_batch):
        if self.parallel_num != 1:
            self.feed(val_data, batch_id, total_batch)
        else:
            logging.info(f"Feed batch {batch_id + 1}/{total_batch}")
            eval_result = run(val_data, batch_id, total_batch,
                              self.input_layout)
            self._results.append(eval_result)

    def feed(self, val_data, batch_id, total_batch):
        self._pool.apply(func=product,
                         args=(self._queue, val_data, batch_id, total_batch))
        r = self._pool.apply_async(func=consumer,
                                   args=(self._queue, batch_id, total_batch,
                                         self.input_layout),
                                   error_callback=logging.error)
        self._results.append(r)

    def close(self):
        if hasattr(self, "_pool"):
            self._pool.close()
            self._pool.join()

    def validate(self):
        provider = sess.get_provider()
        if sess.get_input_type() == 3:
            model_type = "quanti"
        else:
            model_type = "origin"

        if self.parallel_num == 0:
            self.parallel_num = DEFAULT_PARALLEL_NUM[provider][model_type]

        if self.parallel_num == 1:
            warning_message = SINGLE_PROCESS_WARNING_DICT[provider][model_type]
        else:
            warning_message = MULTI_PROCESS_WARNING_DICT[provider][model_type]

        if warning_message:
            logging.warning(warning_message)


def product(queue, val_data, batch_id, total_batch):
    logging.info("Feed batch {}/{}".format(batch_id + 1, total_batch))
    queue.put(val_data)


def consumer(queue, batch_id, total_batch, input_layout):
    return run(queue.get(), batch_id, total_batch, input_layout)


def run(val_data, batch_id, total_batch, input_layout):
    logging.info("Eval batch {}/{}".format(batch_id + 1, total_batch))
    input_name = sess.input_names[0]
    output_names = sess.output_names
    (data, label, filename) = val_data
    # make sure pre-process logic is the same with runtime
    output = sess.run(output_names, {input_name: data})
    predict_label, sample_correction = eval_postprocess(output, label)
    return batch_id, predict_label, sample_correction, filename


def evaluate(image_path, label_path, input_layout, parallel_num):
    if not input_layout:
        logging.warning(f"input_layout not provided. Using {sess.layout[0]}")
        input_layout = sess.layout[0]
    data_loader, loader_size = eval_image_preprocess(image_path, label_path,
                                                     input_layout)
    val_exe = ParallelExector(parallel_num, input_layout)
    total_batch = loader_size
    for batch_id, val_data in enumerate(data_loader):
        val_exe.infer(val_data, batch_id, total_batch)

    val_exe.close()
    metric_result = val_exe.get_accuracy(total_batch)
    gen_report(metric_result)


@click.version_option(version="1.0.0")
@click.command()
@click.option('-m', '--model', type=str, help='Input onnx model(.onnx) file')
@click.option('-i',
              '--image_path',
              type=str,
              help='Evaluation image directory.')
@click.option('-l', '--label_path', type=str, help='Evaluate image label path')
@click.option('-y',
              '--input_layout',
              type=str,
              default="",
              help='Model input layout')
@click.option('-o',
              '--input_offset',
              type=object,
              default=None,
              help='input inference offset.')
@click.option('-p',
              '--parallel_num',
              type=int,
              default=0,
              help="""
    Parallel eval process number. The default value of evaluating
    fixed-point model using CPU is 10, and other defaults are 1
    """)
@click.option('-c',
              '--color_sequence',
              type=str,
              default=None,
              help='Color sequence')
@on_exception_exit
def main(model, image_path, label_path, input_layout, input_offset,
         parallel_num, color_sequence):
    init_root_logger("evaluation",
                     console_level=logging.INFO,
                     file_level=logging.DEBUG)
    if color_sequence:
        logging.warning("option color_sequence is deprecated.")
    if input_offset:
        logging.warning("option input_offset is deprecated.")

    init_sess(model)
    sess.set_dim_param(0, 0, '?')
    evaluate(image_path, label_path, input_layout, parallel_num)


if __name__ == '__main__':
    main()
```

新建 05_evaluate.sh 脚本文件，加入以下代码。另外需要补充说明的是，需要新建 val 文件夹用于存放 ImageNet 验证集，地址在此处：https://pan.baidu.com/s/1MEjNh6evha2hcdrQXjNv8w?pwd=yzza val.txt 和 val_piece.txt 是工具链自带的测试标签，可以直接复制粘贴一下即可。

执行 sh 05_evaluate.sh quanti test。最终输出结果如下：

```shell
#!/usr/bin/env bash
# Copyright (c) 2024 D-Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of D-Robotics Inc. This is proprietary information owned by
# D-Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of D-Robotics Inc.

set -v -e
cd $(dirname $0) || exit

#for converted quanti model inference
quanti_model_file="./RepVGG_224x224_nv12_int8/RepVGG_224x224_nv12_quantized_model.onnx"
quanti_input_layout="NHWC"

#for original float model inference
original_model_file="./RepVGG_224x224_nv12_int8/RepVGG_224x224_nv12_original_float_model.onnx"
original_input_layout="NCHW"

if [[ $1 =~ "origin" ]];  then
  model=$original_model_file
  layout=$original_input_layout
else
  model=$quanti_model_file
  layout=$quanti_input_layout
fi

imagenet_data_path="val"
if [ -z $2 ]; then 
  imagenet_label_path="val.txt"
else
  imagenet_label_path="val_piece.txt"
fi

# -------------------------------------------------------------------------------------------------------------
# shell command "sh 05_evaluate.sh" runs quanti full evaluation by default 
# If quanti model eval is intended, please run the shell via command "sh 05_evaluate.sh quanti"
# If float  model eval is intended, please run the shell via command "sh 05_evaluate.sh origin"#
# If quanti model quick eval test is intended, please run the shell via command "sh 05_evaluate.sh quanti test"
# If float  model quick eval test is intended, please run the shell via command "sh 05_evaluate.sh origin test"
# -------------------------------------------------------------------------------------------------------------

python3 -u cls_evaluate.py \
        --model ${model} \
        --image_path ${imagenet_data_path} \
        --label_path ${imagenet_label_path} \
        --input_layout ${layout}  
```

```shell
2024-08-22 19:49:59,095 file: postprocess.py func: postprocess line No: 55 Batch:10/400; accuracy(all):0.6000
2024-08-22 19:49:59,095 file: postprocess.py func: postprocess line No: 55 Batch:20/400; accuracy(all):0.7000
2024-08-22 19:49:59,096 file: postprocess.py func: postprocess line No: 55 Batch:30/400; accuracy(all):0.6333
2024-08-22 19:49:59,096 file: postprocess.py func: postprocess line No: 55 Batch:40/400; accuracy(all):0.6500
2024-08-22 19:49:59,096 file: postprocess.py func: postprocess line No: 55 Batch:50/400; accuracy(all):0.6400
2024-08-22 19:49:59,096 file: postprocess.py func: postprocess line No: 55 Batch:60/400; accuracy(all):0.6500
2024-08-22 19:49:59,096 file: postprocess.py func: postprocess line No: 55 Batch:70/400; accuracy(all):0.6429
2024-08-22 19:49:59,096 file: postprocess.py func: postprocess line No: 55 Batch:80/400; accuracy(all):0.6625
2024-08-22 19:49:59,096 file: postprocess.py func: postprocess line No: 55 Batch:90/400; accuracy(all):0.6667
2024-08-22 19:49:59,096 file: postprocess.py func: postprocess line No: 55 Batch:100/400; accuracy(all):0.6700
2024-08-22 19:49:59,096 file: postprocess.py func: postprocess line No: 55 Batch:110/400; accuracy(all):0.6636
2024-08-22 19:49:59,096 file: postprocess.py func: postprocess line No: 55 Batch:120/400; accuracy(all):0.6583
2024-08-22 19:49:59,097 file: postprocess.py func: postprocess line No: 55 Batch:130/400; accuracy(all):0.6462
2024-08-22 19:49:59,097 file: postprocess.py func: postprocess line No: 55 Batch:140/400; accuracy(all):0.6429
2024-08-22 19:49:59,097 file: postprocess.py func: postprocess line No: 55 Batch:150/400; accuracy(all):0.6533
2024-08-22 19:49:59,097 file: postprocess.py func: postprocess line No: 55 Batch:160/400; accuracy(all):0.6438
2024-08-22 19:49:59,097 file: postprocess.py func: postprocess line No: 55 Batch:170/400; accuracy(all):0.6353
2024-08-22 19:49:59,097 file: postprocess.py func: postprocess line No: 55 Batch:180/400; accuracy(all):0.6444
2024-08-22 19:49:59,097 file: postprocess.py func: postprocess line No: 55 Batch:190/400; accuracy(all):0.6368
2024-08-22 19:49:59,097 file: postprocess.py func: postprocess line No: 55 Batch:200/400; accuracy(all):0.6300
2024-08-22 19:49:59,097 file: postprocess.py func: postprocess line No: 55 Batch:210/400; accuracy(all):0.6286
2024-08-22 19:49:59,097 file: postprocess.py func: postprocess line No: 55 Batch:220/400; accuracy(all):0.6364
2024-08-22 19:49:59,097 file: postprocess.py func: postprocess line No: 55 Batch:230/400; accuracy(all):0.6304
2024-08-22 19:49:59,098 file: postprocess.py func: postprocess line No: 55 Batch:240/400; accuracy(all):0.6250
2024-08-22 19:49:59,098 file: postprocess.py func: postprocess line No: 55 Batch:250/400; accuracy(all):0.6240
2024-08-22 19:49:59,098 file: postprocess.py func: postprocess line No: 55 Batch:260/400; accuracy(all):0.6269
2024-08-22 19:49:59,098 file: postprocess.py func: postprocess line No: 55 Batch:270/400; accuracy(all):0.6259
2024-08-22 19:49:59,098 file: postprocess.py func: postprocess line No: 55 Batch:280/400; accuracy(all):0.6250
2024-08-22 19:49:59,098 file: postprocess.py func: postprocess line No: 55 Batch:290/400; accuracy(all):0.6241
2024-08-22 19:49:59,098 file: postprocess.py func: postprocess line No: 55 Batch:300/400; accuracy(all):0.6133
2024-08-22 19:49:59,098 file: postprocess.py func: postprocess line No: 55 Batch:310/400; accuracy(all):0.6097
2024-08-22 19:49:59,098 file: postprocess.py func: postprocess line No: 55 Batch:320/400; accuracy(all):0.6062
2024-08-22 19:49:59,099 file: postprocess.py func: postprocess line No: 55 Batch:330/400; accuracy(all):0.6182
2024-08-22 19:49:59,099 file: postprocess.py func: postprocess line No: 55 Batch:340/400; accuracy(all):0.6176
2024-08-22 19:49:59,099 file: postprocess.py func: postprocess line No: 55 Batch:350/400; accuracy(all):0.6229
2024-08-22 19:49:59,099 file: postprocess.py func: postprocess line No: 55 Batch:360/400; accuracy(all):0.6250
2024-08-22 19:49:59,099 file: postprocess.py func: postprocess line No: 55 Batch:370/400; accuracy(all):0.6297
2024-08-22 19:49:59,099 file: postprocess.py func: postprocess line No: 55 Batch:380/400; accuracy(all):0.6237
2024-08-22 19:49:59,099 file: postprocess.py func: postprocess line No: 55 Batch:390/400; accuracy(all):0.6205
2024-08-22 19:49:59,099 file: postprocess.py func: postprocess line No: 55 Batch:400/400; accuracy(all):0.6200
2024-08-22 19:49:59,100 file: tool_utils.py func: tool_utils line No: 144 ===REPORT-START{MAPPER-EVAL}===
2024-08-22 19:49:59,100 file: postprocess.py func: postprocess line No: 65 0.6200
2024-08-22 19:49:59,100 file: tool_utils.py func: tool_utils line No: 148 ===REPORT-END{MAPPER-EVAL}===
```


根据 [环境安装 | RDK DOC (d-robotics.cc)](https://developer.d-robotics.cc/rdk_doc/Advanced_development/toolchain_development/intermediate/environment_config?_highlight=hrt_model_exec#%E8%A1%A5%E5%85%85%E6%96%87%E4%BB%B6%E5%87%86%E5%A4%87) 一节，将算法工具链中板端所需要的 hrt_model_exec 和 hrt_bin_dump 可执行文件安装到 X5 上。进入文件夹  Ai_Toolchain_Package-release-vX.X.X-OE-vX.X.X/package/board 文件夹中并执行:

```shell
bash install.sh ${board_ip}
```

`${board_ip}` 是开发板设置的IP地址，并输入密码后便可安装到 X5 开发板上。执行完毕后，上传编译好的
RepVGG_224x224_nv12.bin 文件，执行以下命令：

```shell
hrt_model_exec perf --model_file RepVGG_224x224_nv12.bin \
                      --model_name="" \
                      --core_id=0 \
                      --frame_count=200 \
                      --perf_time=0 \
                      --thread_num=4 \
                      --profile_path="."
```

```shell
hrt_model_exec perf --model_file model/RepVGG_A2-deploy_224x224_nv12.bin --model_name= --core_id=0 --frame_count=200 --perf_time=0 --thread_num=4 --profile_path=.
I0000 00:00:00.000000 1283695 vlog_is_on.cc:197] RAW: Set VLOG level for "*" to 3
I0822 08:15:01.979738 1283695 main.cpp:233] profile_path: . exsit!
[BPU_PLAT]BPU Platform Version(1.3.6)!
[HBRT] set log level as 0. version = 3.15.47.0
[DNN] Runtime version = 1.23.5_(3.15.47 HBRT)
[A][DNN][packed_model.cpp:247][Model](2024-08-22,08:15:02.346.220) [HorizonRT] The model builder version = 1.23.8
[W][DNN]bpu_model_info.cpp:491][Version](2024-08-22,08:15:02.483.355) Model: RepVGG-deploy_224x224_nv12. Inconsistency between the hbrt library version 3.15.47.0 and the model build version 3.15.54.0 detected, in order to ensure correct model results, it is recommended to use compilation tools and the BPU SDK from the same OpenExplorer package.
Load model to DDR cost 508.656ms.
I0822 08:15:02.488767 1283695 function_util.cpp:323] get model handle success
I0822 08:15:02.488816 1283695 function_util.cpp:656] get model input count success
I0822 08:15:02.489009 1283695 function_util.cpp:687] prepare input tensor success!
I0822 08:15:02.489038 1283695 function_util.cpp:697] get model output count success
Frame count: 200,  Thread Average: 21.258760 ms,  thread max latency: 22.173000 ms,  thread min latency: 6.904000 ms,  FPS: 186.498093

Running condition:
  Thread number is: 4
  Frame count   is: 200
  Program run time: 1072.543000 ms
Perf result:
  Frame totally latency is: 4251.751953 ms
  Average    latency    is: 21.258760 ms
  Frame      rate       is: 186.472710 FPS
```

此时，模型量化的全过程就已经完成了。


## 5. 模型性能数据

以下表格是在 RDK X5 & RDK X5 Module 上实际测试得到的性能数据，可以根据自己推理实际需要的性能和精度，对模型的大小做权衡取舍


| 架构     | 模型      | 尺寸(像素)   | 类别数     | 参数量(M)   | 浮点精度     | 量化精度      | 延迟/吞吐量(单线程)   | 延迟/吞吐量(多线程)   | 帧率       |     |
| ----------- | ------------------------ | ------------- | ---------- | ----------- | ----------- | ----------- | ---------------- | ---------------- | ------------ | --- |
| Transformer | EdgeNeXt_base            | 224x224       | 1000       | 18.51       | 78.21       | 74.52       | 8.80             | 32.31            | 113.35       |     |
|             | EdgeNeXt_small           | 224x224       | 1000       | 5.59        | 76.50       | 71.75       | 4.41             | 14.93            | 226.15       |     |
|             | **EdgeNeXt_x_small**     | **224x224**   | **1000**   | **2.34**    | **71.75**   | **66.25**   | **2.88**         | **9.63**         | **345.73**   |     |
|             | EdgeNeXt_xx_small        | 224x224       | 1000       | 1.33        | 69.50       | 64.25       | 2.47             | 7.24             | 403.49       |     |
|             | EfficientFormer_l3       | 224x224       | 1000       | 31.3        | 76.75       | 76.05       | 17.55            | 65.56            | 60.52        |     |
|             | **EfficientFormer_l1**   | **224x224**   | **1000**   | **12.3**    | **76.12**   | **65.38**   | **5.88**         | **20.69**        | **191.605**  |     |
|             | EfficientFormerv2_s2     | 224x224       | 1000       | 12.6        | 77.50       | 70.75       | 6.99             | 26.01            | 152.40       |     |
|             | **EfficientFormerv2_s1** | **224x224**   | **1000**   | **6.1**     | **77.25**   | **68.75**   | **4.24**         | **14.35**        | **275.95**   |     |
|             | EfficientFormerv2_s0     | 224x224       | 1000       | 3.5         | 74.25       | 68.50       | 5.79             | 19.96            | 198.45       |     |
|             | **EfficientViT_MSRA_m5** | **224x224**   | **1000**   | **12.4**    | **73.75**   | **72.50**   | **6.34**         | **22.69**        | **174.70**   |     |
|             | FastViT_SA12             | 224x224       | 1000       | 10.9        | 78.25       | 74.50       | 11.56            | 42.45            | 93.44        |     |
|             | FastViT_S12              | 224x224       | 1000       | 8.8         | 76.50       | 72.0        | 5.86             | 20.45            | 193.87       |     |
|             | **FastViT_T12**          | **224x224**   | **1000**   | **6.8**     | **74.75**   | **70.43**   | **4.97**         | **16.87**        | **234.78**   |     |
|             | FastViT_T8               | 224x224       | 1000       | 3.6         | 73.50       | 68.50       | 2.09             | 5.93             | 667.21       |     |
| CNN         | FasterNet_S              | 224x224       | 1000       | 31.1        | 77.04       | 76.15       | 6.73             | 24.34            | 162.83       |     |
|             | FasterNet_T2             | 224x224       | 1000       | 15.0        | 76.50       | 76.05       | 3.39             | 11.56            | 342.48       |     |
|             | **FasterNet_T1**         | **224x224**   | **1000**   | **7.6**     | **74.29**   | **71.25**   | **1.96**         | **5.58**         | **708.40**   |     |
|             | FasterNet_T0             | 224x224       | 1000       | 3.9         | 71.75       | 68.50       | 1.41             | 3.48             | 1135.13      |     |
|             | RepVGG_B1g2              | 224x224       | 1000       | 41.36       | 77.78       | 68.25       | 9.77             | 36.19            | 109.61       |     |
|             | RepVGG_B1g4              | 224x224       | 1000       | 36.12       | 77.58       | 62.75       | 7.58             | 27.47            | 144.39       |     |
|             | RepVGG_B0                | 224x224       | 1000       | 14.33       | 75.14       | 60.36       | 3.07             | 9.65             | 410.55       |     |
|             | RepVGG_A2                | 224x224       | 1000       | 25.49       | 76.48       | 62.97       | 6.07             | 21.31            | 186.04       |     |
|             | **RepVGG_A1**            | **224x224**   | **1000**   | **12.78**   | **74.46**   | **62.78**   | **2.67**         | **8.21**         | **482.20**   |     |
|             | RepVGG_A0                | 224x224       | 1000       | 8.30        | 72.41       | 51.75       | 1.85             | 5.21             | 757.73       |     |
|             | RepViT_m1_1              | 224x224       | 1000       | 8.2         | 77.73       | 77.50       | 2.32             | 6.69             | 590.42       |     |
|             | **RepViT_m1_0**          | **224x224**   | **1000**   | **6.8**     | **76.75**   | **76.50**   | **1.97**         | **5.71**         | **692.29**   |     |
|             | RepViT_m0_9              | 224x224       | 1000       | 5.1         | 76.32       | 75.75       | 1.65             | 4.37             | 902.69       |     |
|             | MobileOne_S4             | 224x224       | 1000       | 14.8        | 78.75       | 76.50       | 4.58             | 15.44            | 256.52       |     |
|             | MobileOne_S3             | 224x224       | 1000       | 10.1        | 77.27       | 75.75       | 2.93             | 9.04             | 437.85       |     |
|             | MobileOne_S2             | 224x224       | 1000       | 7.8         | 74.75       | 71.25       | 2.11             | 6.04             | 653.68       |     |
|             | **MobileOne_S1**         | **224x224**   | **1000**   | **4.8**     | **72.31**   | **70.45**   | **1.31**         | **3.69**         | **1066.95**  |     |
|             | MobileOne_S0             | 224x224       | 1000       | 2.1         | 69.25       | 67.58       | 0.80             | 1.59             | 2453.17      |     |
|             | Mobilenetv2              | 224x224       | 1000       | 3.4         | 72.0        | 68.17       | 1.42             | 3.43             | 1152.07      |     |
|             | ResNet18                 | 224x224       | 1000       | 11.2        | 71.49       | 70.50       | 2.95             | 8.81             | 448.79       |     |



说明: 
1. X5的状态为最佳状态：CPU为8xA55@1.8G, 全核心Performance调度, BPU为1xBayes-e@1G, 共10TOPS等效int8算力。
2. 单线程延迟为单帧，单线程，单BPU核心的延迟，BPU推理一个任务最理想的情况。
3. 4线程工程帧率为4个线程同时向双核心BPU塞任务，一般工程中4个线程可以控制单帧延迟较小，同时吃满所有BPU到100%，在吞吐量(FPS)和帧延迟间得到一个较好的平衡。
4. 8线程极限帧率为8个线程同时向X3的双核心BPU塞任务，目的是为了测试BPU的极限性能，一般来说4核心已经占满，如果8线程比4线程还要好很多，说明模型结构需要提高"计算/访存"比，或者编译时选择优化DDR带宽。
5. 浮点/定点精度：浮点精度使用的是模型未量化前onnx的 Top-1 推理精度，量化精度则为量化后模型实际推理的精度。
6. 表格中粗体部分是在平衡推理速度和推理精度推荐选用的模型，可根据实际部署情况使用推理精度更高或推理速度更快的模型