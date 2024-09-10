English | [简体中文](./模型量化部署.md)

# Model quantization deployment

- [Model quantization deployment](#model-quantization-deployment)
  - [1. Model Preparation](#1-model-preparation)
  - [2. Model Checking](#2-model-checking)
  - [3. Calibration Set Construction](#3-calibration-set-construction)
  - [4. Model Quantization and Compiling](#4-model-quantization-and-compiling)
  - [5. Model Inference](#5-model-inference)

Model quantification deployment is an important part of edge-side upper board operation. This document refers to the quantification deployment section of Tiangong Kaiwu toolchain and simplifies some content to facilitate users to quickly get started with development. You can also refer to the document link for a deeper understanding and application of the toolchain.

Toolchain documentation link: https://developer.d-robotics.cc/rdk_doc/en/04_toolchain_development

Take the mode [RepVGG](./RepVGG/README.md) as an example to quantify the deployment process of onnx model

## 1. Model Preparation

First, use the git command to download the RepVGG source code:

```shell
git clone https://github.com/DingXiaoH/RepVGG.git
```

Weight file link: https://drive.google.com/drive/folders/1Avome4KvNp0Lqh2QwhXO6L5URQjzCjUq

Since the model is not entered in the timm library or the built-in model framework of pytorch, and [RepVGG](./RepVGG/README.md)  is a very friendly model for deployment, it is necessary to download the model code for model conversion operation. According to the "Environment Deployment" section of the [Environment Installation | RDK DOC (d-robotics.cc)](https://developer.d-robotics.cc/rdk_doc/en/Advanced_development/toolchain_development/intermediate/environment_config), install the Docker environment of the developer computer. After installing the Docker environment on the developer computer, enable the Docker environment and create a new Python file in the source code of [RepVGG](./RepVGG/README.md) git in your developer computer. The content is as follows:

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
    model = repvgg_model_convert(model) # Reparameterization

    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")
    onnx_file_path = "RepVGG-A2.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        opset_version=11,
        verbose=True,
        input_names=["data"],  # enter name
        output_names=["output"],  # output name 
    )
```

Run this code to generate an onnx file for the RepVGG inference model

## 2. Model Checking

After exporting the RepVGG-A2.onnx onnx model, it is necessary to check the model, mainly to see if there is a situation where the data shape in the operator does not match and the input data is not fixed (the classification model does not support dynamic multi-Batch and multi-Shape input for the time being). Create a new `01_check.sh` and add the following Shell script code and execute the `sh 01_check.sh` command:

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

The generated part of the file log `hb_mapper_checker.log` is as follows:

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

It can be seen that all operators of [RepVGG](./RepVGG/README.md) can run on BPU, and the model is int8 quantization, which does not need to be quantized to int16 format, making it more friendly for deployment inference.


## 3. Calibration Set Construction

As mentioned earlier, the model is trained using the ImageNet dataset. In the calibration step, 100-300 datasets need to be prepared to calibrate the model. The model dataset can be calibrated using the calibration dataset provided by Tiangong Kaiwu toolchain, or you can directly download the ILSVRC2012 test set and randomly select about 100 images (about 10G). Create a new preprocess.py Python preprocessing file and add the following code:

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

Create a `02_preprocess.sh` script file, add the following code and execute the `sh 02_preprocess.sh`:

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

This script file will call the calibration_transformers function in Python code, which adjusts the short side of the image to 256 pixels, keeps the aspect ratio unchanged, crops an image block with a size of 224 x 224 from the center of the image, and converts the image from NHWC (channel at the end) format to NCHW (channel before) format (according to the documentation of the toolchain, the input of the quantization model must be in channel before format), and finally converts the image from OpenCV's BGR format to the RGB format of the model input. The output after conversion is as follows:

```shell
write:./calibration_data_rgb_f32/ILSVRC2012_val_00000092.rgb
write:./calibration_data_rgb_f32/ILSVRC2012_val_00000096.rgb
write:./calibration_data_rgb_f32/ILSVRC2012_val_00000079.rgb
write:./calibration_data_rgb_f32/ILSVRC2012_val_00000099.rgb
write:./calibration_data_rgb_f32/ILSVRC2012_val_00000100.rgb
write:./calibration_data_rgb_f32/ILSVRC2012_val_00000098.rgb
write:./calibration_data_rgb_f32/ILSVRC2012_val_00000097.rgb
```

## 4. Model Quantization and Compiling

After the model preparation, model checking and calibration set building steps, you can enter the most important step - model quantization and compile. Model transformation requires configuring the yaml configuration file of the model. The configuration parameters in the file can refer to the configuration template and [Explanation of Parameters in Model Conversion YAML Configuration](https://developer.d-robotics.cc/rdk_doc/en/Advanced_development/toolchain_development/intermediate/ptq_process#model-conversion) For specific parameter information, create a new `RepVGG_deploy_config.yaml` and write the following configuration:

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

Create a new 03_build.sh script file, add the following code and execute `sh 03_build.sh`：

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

The file log generated during the quantization process hb_mapper_makerbin as follows:

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

Since RepVGG is a serial VGG-like structure, it **does not require int16 quantization, so the quantization tuning measurement tends to change the internal fusion measurement and quantization correction method of the model, and the public version model quantization cosine similarity is low**. The reference model can be downloaded according to the toolchain reference document to further optimize the model.

Of course, if you want to further optimize the resulting model_output/RepVGG_224x224_nv12.bin file, you can execute the following instructions:

```shell
hb_model_modifier RepVGG_224x224_nv12/RepVGG_224x224_nv12.bin -r /linear/Gemm_reshape_output
``` 

This instruction removes the last Reshape operator from the model, causing the model to output as INT32, **enabling all operators to run on the BPU**.

## 5. Model Inference

Create postprocess.py Python post-processing file and add the following code and images:

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

Create a new `cls_inference.py` and add the following:

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

Create a new `04_inference.sh` script file, add the following code and execute `sh 04_inference.sh`, the final output is as follows:

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

It can be seen that the output Confidence Level is 0.9989, and the Confidence Level for inferring a single image is still very high.

Create a new cls_evaluate.py and add the following code:

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

Create a new 05_evaluate.sh script file and add the following code. In addition, you need to create a new val folder to store the ImageNet validation set. The address is here: https://pan.baidu.com/s/1MEjNh6evha2hcdrQXjNv8w?pwd=yzza val.txt and val_piece.txt are the test tags that come with the toolchain. You can directly copy and paste them.

Execute sh 05_evaluate.sh quanti test. The final output is as follows:

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


According to [Explanation of Parameters in Model Conversion YAML Configuration](https://developer.d-robotics.cc/rdk_doc/en/Advanced_development/toolchain_development/intermediate/ptq_process#model-conversion) section, install the required hrt_model_exec and hrt_bin_dump executable files from the board end of the algorithm toolchain onto X5. Go to the folder Ai_Toolchain_Package -release-vX.X.X-OE-vX.X/package/board and execute:

```shell
bash install.sh ${board_ip}
```

`${board_ip}` is the IP Address set by the development board, and after entering the password, it can be installed on the X5 development board. After execution, upload the compiled
RepVGG_224x224_nv12 file, execute the following command:

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

At this point, the entire process of model quantification has been completed.