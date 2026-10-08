- [COCO2017数据集介绍](#coco2017数据集介绍)
  - [参考 (Reference)](#参考-reference)
- [利用pycocotools工具验证COCO2017验证集5000张照片的目标检测的mAP精度](#利用pycocotools工具验证coco2017验证集5000张照片的目标检测的map精度)
  - [COCO2017 Val Images](#coco2017-val-images)
  - [COCO2017 Val Labels](#coco2017-val-labels)
  - [安装pycocotools](#安装pycocotools)
  - [准备预测值json](#准备预测值json)
  - [验证精度](#验证精度)
  - [结果解释](#结果解释)
  - [为什么pycocotools计算的精度比ultralytics YOLO计算的精度略低？](#为什么pycocotools计算的精度比ultralytics-yolo计算的精度略低)
- [ultralytics YOLO的pt模型推理得到pycocotools精度验证所需要的json文件](#ultralytics-yolo的pt模型推理得到pycocotools精度验证所需要的json文件)
  - [1. 加载模型：](#1-加载模型)
  - [2. 进行推理：](#2-进行推理)
  - [3. 解析`results`对象：](#3-解析results对象)
  - [完整代码](#完整代码)
- [ultralytics YOLO导出的onnx模型推理](#ultralytics-yolo导出的onnx模型推理)
  - [ONNXRuntime介绍](#onnxruntime介绍)
  - [前处理介绍](#前处理介绍)
  - [参考程序](#参考程序)
- [OpenExplore编译中间产物quantized\_model.onnx模型推理](#openexplore编译中间产物quantized_modelonnx模型推理)
  - [HB\_ONNXRuntime](#hb_onnxruntime)
  - [对应的编译配置yaml文件：](#对应的编译配置yaml文件)
  - [参考程序](#参考程序-1)




## COCO2017数据集介绍
MS COCO的全称是Microsoft Common Objects in Context，起源于微软于2014年出资标注的Microsoft COCO数据集，与ImageNet竞赛一样，被视为是计算机视觉领域最受关注和最权威的比赛之一。 

### 参考 (Reference)
[https://cocodataset.org/](https://cocodataset.org/)

[https://docs.ultralytics.com/zh/datasets/detect/coco/](https://docs.ultralytics.com/zh/datasets/detect/coco/)

[https://blog.csdn.net/weixin_50727642/article/details/122892088](https://blog.csdn.net/weixin_50727642/article/details/122892088)

## 利用pycocotools工具验证COCO2017验证集5000张照片的目标检测的mAP精度
### COCO2017 Val Images
获取COCO2017数据集的验证集, 一共有5000张照片.
```bash
wget -c http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
```

文件夹的结构如下所示
```bash
val2017
├── 000000000139.jpg
├── 000000000285.jpg
├── 000000000632.jpg
├── 000000000724.jpg
├── ... ...
├── 000000581615.jpg
└── 000000581781.jpg

0 directories, 5000 files
```

### COCO2017 Val Labels
获取COCO2017数据集的验证集标签, 是一个json文件, 里面有5000张照片的标注信息.
```bash
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

文件夹的结构如下所示, 这里主要是使用`instances_val2017.json`文件.
```bash
annotations
├── captions_train2017.json
├── captions_val2017.json
├── instances_train2017.json
├── instances_val2017.json
├── person_keypoints_train2017.json
└── person_keypoints_val2017.json

0 directories, 6 files
```
前缀的解释如下
- captions: 描述图片的文字.
- instances: 描述图片的物体, bbox和seg.
- person_keypoints: 描述图片的人体关键点.

### 安装pycocotools

1. 参考COCO2017的网站, 或者互联网的公开教程, 直接安装.
2. 使用OpenExplore的Docker.


### 准备预测值json
使用您的算法对COCO2017 Val验证集的5000张图片进行推理，这里需要按照以下形式来准备预测结果的json文件.

在Python中
```python
import json
import os

# 由于COCO有91个类别, 而ultralytics YOLO使用的是其中的80个类别, 所以需要将id映射到coco_id上.
coco_id = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,51,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90]

# 生成pycocotools的预测结果json文件
pridect_json = []

# COCO2017验证集路径
COCO2017_Val_Path = "./val2017/"

# 板端直接推理所有验证集，并生成标签
for img_name in os.listdir(opt.image_path):
    # 此处需要替换为您的算法推理逻辑
    img_path = os.path.join(opt.image_path, img_name)
    results = model.forward()   
    # 将算法推理结果填充到pridect_json中
    for class_id, score, x1, y1, x2, y2 in results:
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        x1, y1, x2, y2, width, height = float(x1), float(y1), float(x2), float(y2), float(width), float(height)
        pridect_json.append({
                'area': width * height,
                'bbox': [x1, y1, width, height],
                'category_id': int(coco_id[class_id]),
                'id': id_cnt,
                "score": float(score),
                'image_id': int(img_name[:-4]),
                'iscrowd': 0,
                'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
            })

    # 保存标签
    with open('predict_results.json', 'w') as f:
        json.dump(pridect_json, f, ensure_ascii=False, indent=1)

```

在C/C++中，您可以先将结果保存为一个txt文件，然后再使用Python将这个txt文件保存为json文件
```C++
#define TXT_RESULT_PATH "cpp_json_result.txt" // 临时保存推理结果的txt文件的路径
#define IMAGE_PATH "./val2017/"               // COCO2017验证集路径

// C/C++ Standard Librarys
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>
namespace fs = std::filesystem;
int main()
{
    // 遍历验证集目录下的所有图像
    std::vector<fs::path> file_paths;
    try
    {
        if (!fs::exists(IMAGE_PATH) || !fs::is_directory(IMAGE_PATH))
        {
            std::cerr << "The provided path is not a valid directory.\n";
            return 1;
        }
        for (const auto &entry : fs::directory_iterator(IMAGE_PATH))
        {
            if (fs::is_regular_file(entry.status()))
                file_paths.push_back(entry.path());
        }
    }
    catch (const fs::filesystem_error &e)
    {
        std::cerr << "Filesystem error: " << e.what() << '\n';
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << '\n';
    }

    // 板端直接推理所有验证集，并生成*.txt标签
    size_t image_total_cnt = file_paths.size();
    std::stringstream cpp_results;

    for (size_t i = 0; i < image_total_cnt; ++i)
    {
        // 此处需要替换为您的算法推理逻辑
        // img_path = file_paths[i].string()
        // result = model.forward()

        for (auto result : results)
        {
            // 保存pycocotools所需要的标签信息
            cpp_results << file_paths[i].filename().string() << " ";
            cpp_results << result.cls_id << " " << result.score << " ";
            cpp_results << result.x1 << " " << result.y1 << " " << result.x2 << " " << result.y2 << std::endl;
        }
    }

    // 将pycocotools所需要的标签信息保存为txt文件
    std::ofstream file_stream(TXT_RESULT_PATH);
    if (!file_stream.is_open())
    {
        std::cerr << "Failed to open/create the file: " << TXT_RESULT_PATH << '\n';
        return 1;
    }

    file_stream << cpp_results.rdbuf();
    file_stream.close();

    // 验证文件是否已经正确关闭
    if (file_stream.is_open())
        std::cerr << "File did not close properly.\n";
    else
        std::cout << "result label saved: \"" << TXT_RESULT_PATH << "\"" << std::endl;
    return 0;
}
```
然后利用以下Python脚本将这个txt文件转化为pycocotools工具所需要的json文件 
```python
import json
import os

coco_id = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,51,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90]

# 生成pycocotools的预测结果json文件
pridect_json = []
id_cnt = 0
with open('cpp_json_result.txt', 'r') as file:
    for line in file:
        # 去除行末尾的换行符并按照空格分割行内容
        parts = line.strip().split()

        # 解析各个字段
        img_name = parts[0]  # 去除可能存在的引号
        class_id = int(parts[1])
        score = float(parts[2])
        x1 = float(parts[3])
        y1 = float(parts[4])
        x2 = float(parts[5])
        y2 = float(parts[6])

        # 将解析后的数据添加到列表中
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        x1, y1, x2, y2, width, height = float(x1), float(y1), float(x2), float(y2), float(width), float(height)
        pridect_json.append({
                'area': width * height,
                'bbox': [x1, y1, width, height],
                'category_id': int(coco_id[class_id]),
                'id': id_cnt,
                "score": float(score),
                'image_id': int(img_name[:-4]),
                'iscrowd': 0,
                'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
            })
        id_cnt += 1
# 保存标签
with open(opt.json_path, 'w') as f:
    json.dump(pridect_json, f, ensure_ascii=False, indent=1)
```


### 验证精度

此代码段展示了如何使用Python中的pycocotools库来评估目标检测模型在COCO数据集上的性能。首先，通过加载标准的验证集标注文件和模型预测结果文件初始化COCO对象。接着，创建一个COCOeval对象来进行评估，其中指定了评估类型为边界框（bbox）。之后，依次调用evaluate()、accumulate()和summarize()方法来完成从计算到汇总并输出评估结果的过程。这个过程可以帮助理解模型的精确度和召回率表现，是衡量模型性能的重要步骤。
```python
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 加载验证集的标准标注文件（真值），'instances_val2017.json'是COCO数据集中验证集的标准标注文件路径
coco_true = COCO(annotation_file='instances_val2017.json')  
# 使用loadRes方法将预测结果文件加载进来作为预测数据集。'predict_results.json'应包含模型对验证集的预测结果
# 注意：预测结果需要按照COCO API要求的格式生成，并保存为json文件
coco_pre = coco_true.loadRes('predict_results.json')  
# 初始化COCOeval对象，用于计算边界框（bbox）类型的IoU（Intersection over Union）指标
# cocoGt参数是标准数据集（Ground Truth），cocoDt是预测数据集，iouType指定评估类型为"bbox"
coco_evaluator = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType="bbox")    
# 执行评估函数，计算每个图像上的各项指标
coco_evaluator.evaluate()
# 累积所有图像上的评估结果，准备进行总结
coco_evaluator.accumulate()
# 输出最终的评估结果摘要，包括AP（Average Precision）、AR（Average Recall）等重要指标
coco_evaluator.summarize()
```

### 结果解释
运行上述程序，会获得以下结果，这些结果展示了模型在不同条件下的性能表现，包括不同的IoU阈值、目标尺寸以及最大检测数量限制。通过分析这些数据，可以了解到模型在哪种情况下表现最佳或有待改进，例如它可能对大目标有更好的检测效果，而对小目标的检测则相对困难。这有助于指导进一步的模型优化工作。
```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.465
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.606
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.509
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.265
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.519
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.656
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.355
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.523
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.530
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.300
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.584
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.733
```
Average Precision (AP): 平均精度，衡量模型对不同IoU（Intersection over Union）阈值的目标检测准确性。
  - `AP @[ IoU=0.50:0.95 | area=all | maxDets=100 ] = 0.465`: 这是主要的mAP（mean Average Precision），计算了从IoU=0.5到IoU=0.95（步长为0.05）的所有类别和所有目标大小的平均精度，最大检测数设为100。这个值为0.465。
  - `AP @[ IoU=0.50 | area=all | maxDets=100 ] = 0.606`: 当IoU阈值设置为0.5时，所有目标大小的平均精度。
  - `AP @[ IoU=0.75 | area=all | maxDets=100 ] = 0.509`: 当IoU阈值设置为0.75时，所有目标大小的平均精度。
  - 对于不同大小的目标（小、中、大），分别在IoU范围0.5至0.95下的平均精度，小目标（small）：0.265，中等目标（medium）：0.519，大目标（large）：0.656

Average Recall (AR): 平均召回率，衡量模型正确识别出的目标占所有实际目标的比例。
  - `AR @[ IoU=0.50:0.95 | area=all | maxDets=1 ] = 0.355`: 在最大检测数量为1的情况下，所有目标大小的平均召回率。
  - `AR @[ IoU=0.50:0.95 | area=all | maxDets=10 ] = 0.523`: 在最大检测数量为10的情况下，所有目标大小的平均召回率。
  - `AR @[ IoU=0.50:0.95 | area=all | maxDets=100 ] = 0.530`: 在最大检测数量为100的情况下，所有目标大小的平均召回率。
  - 对于不同大小的目标（小、中、大），分别在IoU范围0.5至0.95下的平均召回率，小目标（small）：0.300，中等目标（medium）：0.584，大目标（large）：0.733




### 为什么pycocotools计算的精度比ultralytics YOLO计算的精度略低？
- ultralytics官方测mAP时，使用动态shape模型, 而BPU使用了固定shape模型，map测试结果会比动态shape的低一些。
- 在计算IOU-AP曲线下面积时，ultralytics YOLO采用了线性插值，pycocotools使用了临近插值，简单理解就是前者是使用梯形去近似，后者是使用矩形去近似，导致了两者计算存在的差异，尤其在PR曲线波动比较大的类别，两者的结果会差距更大。
- 我们主要是关注同样的一套计算方式去测试定点模型和浮点模型的精度, 包括Score和IOU的阈值，从而来评估量化过程中的精度损失.
- BPU 模型在量化和NCHW-RGB888输入转换为YUV420SP(nv12)输入后, 也会有一部分精度损失。


## ultralytics YOLO的pt模型推理得到pycocotools精度验证所需要的json文件

本文介绍了如何使用原版的ultralytics算法框架和原版的pt模型进行推理，通过解析推理后的`results`对象提取并填充到最终pycocotools在精度验证时所需要的JSON输出中所需的各种变量.
### 1. 加载模型：
```python
model = YOLO('yolo11s.pt')
```
这里实例化了一个YOLO模型，使用预训练权重`yolo11s.pt`。

### 2. 进行推理：
```python
results = model([os.path.join(opt.image_path, img_name)], conf=0.45, iou=0.25)
```
对每张图片进行推理，返回的结果是一个包含检测信息的对象列表。每个对象包含了多个属性如边界框、类别ID、置信度分数等。

### 3. 解析`results`对象：
- `results[0].boxes.cls`: 包含所有检测目标的类别ID（tensor形式）。通过`int(class_id)`将其转换为整数。
- `results[0].boxes.conf`: 每个检测目标的置信度分数（tensor形式），需要转换为浮点数。
- `results[0].boxes.xyxy`: 边界框的位置信息，采用的是[x1, y1, x2, y2]格式，其中(x1, y1)是左上角坐标，(x2, y2)是右下角坐标。这些值也需要转换成整数或浮点数，具体取决于它们在最终JSON中的表示方式。

### 完整代码
```python
from ultralytics import YOLO

import json
import logging 
import os

def main():
    # id -> coco_id
    coco_id = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,51,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90]

    # 实例化
    model = YOLO('yolo11s.pt')
    
    # 生成pycocotools的预测结果json文件
    pridect_json = []

    # 板端直接推理所有验证集，并生成标签和渲染结果 (串行程序)
    img_names = os.listdir(opt.image_path)
    for cnt, img_name in enumerate(img_names, 1):
        # 端到端推理
        results = model([os.path.join(opt.image_path, img_name)], conf=0.45, iou=0.25)
        # 保存到JSON
        id_cnt = 0
        for i in range(len(results[0].boxes.cls)):
            class_id, score, xyxy = results[0].boxes.cls[i], results[0].boxes.conf[i], results[0].boxes.xyxy[i]
            class_id = int(class_id)
            score = float(score)
            x1, x2, y1, y2 = int(xyxy[0]), int(xyxy[2]), int(xyxy[1]), int(xyxy[3])
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            x1, y1, x2, y2, width, height = float(x1), float(y1), float(x2), float(y2), float(width), float(height)
            pridect_json.append({
                    'area': width * height,
                    'bbox': [x1, y1, width, height],
                    'category_id': int(coco_id[class_id]),
                    'id': id_cnt,
                    "score": float(score),
                    'image_id': int(img_name[:-4]),
                    'iscrowd': 0,
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
            id_cnt += 1

    # 保存标签
    with open(opt.json_path, 'w') as f:
        json.dump(pridect_json, f, ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main()
```


## ultralytics YOLO导出的onnx模型推理

本文主要讲述如何使用ultralytics YOLO导出的onnx模型如何推理，要点在于onnxruntime框架的使用和对应的前处理。

### ONNXRuntime介绍
ONNX Runtime不仅仅是一个开源的深度学习框架，它更专注于提供高效的模型推理能力。支持多种硬件平台和操作系统，包括但不限于Windows、Linux、macOS等，并且可以运行在CPU、GPU（通过CUDA、DirectML或OpenVINO）、以及各类移动设备上。ONNX Runtime的核心优势在于其优化的执行引擎，该引擎能够显著提升模型推理的速度，同时降低资源消耗。此外，ONNX Runtime还支持自动混合精度，这使得模型可以在保持高精度的同时减少计算量和内存使用。


### 前处理介绍
前处理是深度学习模型推理过程中的关键步骤之一，特别是对于计算机视觉任务而言。以YOLO系列模型为例，前处理通常包括以下几个步骤：

- 尺寸调整：根据模型训练时使用的输入尺寸调整图像大小。常见的做法是保持图像的长宽比不变，同时将最短边缩放到指定长度（例如640像素），然后填充剩余空间以达到目标尺寸。
- 色彩空间转换：大多数预训练的模型都是基于RGB格式的图像进行训练的。因此，如果输入图像是BGR格式（如OpenCV默认读取的图像格式），需要将其转换为RGB格式。
- 归一化：为了使输入数据适应模型训练所用的数据分布，通常会对图像数据进行归一化处理。这一步骤可能包括将像素值从[0, 255]范围映射到[0, 1]范围，或者减去平均值后除以标准差。
- 维度变换：最后，还需要对图像数据进行维度上的调整，以便符合模型输入的要求。一般情况下，图像数据需要被转换成NCHW格式（批量大小、通道数、高度、宽度）。
上述步骤确保了输入数据与模型预期的输入格式相匹配，从而保证了模型推理的准确性。

### 参考程序
```python
import cv2
import numpy as np
import onnxruntime as ort

class ONNX_YOLOv8_Detect():
    def __init__(self, opt):
        # 利用ONNXRuntime加载onnx模型
        self.session = ort.InferenceSession(opt.model_path)

        # 打印onnx模型的输入信息
        print("-> input tensors")
        for cnt, inp in enumerate(self.session.get_inputs()):
            print(f"input[{cnt}] - Name: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")

        # 打印onnx模型的输出信息
        print("-> output tensors")
        for cnt, out in enumerate(self.session.get_outputs()):
            print(f"output[{cnt}] - Name: {out.name}, Shape: {out.shape}, Type: {out.type}")

    def preprocess(self, img):
        # 这里是letter box的处理过程
        self.x_scale = min(1.0 * self.input_H / self.img_h, 1.0 * self.input_W / self.img_w)
        self.y_scale = self.x_scale
        
        if self.x_scale <= 0 or self.y_scale <= 0:
            raise ValueError("Invalid scale factor.")

        new_w = int(self.img_w * self.x_scale)
        self.x_shift = (self.input_W - new_w) // 2
        x_other = self.input_W - new_w - self.x_shift
        new_h = int(self.img_h * self.y_scale)
        self.y_shift = (self.input_H - new_h) // 2
        y_other = self.input_H - new_h - self.y_shift
        
        # 对图像进行缩放，这时的图像是长或者宽是640
        input_tensor = cv2.resize(img, (new_w, new_h))

        # 对缩放后的图像进行padding，将长和宽均变为640
        input_tensor = cv2.copyMakeBorder(input_tensor, self.y_shift, y_other, self.x_shift, x_other, cv2.BORDER_CONSTANT, value=[127, 127, 127])

        # 将BGR图转化为RGB图，这里是对C通道里面的3个数字进行一次变化
        input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2RGB)

        # 对每个数据进行归一化，使其分布在0～1之间
        input_tensor = np.array(input_tensor) / 255.0 

        # 这里将HWC的数据变为CHW的数据，数据从(640, 640, 3) -> (3, 640, 640)
        input_tensor = np.transpose(input_tensor, (2, 0, 1))

        # 这里将数据转化为4维数据，NCHW，数据从(3, 640, 640) -> (1, 3, 640, 640)
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)  # NCHW
        return input_tensor

    def forward(self, input_tensor):
        # 调用onnxruntime的session.run方法进行推理
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: input_tensor})
        return outputs

    def postProcess(self, outputs):
        # 这里的onnx有6个输出头，按照列表进行索引
        s_clses = outputs[0].reshape(-1, self.CLASSES_NUM)
        s_bboxes = outputs[1].reshape(-1, self.REG * 4)
        m_clses = outputs[2].reshape(-1, self.CLASSES_NUM)
        m_bboxes = outputs[3].reshape(-1, self.REG * 4)
        l_clses = outputs[4].reshape(-1, self.CLASSES_NUM)
        l_bboxes = outputs[5].reshape(-1, self.REG * 4)

        # 后处理过程请参考RDK Model Zoo仓库，此处不再赘述
        # https://github.com/D-Robotics/rdk_model_zoo

        return results
```




## OpenExplore编译中间产物quantized_model.onnx模型推理

本文主要讲述如何使用OpenExplore编译中间产物quantized_model.onnx模型如何推理，要点在于HB_ONNXRuntime框架的使用和对应的前处理。

### HB_ONNXRuntime
HB_ONNXRuntime是地平线基于公版ONNXRuntime封装的一套x86端的ONNX模型推理库。 除支持Pytorch、TensorFlow、PaddlePaddle、MXNet等各训练框架直接导出的ONNX原始模型外，还支持对地平线工具链进行PTQ转换过程中产出的各阶段ONNX模型进行推理。
可以在OpenExplore提供的Docker中直接使用


### 对应的编译配置yaml文件：
这里的onnx模型是NCHW-RGB888的输入，在runtime时，我们将其配置为了YUV420SP(nv12)的输入，同时选择让BPU来进行除以255（乘以255分之一）的操作。
为什么算法数据选择NCHW-RGB888，而Runtime数据选择YUV420SP，这里不再赘述。当然，最优的情况应该是算法数据选择YUV420SP，将其处理为NCHW-YUV444再进行算法训练，Runtime数据选择YUV420SP。

```yaml
input_parameters:
  input_type_rt: 'nv12'
  input_type_train: 'rgb'
  input_layout_train: 'NCHW'
  norm_type: 'data_scale'
  scale_value: 0.003921568627451
```


### 参考程序
```python
from horizon_tc_ui import HB_ONNXRuntime

class HB_ONNX_YOLOv8_Detect():
    def __init__(self, opt):
        # 利用HB_ONNXRuntime加载quantized_model.onnx模型
        self.session = HB_ONNXRuntime(opt.model_path)

        # 打印quantized_model.onnx模型的输入信息
        print("-> input tensors")
        for cnt, inp in enumerate(self.session.get_inputs()):
            print(f"input[{cnt}] - Name: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")

        # 打印quantized_model.onnx模型的输入信息
        print("-> output tensors")
        for cnt, out in enumerate(self.session.get_outputs()):
            logger.info(f"output[{cnt}] - Name: {out.name}, Shape: {out.shape}, Type: {out.type}")


    def yuv444_preprocess(self, img):
        # cv2的BGR图转YUV420SP图
        def bgr2nv12(image): 
            image = image.astype(np.uint8) 
            height, width = image.shape[0], image.shape[1] 
            yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape((height * width * 3 // 2, )) 
            y = yuv420p[:height * width] 
            uv_planar = yuv420p[height * width:].reshape((2, height * width // 4)) 
            uv_packed = uv_planar.transpose((1, 0)).reshape((height * width // 2, )) 
            nv12 = np.zeros_like(yuv420p) 
            nv12[:height * width] = y 
            nv12[height * width:] = uv_packed 
            return nv12 
        # YUV420SP图转YUV444图
        def nv12Toyuv444(nv12, target_size): 
            height = target_size[0] 
            width = target_size[1] 
            nv12_data = nv12.flatten() 
            yuv444 = np.empty([height, width, 3], dtype=np.uint8) 
            yuv444[:, :, 0] = nv12_data[:width * height].reshape(height, width) 
            u = nv12_data[width * height::2].reshape(height // 2, width // 2) 
            yuv444[:, :, 1] = Image.fromarray(u).resize((width, height),resample=0) 
            v = nv12_data[width * height + 1::2].reshape(height // 2, width // 2) 
            yuv444[:, :, 2] = Image.fromarray(v).resize((width, height),resample=0) 
            return yuv444 

        # lettet box的部分
        self.x_scale = min(1.0 * self.input_H / self.img_h, 1.0 * self.input_W / self.img_w)
        self.y_scale = self.x_scale
        
        if self.x_scale <= 0 or self.y_scale <= 0:
            raise ValueError("Invalid scale factor.")
        
        new_w = int(self.img_w * self.x_scale)
        self.x_shift = (self.input_W - new_w) // 2
        x_other = self.input_W - new_w - self.x_shift
        
        new_h = int(self.img_h * self.y_scale)
        self.y_shift = (self.input_H - new_h) // 2
        y_other = self.input_H - new_h - self.y_shift
        
        input_tensor = cv2.resize(img, (new_w, new_h))
        input_tensor = cv2.copyMakeBorder(input_tensor, self.y_shift, y_other, self.x_shift, x_other, cv2.BORDER_CONSTANT, value=[127, 127, 127])
        
        # 色彩空间转化的部分
        input_tensor = bgr2nv12(input_tensor)
        yuv444 = nv12Toyuv444(input_tensor, (640,640))
        yuv444 = yuv444[np.newaxis,:,:,:]

        # -128并从uint8转为int8，注意，这里已经没有除以255的计算了
        input_tensor = (yuv444-128).astype(np.int8)

        return input_tensor


    def forward(self, input_tensor):
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: input_tensor})
        return outputs

    def postProcess(self, outputs):
        # 这里的quantized_model.onnx有6个输出头，按照列表进行索引
        s_clses = outputs[0].reshape(-1, self.CLASSES_NUM)
        s_bboxes = outputs[1].reshape(-1, self.REG * 4)
        m_clses = outputs[2].reshape(-1, self.CLASSES_NUM)
        m_bboxes = outputs[3].reshape(-1, self.REG * 4)
        l_clses = outputs[4].reshape(-1, self.CLASSES_NUM)
        l_bboxes = outputs[5].reshape(-1, self.REG * 4)

        # 后处理过程请参考RDK Model Zoo仓库，此处不再赘述
        # https://github.com/D-Robotics/rdk_model_zoo

        return results
```