/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

Copyright (c) 2024，WuChao D-Robotics.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// 注意: 此程序在RDK板端端运行
// Attention: This program runs on RDK board.

#include <omp.h>
#include <fstream>

// D-Robotics *.bin 模型路径
// Path of D-Robotics *.bin model.
#define MODEL_PATH "yolov8n_instance_seg_bayese_640x640_nv12_modified.bin"

// 推理使用的测试图片路径
// Path of the test image used for inference.
#define TESR_IMG_PATH "../../../../../../resource/datasets/COCO2017/assets/bus.jpg"

// 前处理方式选择, 0:Resize, 1:LetterBox
// Preprocessing method selection, 0: Resize, 1: LetterBox
#define RESIZE_TYPE 0
#define LETTERBOX_TYPE 1
#define PREPROCESS_TYPE LETTERBOX_TYPE

// 推理结果保存路径
// Path where the inference result will be saved
#define IMG_SAVE_PATH "cpp_result.jpg"

// 模型的类别数量, 默认80
// Number of classes in the model, default is 80
#define CLASSES_NUM 1

// NMS的阈值, 默认0.45
// Non-Maximum Suppression (NMS) threshold, default is 0.45
#define NMS_THRESHOLD 0.7

// 分数阈值, 默认0.25
// Score threshold, default is 0.25
#define SCORE_THRESHOLD 0.25

// NMS选取的前K个框数, 默认300
// Number of top-K boxes selected by NMS, default is 300
#define NMS_TOP_K 300

// 控制回归部分离散化程度的超参数, 默认16
// A hyperparameter that controls the discretization level of the regression part, default is 16
#define REG 16

// 用于合成最终Mask的系数的数量
// Mask Coefficients
#define MCES 32

// 是否生成和绘制轮廓点
// Whether to generate and draw contour points
#define IS_POINT True

// 绘制标签的字体尺寸, 默认1.0
// Font size for drawing labels, default is 1.0.
#define FONT_SIZE 1.0

// 绘制标签的字体粗细, 默认 1.0
// Font thickness for drawing labels, default is 1.0.
#define FONT_THICKNESS 1.0

// 绘制矩形框的线宽, 默认2.0
// Line width for drawing bounding boxes, default is 2.0.
#define LINE_SIZE 2.0

#define X_SCALE_CORP 0.25
#define Y_SCALE_CORP 0.25

// C/C++ Standard Librarys
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

// Thrid Party Librarys
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

// RDK BPU libDNN API
#include "dnn/hb_dnn.h"
#include "dnn/hb_dnn_ext.h"
#include "dnn/plugin/hb_dnn_layer.h"
#include "dnn/plugin/hb_dnn_plugin.h"
#include "dnn/hb_sys.h"

#define RDK_CHECK_SUCCESS(value, errmsg)                                         \
    do                                                                           \
    {                                                                            \
        auto ret_code = value;                                                   \
        if (ret_code != 0)                                                       \
        {                                                                        \
            std::cout << "[ERROR] " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::cout << errmsg << ", error code:" << ret_code << std::endl;     \
            return ret_code;                                                     \
        }                                                                        \
    } while (0);

// COCO Names
std::vector<std::string> object_names = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

// YOLO colors
std::vector<cv::Scalar> rdk_colors = {
    cv::Scalar(56, 56, 255), cv::Scalar(151, 157, 255), cv::Scalar(31, 112, 255), cv::Scalar(29, 178, 255), cv::Scalar(49, 210, 207), cv::Scalar(10, 249, 72), cv::Scalar(23, 204, 146), cv::Scalar(134, 219, 61), cv::Scalar(52, 147, 26), cv::Scalar(187, 212, 0), cv::Scalar(168, 153, 44), cv::Scalar(255, 194, 0), cv::Scalar(147, 69, 52), cv::Scalar(255, 115, 100), cv::Scalar(236, 24, 0), cv::Scalar(255, 56, 132), cv::Scalar(133, 0, 82), cv::Scalar(255, 56, 203), cv::Scalar(200, 149, 255), cv::Scalar(199, 55, 255)};

int main()
{
    // 0. 加载bin模型
    // 0. Load bin model
    auto begin_time = std::chrono::system_clock::now();
    hbPackedDNNHandle_t packed_dnn_handle;
    const char *model_file_name = MODEL_PATH;
    RDK_CHECK_SUCCESS(
        hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1),
        "hbDNNInitializeFromFiles failed");
    std::cout << "\033[31m Load D-Robotics Quantize model time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 << " ms\033[0m" << std::endl;

    // 1. 打印相关版本信息
    // std::cout << "OpenCV build details: " << cv::getBuildInformation() << std::endl;
    std::cout << "[INFO] OpenCV Version: " << CV_VERSION << std::endl;
    std::cout << "[INFO] MODEL_PATH: " << MODEL_PATH << std::endl;
    std::cout << "[INFO] CLASSES_NUM: " << CLASSES_NUM << std::endl;
    std::cout << "[INFO] NMS_THRESHOLD: " << NMS_THRESHOLD << std::endl;
    std::cout << "[INFO] SCORE_THRESHOLD: " << SCORE_THRESHOLD << std::endl;

    // 2. 打印模型信息
    // 2. Show model message

    // 2.1 模型名称
    // 2.1 Model names
    const char **model_name_list;
    int model_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle),
        "hbDNNGetModelNameList failed");

    // 如果这个bin模型有多个打包，则只使用第一个，一般只有一个
    // If this bin model has multiple packages, only the first one is used, usually there is only one.
    if (model_count > 1)
    {
        std::cout << "This model file have more than 1 model, only use model 0.";
    }
    const char *model_name = model_name_list[0];
    std::cout << "[model name]: " << model_name << std::endl;

    // 2.2 获得Packed模型的第一个模型的handle
    // 2.2 Get the handle of the first model in the packed model
    hbDNNHandle_t dnn_handle;
    RDK_CHECK_SUCCESS(
        hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name),
        "hbDNNGetModelHandle failed");

    // 2.3 模型输入检查
    // 2.3 Model input check
    int32_t input_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetInputCount(&input_count, dnn_handle),
        "hbDNNGetInputCount failed");

    hbDNNTensorProperties input_properties;
    RDK_CHECK_SUCCESS(
        hbDNNGetInputTensorProperties(&input_properties, dnn_handle, 0),
        "hbDNNGetInputTensorProperties failed");

    // 2.3.1 D-Robotics YOLOV8-Seg *.bin 模型应该为单输入
    // 2.3.1 D-Robotics YOLOV8-Seg *.bin model should have only one input
    if (input_count > 1)
    {
        std::cout << "Your Model have more than 1 input, please check!" << std::endl;
        return -1;
    }

    // 2.3.2 D-Robotics YOLOV8-Seg *.bin 模型输入Tensor类型应为nv12
    // tensor type: HB_DNN_IMG_TYPE_NV12
    if (input_properties.tensorType == HB_DNN_IMG_TYPE_NV12)
    {
        std::cout << "input tensor type: HB_DNN_IMG_TYPE_NV12" << std::endl;
    }
    else
    {
        std::cout << "input tensor type is not HB_DNN_IMG_TYPE_NV12, please check!" << std::endl;
        return -1;
    }

    // 2.3.3 D-Robotics YOLOV8-Seg *.bin 模型输入Tensor数据排布应为NCHW
    // tensor layout: HB_DNN_LAYOUT_NCHW
    if (input_properties.tensorLayout == HB_DNN_LAYOUT_NCHW)
    {
        std::cout << "input tensor layout: HB_DNN_LAYOUT_NCHW" << std::endl;
    }
    else
    {
        std::cout << "input tensor layout is not HB_DNN_LAYOUT_NCHW, please check!" << std::endl;
        return -1;
    }

    // 2.3.4 D-Robotics YOLOV8-Seg *.bin 模型输入Tensor数据的valid shape应为(1,3,H,W)
    // valid shape: (1,3,640,640)
    int32_t input_H, input_W;
    if (input_properties.validShape.numDimensions == 4)
    {
        input_H = input_properties.validShape.dimensionSize[2];
        input_W = input_properties.validShape.dimensionSize[3];
        std::cout << "input tensor valid shape: (" << input_properties.validShape.dimensionSize[0];
        std::cout << ", " << input_properties.validShape.dimensionSize[1];
        std::cout << ", " << input_H;
        std::cout << ", " << input_W << ")" << std::endl;
    }
    else
    {
        std::cout << "input tensor validShape.numDimensions is not 4 such as (1,3,640,640), please check!" << std::endl;
        return -1;
    }

    // 2.4 模型输出检查
    // 2.4 Model output check
    int32_t output_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetOutputCount(&output_count, dnn_handle),
        "hbDNNGetOutputCount failed");

    // 2.4.1 D-Robotics YOLOV8-Seg *.bin 模型应该有10个输出
    // 2.4.1 D-Robotics YOLOV8-Seg *.bin model should have 10 outputs
    if (output_count == 10)
    {
        for (int i = 0; i < 10; i++)
        {
            hbDNNTensorProperties output_properties;
            RDK_CHECK_SUCCESS(
                hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i),
                "hbDNNGetOutputTensorProperties failed");
            std::cout << "output[" << i << "] ";
            std::cout << "valid shape: (" << output_properties.validShape.dimensionSize[0];
            std::cout << ", " << output_properties.validShape.dimensionSize[1];
            std::cout << ", " << output_properties.validShape.dimensionSize[2];
            std::cout << ", " << output_properties.validShape.dimensionSize[3] << "), ";
            if (output_properties.quantiType == SHIFT)
                std::cout << "QuantiType: SHIFT" << std::endl;
            if (output_properties.quantiType == SCALE)
                std::cout << "QuantiType: SCALE" << std::endl;
            if (output_properties.quantiType == NONE)
                std::cout << "QuantiType: NONE" << std::endl;
        }
    }
    else
    {
        std::cout << "Your Model's outputs num is not 10, please check!" << std::endl;
        return -1;
    }

    // 2.4.2 调整输出头顺序的映射
    // 2.4.2 Adjust the mapping of output order
    int order[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int32_t H_4 = input_H / 4;
    int32_t H_8 = input_H / 8;
    int32_t H_16 = input_H / 16;
    int32_t H_32 = input_H / 32;
    int32_t W_4 = input_W / 4;
    int32_t W_8 = input_W / 8;
    int32_t W_16 = input_W / 16;
    int32_t W_32 = input_W / 32;
    int32_t order_we_want[10][3] = {
        {H_8, W_8, CLASSES_NUM},   // output[order[0]]: (1, H // 8,  W // 8,  CLASSES_NUM)
        {H_8, W_8, 4 * REG},       // output[order[1]]: (1, H // 8,  W // 8,  64)
        {H_8, W_8, MCES},          // output[order[2]]: (1, H // 8,  W // 8,  MCES)
        {H_16, W_16, CLASSES_NUM}, // output[order[3]]: (1, H // 16, W // 16, CLASSES_NUM)
        {H_16, W_16, 4 * REG},     // output[order[4]]: (1, H // 16, W // 16, 64)
        {H_16, W_16, MCES},        // output[order[5]]: (1, H // 16, W // 16, MCES)
        {H_32, W_32, CLASSES_NUM}, // output[order[6]]: (1, H // 32, W // 32, CLASSES_NUM)
        {H_32, W_32, 4 * REG},     // output[order[7]]: (1, H // 32, W // 32, 64)
        {H_32, W_32, MCES},        // output[order[8]]: (1, H // 32, W // 16, MCES)
        {H_4, W_4, MCES}           // output[order[9]]: (1, H // 4, W // 4, MCES)
    };
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            hbDNNTensorProperties output_properties;
            RDK_CHECK_SUCCESS(
                hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, j),
                "hbDNNGetOutputTensorProperties failed");
            int32_t h = output_properties.validShape.dimensionSize[1];
            int32_t w = output_properties.validShape.dimensionSize[2];
            int32_t c = output_properties.validShape.dimensionSize[3];
            if (h == order_we_want[i][0] && w == order_we_want[i][1] && c == order_we_want[i][2])
            {
                order[i] = j;
                break;
            }
        }
    }

    // 2.4.3 打印并检查调整后的输出头顺序的映射
    // 2.4.3 Print and check the mapping of output order
    if (order[0] + order[1] + order[2] + order[3] + order[4] + order[5] + order[6] + order[7] + order[8] + order[9] == 0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9)
    {
        std::cout << "Outputs order check SUCCESS, continue." << std::endl;
        std::cout << "order = {";
        for (int i = 0; i < 10; i++)
        {
            std::cout << order[i] << ", ";
        }
        std::cout << "}" << std::endl;
    }
    else
    {
        std::cout << "Outputs order check FAILED, use default" << std::endl;
        for (int i = 0; i < 10; i++)
            order[i] = i;
    }

    // 3. 利用OpenCV准备nv12的输入数据
    // 3. Prepare nv12 input data using OpenCV

    // 注：实际量产中, 往往使用Codec, VPU, JPU等硬件来准备nv12输入数据
    // Note: In actual mass production, hardware such as Codec, VPU, JPU, etc., is often used to prepare nv12 input data.
    // H264/H265 -> Decoder -> nv12 -> VPS/VSE -> BPU
    // V4L2 -> Decoder -> nv12 -> VPS/VSE -> BPU
    // MIPI -> nv12 -> VPS/VPS -> BPU

    // 3.1 利用OpenCV读取图像
    // 3.1 Read image using OpenCV
    cv::Mat img = cv::imread(TESR_IMG_PATH);
    std::cout << "img path: " << TESR_IMG_PATH << std::endl;
    std::cout << "img (cols, rows, channels): (";
    std::cout << img.rows << ", ";
    std::cout << img.cols << ", ";
    std::cout << img.channels() << ")" << std::endl;

    // 3.2 前处理
    // 3.2 Preprocess
    float y_scale = 1.0;
    float x_scale = 1.0;
    int x_shift = 0;
    int y_shift = 0;
    cv::Mat resize_img;
    if (PREPROCESS_TYPE == LETTERBOX_TYPE) // letter box
    {
        begin_time = std::chrono::system_clock::now();
        x_scale = std::min(1.0 * input_H / img.rows, 1.0 * input_W / img.cols);
        y_scale = x_scale;
        if (x_scale <= 0 || y_scale <= 0)
        {
            throw std::runtime_error("Invalid scale factor.");
        }

        int new_w = img.cols * x_scale;
        x_shift = (input_W - new_w) / 2;
        int x_other = input_W - new_w - x_shift;

        int new_h = img.rows * y_scale;
        y_shift = (input_H - new_h) / 2;
        int y_other = input_H - new_h - y_shift;

        cv::Size targetSize(new_w, new_h);
        cv::resize(img, resize_img, targetSize);
        cv::copyMakeBorder(resize_img, resize_img, y_shift, y_other, x_shift, x_other, cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));

        std::cout << "\033[31m pre process (LetterBox) time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 << " ms\033[0m" << std::endl;
    }
    else if (PREPROCESS_TYPE == RESIZE_TYPE) // resize
    {
        begin_time = std::chrono::system_clock::now();

        cv::Size targetSize(input_W, input_H);
        cv::resize(img, resize_img, targetSize);

        y_scale = 1.0 * input_H / img.rows;
        x_scale = 1.0 * input_W / img.cols;
        y_shift = 0;
        x_shift = 0;

        std::cout << "\033[31m pre process (Resize) time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 << " ms\033[0m" << std::endl;
    }
    std::cout << "y_scale = " << y_scale << ", ";
    std::cout << "x_scale = " << x_scale << std::endl;
    std::cout << "y_shift = " << y_shift << ", ";
    std::cout << "x_shift = " << x_shift << std::endl;

    // 3.3 cv::Mat的BGR888格式转为YUV420SP格式
    // 3.3 Convert BGR888 to YUV420SP
    begin_time = std::chrono::system_clock::now();
    cv::Mat img_nv12;
    cv::Mat yuv_mat;
    cv::cvtColor(resize_img, yuv_mat, cv::COLOR_BGR2YUV_I420);
    uint8_t *yuv = yuv_mat.ptr<uint8_t>();
    img_nv12 = cv::Mat(input_H * 3 / 2, input_W, CV_8UC1);
    uint8_t *ynv12 = img_nv12.ptr<uint8_t>();
    int uv_height = input_H / 2;
    int uv_width = input_W / 2;
    int y_size = input_H * input_W;
    memcpy(ynv12, yuv, y_size);
    uint8_t *nv12 = ynv12 + y_size;
    uint8_t *u_data = yuv + y_size;
    uint8_t *v_data = u_data + uv_height * uv_width;
    for (int i = 0; i < uv_width * uv_height; i++)
    {
        *nv12++ = *u_data++;
        *nv12++ = *v_data++;
    }
    std::cout << "\033[31m bgr8 to nv12 time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 << " ms\033[0m" << std::endl;

    begin_time = std::chrono::system_clock::now();

    // 3.4 将准备好的输入数据放入hbDNNTensor
    // 3.4 Put input data into hbDNNTensor
    hbDNNTensor input;
    input.properties = input_properties;
    hbSysAllocCachedMem(&input.sysMem[0], int(3 * input_H * input_W / 2));

    memcpy(input.sysMem[0].virAddr, ynv12, int(3 * input_H * input_W / 2));
    hbSysFlushMem(&input.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);

    // 4. 准备模型输出数据的空间
    // 4. Prepare the space for model output data
    hbDNNTensor *output = new hbDNNTensor[output_count];
    for (int i = 0; i < 10; i++)
    {
        hbDNNTensorProperties &output_properties = output[i].properties;
        hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i);
        int out_aligned_size = output_properties.alignedByteSize;
        hbSysMem &mem = output[i].sysMem[0];
        hbSysAllocCachedMem(&mem, out_aligned_size);
    }

    // 5. 推理模型
    // 5. Inference
    hbDNNTaskHandle_t task_handle = nullptr;
    hbDNNInferCtrlParam infer_ctrl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
    hbDNNInfer(&task_handle, &output, &input, dnn_handle, &infer_ctrl_param);

    // 6. 等待任务结束
    // 6. Wait for task to finish
    hbDNNWaitTaskDone(task_handle, 0);
    std::cout << "\033[31m forward time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 << " ms\033[0m" << std::endl;

    // 7. YOLOV8-Seg-Detect 后处理
    // 7. Postprocess
    std::cout << "Starting YOLOv8 Segmentation Postprocess..." << std::endl;
    
    // 利用反函数作用阈值，利用单调性筛选
    float CONF_THRES_RAW = -log(1 / SCORE_THRESHOLD - 1);             
    
    // 临时存储解码后的结果 (在 NMS 之前)
    std::vector<cv::Rect2d> decoded_bboxes_all; // 模型输入尺寸的边界框
    std::vector<float> decoded_scores_all;      // 置信度
    std::vector<int> decoded_classes_all;       // 类别ID
    std::vector<std::vector<float>> decoded_mces_all; // 每个框的掩码系数

    begin_time = std::chrono::system_clock::now();

    // 7.0 Mask Protos - 原型掩码 
    // output[order[9]]: (1, H // 4, W // 4, MCES)

    // 7.0.1 检查反量化类型并根据类型进行处理
    hbSysFlushMem(&(output[order[9]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

    // 7.0.2 获取原型掩码数据并根据量化类型进行处理
    std::vector<float> proto_data_dequant(H_4 * W_4 * MCES);
    begin_time = std::chrono::system_clock::now();

    auto *proto_data_float = reinterpret_cast<float *>(output[order[9]].sysMem[0].virAddr);
    // 复制数据
        for (int i = 0; i < H_4 * W_4 * MCES; ++i) {
            proto_data_dequant[i] = proto_data_float[i];
        }
    std::cout << "Using NONE quantization (float) for proto mask" << std::endl;
    
    // 创建存储掩码原型的矩阵
    cv::Mat proto_mat(H_4 * W_4, MCES, CV_32F, proto_data_dequant.data());
    
    std::cout << "\033[31m Proto processing time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 << " ms\033[0m" << std::endl;

    // 7.1 处理小目标特征图
    // output[order[0]]: (1, H // 8,  W // 8,  CLASSES_NUM)
    // output[order[1]]: (1, H // 8,  W // 8,  4 * REG)
    // output[order[2]]: (1, H // 8,  W // 8,  MCES)
    
    // 获取小目标特征图输出索引
    int s_cls_idx = order[0];
    int s_box_idx = order[1];
    int s_mce_idx = order[2];
    float s_stride = 8.0f;

    // 检查反量化类型
    if (output[s_cls_idx].properties.quantiType != NONE) {
        std::cerr << "Warning: Classification output should have SCALE quantization" 
                 << output[s_cls_idx].properties.quantiType << std::endl;
    }
    if (output[s_box_idx].properties.quantiType != SCALE) {
        std::cerr << "Warning: Bounding box output should have SCALE quantization " 
                 << output[s_box_idx].properties.quantiType << std::endl;
    }
    if (output[s_mce_idx].properties.quantiType != SCALE) {
        std::cerr << "Warning: Mask coefficient output should have SCALE quantization" << std::endl;
    }
    // 刷新内存
    hbSysFlushMem(&output[s_cls_idx].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    hbSysFlushMem(&output[s_box_idx].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    hbSysFlushMem(&output[s_mce_idx].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    // 获取数据指针和量化尺度
    auto* s_cls_raw = reinterpret_cast<float*>(output[s_cls_idx].sysMem[0].virAddr);
    auto* s_box_raw = reinterpret_cast<int32_t*>(output[s_box_idx].sysMem[0].virAddr);
    auto* s_box_scale = reinterpret_cast<float*>(output[s_box_idx].properties.scale.scaleData);
    auto* s_mce_raw = reinterpret_cast<int32_t*>(output[s_mce_idx].sysMem[0].virAddr);
    auto* s_mce_scale = reinterpret_cast<float*>(output[s_mce_idx].properties.scale.scaleData);
    
    // 7.1.1 遍历特征图的每个位置
    for (int h = 0; h < H_8; h++) {
        for (int w = 0; w < W_8; w++) {
            // 计算当前位置在内存中的偏移
            int offset = h * W_8 + w;
            
            // 获取当前位置的特征向量
            float* cur_cls_raw = s_cls_raw + offset * CLASSES_NUM;
            int32_t* cur_box_raw = s_box_raw + offset * (4 * REG);
            int32_t* cur_mce_raw = s_mce_raw + offset * MCES;
            
            // 找到分数最大的类别
            int cls_id = 0;
            for (int i = 1; i < CLASSES_NUM; i++) {
                if (cur_cls_raw[i] > cur_cls_raw[cls_id]) {
                    cls_id = i;
                }
            }
            // 如果置信度低于阈值，跳过处理
            if (cur_cls_raw[cls_id] < CONF_THRES_RAW) {
                continue;
            }
            // 计算Sigmoid激活后的置信度
            float score = 1.0f / (1.0f + std::exp(-cur_cls_raw[cls_id]));
            // 使用DFL解码边界框
            float ltrb[4] = {0.0f}; // left, top, right, bottom offsets
            
            for (int i = 0; i < 4; i++) {
                float sum = 0.0f;
                for (int j = 0; j < REG; j++) {
                    int index = REG * i + j;
                    float dfl = std::exp(float(cur_box_raw[index]) * s_box_scale[index]);
                    ltrb[i] += dfl * j;
                    sum += dfl;
                }
                ltrb[i] /= sum;
            }
            // 如果无效框，跳过处理
            if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0) {
                continue;
            }
            
            // 计算输入尺寸下的边界框坐标
            float x1 = (w + 0.5f - ltrb[0]) * s_stride;
            float y1 = (h + 0.5f - ltrb[1]) * s_stride;
            float x2 = (w + 0.5f + ltrb[2]) * s_stride;
            float y2 = (h + 0.5f + ltrb[3]) * s_stride;
            
            // 反量化掩码系数
            std::vector<float> mask_coeffs(MCES);
            for (int i = 0; i < MCES; i++) {
                mask_coeffs[i] = float(cur_mce_raw[i]) * s_mce_scale[i];
            }
            
            // 保存解码结果
            if (x1 >= 0 && y1 >= 0 && x2 > x1 && y2 > y1 && x2 <= input_W && y2 <= input_H) {
                decoded_bboxes_all.push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
                decoded_scores_all.push_back(score);
                decoded_classes_all.push_back(cls_id);
                decoded_mces_all.push_back(mask_coeffs);
            }
        }
    }

    // 7.2 处理中目标特征图
    // output[order[3]]: (1, H // 16, W // 16, CLASSES_NUM)
    // output[order[4]]: (1, H // 16, W // 16, 4 * REG)
    // output[order[5]]: (1, H // 16, W // 16, MCES)
    
    // 获取中目标特征图输出索引
    int m_cls_idx = order[3];
    int m_box_idx = order[4];
    int m_mce_idx = order[5];
    float m_stride = 16.0f;
    // 刷新内存
    hbSysFlushMem(&output[m_cls_idx].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    hbSysFlushMem(&output[m_box_idx].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    hbSysFlushMem(&output[m_mce_idx].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    // 获取数据指针和量化尺度
    auto* m_cls_raw = reinterpret_cast<float*>(output[m_cls_idx].sysMem[0].virAddr);
    auto* m_box_raw = reinterpret_cast<int32_t*>(output[m_box_idx].sysMem[0].virAddr);
    auto* m_box_scale = reinterpret_cast<float*>(output[m_box_idx].properties.scale.scaleData);
    auto* m_mce_raw = reinterpret_cast<int32_t*>(output[m_mce_idx].sysMem[0].virAddr);
    auto* m_mce_scale = reinterpret_cast<float*>(output[m_mce_idx].properties.scale.scaleData);
    
    // 7.2.1 遍历特征图的每个位置
    for (int h = 0; h < H_16; h++) {
        for (int w = 0; w < W_16; w++) {
            // 计算当前位置在内存中的偏移
            int offset = h * W_16 + w;
            
            // 获取当前位置的特征向量
            float* cur_cls_raw = m_cls_raw + offset * CLASSES_NUM;
            int32_t* cur_box_raw = m_box_raw + offset * (4 * REG);
            int32_t* cur_mce_raw = m_mce_raw + offset * MCES;
            
            // 找到分数最大的类别
            int cls_id = 0;
            for (int i = 1; i < CLASSES_NUM; i++) {
                if (cur_cls_raw[i] > cur_cls_raw[cls_id]) {
                    cls_id = i;
                }
            }
            
            // 如果置信度低于阈值，跳过处理
            if (cur_cls_raw[cls_id] < CONF_THRES_RAW) {
                continue;
            }
            
            // 计算Sigmoid激活后的置信度
            float score = 1.0f / (1.0f + std::exp(-cur_cls_raw[cls_id]));
            
            // 使用DFL解码边界框
            float ltrb[4] = {0.0f}; // left, top, right, bottom offsets
            
            for (int i = 0; i < 4; i++) {
                float sum = 0.0f;
                for (int j = 0; j < REG; j++) {
                    int index = REG * i + j;
                    float dfl = std::exp(float(cur_box_raw[index]) * m_box_scale[index]);
                    ltrb[i] += dfl * j;
                    sum += dfl;
                }
                ltrb[i] /= sum;
            }
            
            // 如果无效框，跳过处理
            if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0) {
                continue;
            }
            
            // 计算输入尺寸下的边界框坐标
            float x1 = (w + 0.5f - ltrb[0]) * m_stride;
            float y1 = (h + 0.5f - ltrb[1]) * m_stride;
            float x2 = (w + 0.5f + ltrb[2]) * m_stride;
            float y2 = (h + 0.5f + ltrb[3]) * m_stride;
            
            // 反量化掩码系数
            std::vector<float> mask_coeffs(MCES);
            for (int i = 0; i < MCES; i++) {
                mask_coeffs[i] = float(cur_mce_raw[i]) * m_mce_scale[i];
            }
            
            // 保存解码结果
            if (x1 >= 0 && y1 >= 0 && x2 > x1 && y2 > y1 && x2 <= input_W && y2 <= input_H) {
                decoded_bboxes_all.push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
                decoded_scores_all.push_back(score);
                decoded_classes_all.push_back(cls_id);
                decoded_mces_all.push_back(mask_coeffs);
            } else {
                std::cerr << "Warning: Invalid bbox coordinates: (" << x1 << "," << y1 << "," << x2 << "," << y2 
                         << ") for anchor at (" << w << "," << h << ") with stride " << m_stride << std::endl;
            }
        }
    }

    // 7.3 处理大目标特征图
    // output[order[6]]: (1, H // 32, W // 32, CLASSES_NUM)
    // output[order[7]]: (1, H // 32, W // 32, 4 * REG)
    // output[order[8]]: (1, H // 32, W // 32, MCES)
    
    // 获取大目标特征图输出索引
    int l_cls_idx = order[6];
    int l_box_idx = order[7];
    int l_mce_idx = order[8];
    float l_stride = 32.0f;

    // 刷新内存
    hbSysFlushMem(&output[l_cls_idx].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    hbSysFlushMem(&output[l_box_idx].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    hbSysFlushMem(&output[l_mce_idx].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    
    // 获取数据指针和量化尺度
    auto* l_cls_raw = reinterpret_cast<float*>(output[l_cls_idx].sysMem[0].virAddr);
    auto* l_box_raw = reinterpret_cast<int32_t*>(output[l_box_idx].sysMem[0].virAddr);
    auto* l_box_scale = reinterpret_cast<float*>(output[l_box_idx].properties.scale.scaleData);
    auto* l_mce_raw = reinterpret_cast<int32_t*>(output[l_mce_idx].sysMem[0].virAddr);
    auto* l_mce_scale = reinterpret_cast<float*>(output[l_mce_idx].properties.scale.scaleData);
    
    // 7.3.1 遍历特征图的每个位置
    for (int h = 0; h < H_32; h++) {
        for (int w = 0; w < W_32; w++) {
            // 计算当前位置在内存中的偏移
            int offset = h * W_32 + w;
            
            // 获取当前位置的特征向量
            float* cur_cls_raw = l_cls_raw + offset * CLASSES_NUM;
            int32_t* cur_box_raw = l_box_raw + offset * (4 * REG);
            int32_t* cur_mce_raw = l_mce_raw + offset * MCES;
            
            // 找到分数最大的类别
            int cls_id = 0;
            for (int i = 1; i < CLASSES_NUM; i++) {
                if (cur_cls_raw[i] > cur_cls_raw[cls_id]) {
                    cls_id = i;
                }
            }
            
            // 如果置信度低于阈值，跳过处理
            if (cur_cls_raw[cls_id] < CONF_THRES_RAW) {
                continue;
            }
            
            // 计算Sigmoid激活后的置信度
            float score = 1.0f / (1.0f + std::exp(-cur_cls_raw[cls_id]));
            
            // 使用DFL解码边界框
            float ltrb[4] = {0.0f}; // left, top, right, bottom offsets
            
            for (int i = 0; i < 4; i++) {
                float sum = 0.0f;
                for (int j = 0; j < REG; j++) {
                    int index = REG * i + j;
                    float dfl = std::exp(float(cur_box_raw[index]) * l_box_scale[index]);
                    ltrb[i] += dfl * j;
                    sum += dfl;
                }
                ltrb[i] /= sum;
            }
            
            // 如果无效框，跳过处理
            if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0) {
                continue;
            }
            
            // 计算输入尺寸下的边界框坐标
            float x1 = (w + 0.5f - ltrb[0]) * l_stride;
            float y1 = (h + 0.5f - ltrb[1]) * l_stride;
            float x2 = (w + 0.5f + ltrb[2]) * l_stride;
            float y2 = (h + 0.5f + ltrb[3]) * l_stride;
            
            // 反量化掩码系数
            std::vector<float> mask_coeffs(MCES);
            for (int i = 0; i < MCES; i++) {
                mask_coeffs[i] = float(cur_mce_raw[i]) * l_mce_scale[i];
            }
            
            // 保存解码结果
            if (x1 >= 0 && y1 >= 0 && x2 > x1 && y2 > y1 && x2 <= input_W && y2 <= input_H) {
                decoded_bboxes_all.push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
                decoded_scores_all.push_back(score);
                decoded_classes_all.push_back(cls_id);
                decoded_mces_all.push_back(mask_coeffs);
            } else {
                std::cerr << "Warning: Invalid bbox coordinates: (" << x1 << "," << y1 << "," << x2 << "," << y2 
                         << ") for anchor at (" << w << "," << h << ") with stride " << l_stride << std::endl;
            }
        }
    }

    std::cout << "Feature map processing done. Total detections before NMS: " << decoded_bboxes_all.size() << std::endl;

    cv::Mat img_display = resize_img.clone(); 
    // 创建掩膜图像
    cv::Mat zeros = cv::Mat::zeros(input_H, input_W, CV_8UC3);
    // 创建最终合成图像（原图+掩膜）
    cv::Mat result_overlay;

    // 8. 执行NMS
    std::vector<int> nms_indices_output;
    std::vector<int> original_indices_map;
    std::vector<cv::Rect2d> final_bboxes;
    std::vector<float> final_scores;
    std::vector<int> final_class_ids;
    std::vector<std::vector<float>> final_mces;

    if (!decoded_bboxes_all.empty()) {
        std::vector<cv::Rect> nms_bboxes_cv;
        std::vector<float> nms_scores_filtered;

        for(size_t idx = 0; idx < decoded_bboxes_all.size(); ++idx) {
            const auto& box = decoded_bboxes_all[idx];
            int x = std::max(0.0, box.x);
            int y = std::max(0.0, box.y);
            int width = std::max(1.0, box.width);
            int height = std::max(1.0, box.height);
            // 确保边界框不超出图像
            if (x + width > input_W) width = input_W - x;
            if (y + height > input_H) height = input_H - y;
            if (width <= 0 || height <= 0) continue;

            nms_bboxes_cv.push_back(cv::Rect(x, y, width, height));
            nms_scores_filtered.push_back(decoded_scores_all[idx]);
            original_indices_map.push_back(idx);
        }

        if (!nms_bboxes_cv.empty()){
            cv::dnn::NMSBoxes(nms_bboxes_cv, nms_scores_filtered, SCORE_THRESHOLD, NMS_THRESHOLD, nms_indices_output);
            std::cout << "NMS done. Detections after NMS: " << nms_indices_output.size() << std::endl;
            
            // 收集NMS后的结果
            for (int filtered_idx : nms_indices_output) {
                if (filtered_idx < 0 || filtered_idx >= static_cast<int>(original_indices_map.size())) {
                    std::cerr << "Error: Invalid index from NMSBoxes output" << std::endl;
                    continue;
                }
                
                int original_idx = original_indices_map[filtered_idx];
                final_bboxes.push_back(decoded_bboxes_all[original_idx]);
                final_scores.push_back(decoded_scores_all[original_idx]);
                final_class_ids.push_back(decoded_classes_all[original_idx]);
                final_mces.push_back(decoded_mces_all[original_idx]);
            }
        } else {
            std::cout << "No valid boxes remaining before NMS." << std::endl;
        }
    } else {
        std::cout << "No detections before NMS." << std::endl;
    }

    // 9. 生成掩码并绘制结果
    begin_time = std::chrono::system_clock::now();
    std::vector<cv::Mat> all_masks;

    // 9.1 处理每个检测结果
    for (size_t i = 0; i < final_bboxes.size(); i++) {
        cv::Rect2d bbox = final_bboxes[i];
        int cls_id = final_class_ids[i];
        float score = final_scores[i];
        std::vector<float>& mce = final_mces[i];
        
        // 确保类别ID在有效范围内
        if (cls_id < 0 || cls_id >= static_cast<int>(object_names.size())) {
            std::cerr << "Warning: Invalid class ID: " << cls_id << ", defaulting to 0" << std::endl;
            cls_id = 0;
        }
        
        std::cout << "Object " << i+1 << ": " << object_names[cls_id] << ", score=" 
                 << std::fixed << std::setprecision(4) << score
                 << ", bbox=(" << bbox.x << "," << bbox.y << "," 
                 << bbox.width << "," << bbox.height << ")" << std::endl;
        
        // 9.1.1 绘制边界框
        cv::rectangle(img_display, bbox, 
                     cv::Scalar(rdk_colors[cls_id % rdk_colors.size()][0], 
                               rdk_colors[cls_id % rdk_colors.size()][1], 
                               rdk_colors[cls_id % rdk_colors.size()][2]), 2);
        
        // 9.1.2 绘制标签
        cv::putText(img_display, 
                   object_names[cls_id] + ": " + std::to_string(score).substr(0, 5), 
                   cv::Point(bbox.x, bbox.y - 2), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, 
                   cv::Scalar(rdk_colors[cls_id % rdk_colors.size()][0], 
                             rdk_colors[cls_id % rdk_colors.size()][1], 
                             rdk_colors[cls_id % rdk_colors.size()][2]), 2);

        // 9.1.3 生成实例掩码
        // 9.1.3.1 矩阵乘法计算低分辨率掩码
        cv::Mat mce_mat(1, MCES, CV_32F, mce.data());
        cv::Mat instance_mask_flat = proto_mat * mce_mat.t();
        cv::Mat instance_mask_low_res = instance_mask_flat.reshape(1, H_4);
        
        // 9.1.3.2 应用sigmoid激活函数
        cv::Mat sigmoid_mask;
        cv::exp(-instance_mask_low_res, sigmoid_mask);
        sigmoid_mask = 1.0 / (1.0 + sigmoid_mask);
        
        // 9.1.3.3 上采样到输入图像尺寸
        cv::Mat resized_sigmoid_mask;
        cv::resize(sigmoid_mask, resized_sigmoid_mask, cv::Size(input_W, input_H), 0, 0, cv::INTER_LINEAR);
        
        // 9.1.3.4 二值化掩码
        cv::Mat binary_mask;
        cv::threshold(resized_sigmoid_mask, binary_mask, 0.5, 1.0, cv::THRESH_BINARY);
        
        // 9.1.3.5 确保掩码为8位格式，以便与bitwise_and操作兼容
        cv::Mat binary_mask_8u;
        binary_mask.convertTo(binary_mask_8u, CV_8U, 255);
        
        // 9.1.3.6 裁剪掩码到检测框区域
        float x1 = std::max(0.0, bbox.x);
        float y1 = std::max(0.0, bbox.y);
        float x2 = std::min(static_cast<double>(input_W), bbox.x + bbox.width);
        float y2 = std::min(static_cast<double>(input_H), bbox.y + bbox.height);
        
        int mask_h = static_cast<int>(y2 - y1);
        int mask_w = static_cast<int>(x2 - x1);
        
        if (mask_h <= 0 || mask_w <= 0)
            continue;
            
        // 9.1.3.7 确保ROI不超出图像边界
        if (x1 + mask_w > input_W || y1 + mask_h > input_H)
            continue;
            
        cv::Rect roi(static_cast<int>(x1), static_cast<int>(y1), mask_w, mask_h);
        cv::Mat roi_mask = binary_mask_8u(roi);
        
        // 创建彩色掩码
        cv::Mat color_mask = cv::Mat::zeros(roi.height, roi.width, CV_8UC3);
        color_mask.setTo(cv::Scalar(rdk_colors[cls_id % rdk_colors.size()][0], 
                                   rdk_colors[cls_id % rdk_colors.size()][1], 
                                   rdk_colors[cls_id % rdk_colors.size()][2]));
        
        // 9.1.3.8 应用掩码到颜色
        cv::Mat color_instance_mask;
        cv::bitwise_and(color_mask, color_mask, color_instance_mask, roi_mask);
        
        // 9.1.3.9 将掩码添加到零图像
        cv::Mat zeros_roi = zeros(roi);
        cv::addWeighted(zeros_roi, 1.0, color_instance_mask, 0.5, 0, zeros_roi);
    }
    
    // 9.2 合成最终结果图
    cv::Mat final_result;
    cv::addWeighted(img_display, 0.6, zeros, 0.4, 0, final_result);

    // 9.2.1 创建三图并排的结果图：检测图、掩膜图、最终结果图
    cv::Mat concatenated_result;
    cv::hconcat(img_display, zeros, concatenated_result);  // 先拼接检测图和掩膜图
    cv::hconcat(concatenated_result, final_result, concatenated_result);  // 再拼接最终结果图
    
    std::cout << "\033[31m Mask Result time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 << " ms\033[0m" << std::endl;

    // 9.2.2 保存结果
    cv::imwrite(IMG_SAVE_PATH, concatenated_result);

    // 10. 释放任务
    // 10. Release task
    hbDNNReleaseTask(task_handle);

    // 11. 释放内存
    // 11. Release memory
    hbSysFreeMem(&(input.sysMem[0]));
    for (int i = 0; i < 10; i++)
        hbSysFreeMem(&(output[i].sysMem[0]));

    // 12. 释放模型
    // 12. Release model
    hbDNNRelease(packed_dnn_handle);

    return 0;
}