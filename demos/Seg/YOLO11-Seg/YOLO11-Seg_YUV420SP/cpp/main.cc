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
#define MODEL_PATH "../../ptq_models/yolo11n_seg_bayese_640x640_nv12_modified.bin"

// 推理使用的测试图片路径
// Path of the test image used for inference.
// #define TESR_IMG_PATH "../../../../../../resource/datasets/COCO2017/assets/bus.jpg"
#define TESR_IMG_PATH "../../../../../../resource/datasets/COCO2017/assets/zidane.jpg"

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
#define CLASSES_NUM 80

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
    std::cout << "这个代码还有一些问题，分割的mask无法正确处理，如果您有idea，欢迎PR！" << std::endl;
    std::cout << "There are still some problems with this code, the split mask cannot be handled correctly, if you have an idea, welcome PR!" << std::endl;
    return -1;
    std::ofstream debug;
    debug.open("proto.txt");

    std::ofstream l_mces;
    l_mces.open("l_mces.txt");
    // 0. 加载bin模型
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

    // 2.3.1 D-Robotics YOLO11-Seg *.bin 模型应该为单输入
    // 2.3.1 D-Robotics YOLO11-Seg *.bin model should have only one input
    if (input_count > 1)
    {
        std::cout << "Your Model have more than 1 input, please check!" << std::endl;
        return -1;
    }

    // 2.3.2 D-Robotics YOLO11-Seg *.bin 模型输入Tensor类型应为nv12
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

    // 2.3.3 D-Robotics YOLO11-Seg *.bin 模型输入Tensor数据排布应为NCHW
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

    // 2.3.4 D-Robotics YOLO11-Seg *.bin 模型输入Tensor数据的valid shape应为(1,3,H,W)
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

    // 2.4.1 D-Robotics YOLO11-Seg *.bin 模型应该有10个输出
    // 2.4.1 D-Robotics YOLO11-Seg *.bin model should have 10 outputs
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

    // 7. YOLO11-Seg-Detect 后处理
    // 7. Postprocess
    float CONF_THRES_RAW = -log(1 / SCORE_THRESHOLD - 1);             // 利用反函数作用阈值，利用单调性筛选
    std::vector<std::vector<cv::Rect2d>> bboxes(CLASSES_NUM);         // 每个id的xyhw 信息使用一个std::vector<cv::Rect2d>存储
    std::vector<std::vector<float>> scores(CLASSES_NUM);              // 每个id的score信息使用一个std::vector<float>存储
    std::vector<std::vector<std::vector<float>>> maskes(CLASSES_NUM); // 每个id的mask信息使用一个std::vector<std::vector>存储

    begin_time = std::chrono::system_clock::now();

    // 7.0 Mask Protos
    // output[order[9]]: (1, H // 4, W // 4, MCES)

    // 7.0.1 检查反量化类型是否符合RDK Model Zoo的README导出的bin模型规范
    // 7.0.1 Check if the dequantization type complies with the bin model specification exported in the RDK Model Zoo README.
    if (output[order[9]].properties.quantiType != SCALE)
    {
        std::cout << "output[order[9]] QuantiType is not SCALE, please check!" << std::endl;
        return -1;
    }

    // 7.0.2 对缓存的BPU内存进行刷新
    // 7.0.2 Flush the cached BPU memory
    hbSysFlushMem(&(output[order[9]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

    // 7.0.3 将BPU推理完的内存地址转换为对应类型的指针
    // 7.0.3 Convert the memory address of BPU inference to a pointer of the corresponding type
    auto *proto_data = reinterpret_cast<int16_t *>(output[order[9]].sysMem[0].virAddr);
    float proto_scale_data = output[order[9]].properties.scale.scaleData[0];

    // 7.0.4 反量化
    // 7.0.4 Dequantization
    std::vector<float> proto(H_4 * W_4 * MCES);
    for (int w = 0; w < W_4; w++)
    {
        for (int h = 0; h < H_4; h++)
        {
            for (int c = 0; c < MCES; c++)
            {
                // int index = (h * W_4 * MCES) + (w * MCES) + c;
                int index = (w * H_4 * MCES) + (h  * MCES) + c;
                proto[index] = static_cast<float>(proto_data[index]) * proto_scale_data;
            }
        }
    }
    // debug
    for(int i=0; i < H_4 * W_4 * MCES; i++){
        debug << proto[i] << "   ";
    }

    // debug
    // for (int c = 0; c < MCES; c++)
    // {
    //     for (int h = 0; h < H_4; h++)
    //     {
    //         for (int w = 0; w < W_4; w++)
    //         {
    //             int index = (h  * MCES) + (w * H_4 * MCES) + c;
    //             debug << proto[index] << "   ";
    //         }
    //         debug << std::endl;
    //     }
    //     debug << "\n\n\n" <<std::endl;
    // }

    // 7.1 小目标特征图
    // 7.1 Small Object Feature Map
    // output[order[0]]: (1, H // 8,  W // 8,  CLASSES_NUM)
    // output[order[1]]: (1, H // 8,  W // 8,  4 * REG)
    // output[order[2]]: (1, H // 8,  W // 8,  MCES)

    // 7.1.1 检查反量化类型是否符合RDK Model Zoo的README导出的bin模型规范
    // 7.1.1 Check if the dequantization type complies with the bin model specification exported in the RDK Model Zoo README.
    if (output[order[0]].properties.quantiType != NONE)
    {
        std::cout << "output[order[0]] QuantiType is not NONE, please check!" << std::endl;
        return -1;
    }
    if (output[order[1]].properties.quantiType != SCALE)
    {
        std::cout << "output[order[1]] QuantiType is not SCALE, please check!" << std::endl;
        return -1;
    }
    if (output[order[2]].properties.quantiType != SCALE)
    {
        std::cout << "output[order[2]] QuantiType is not SCALE, please check!" << std::endl;
        return -1;
    }

    // 7.1.2 对缓存的BPU内存进行刷新
    // 7.1.2 Flush the cached BPU memory
    hbSysFlushMem(&(output[order[0]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    hbSysFlushMem(&(output[order[1]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    hbSysFlushMem(&(output[order[2]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

    // 7.1.3 将BPU推理完的内存地址转换为对应类型的指针
    // 7.1.3 Convert the memory address of BPU inference to a pointer of the corresponding type
    auto *s_cls_raw = reinterpret_cast<float *>(output[order[0]].sysMem[0].virAddr);
    auto *s_bbox_raw = reinterpret_cast<int32_t *>(output[order[1]].sysMem[0].virAddr);
    auto *s_bbox_scale = reinterpret_cast<float *>(output[order[1]].properties.scale.scaleData);
    auto *s_mces_raw = reinterpret_cast<int32_t *>(output[order[2]].sysMem[0].virAddr);
    auto *s_mces_scale = reinterpret_cast<float *>(output[order[2]].properties.scale.scaleData);
    for (int h = 0; h < H_8; h++)
    {
        for (int w = 0; w < W_8; w++)
        {
            // 7.1.4 取对应H和W位置的C通道, 记为数组的形式
            // cls对应CLASSES_NUM个分数RAW值, 也就是Sigmoid计算之前的值，这里利用函数单调性先筛选, 再计算
            // bbox对应4个坐标乘以REG的RAW值, 也就是DFL计算之前的值, 仅仅分数合格了, 才会进行这部分的计算
            // 7.1.4 Get the C channel at the corresponding H and W positions, represented as an array.
            // cls corresponds to CLASSES_NUM raw score values, which are the values before Sigmoid calculation. Here, we use the monotonicity of the function to filter first, then calculate.
            // bbox corresponds to the raw values of 4 coordinates multiplied by REG, which are the values before DFL calculation. This part of the calculation is only performed if the score is qualified.
            float *cur_s_cls_raw = s_cls_raw;
            int32_t *cur_s_bbox_raw = s_bbox_raw;
            int32_t *cur_s_mces_raw = s_mces_raw;

            // 7.1.5 找到分数的最大值索引, 如果最大值小于阈值，则舍去
            // 7.1.5 Find the index of the maximum score value and discard if the maximum value is less than the threshold
            int cls_id = 0;
            for (int i = 1; i < CLASSES_NUM; i++)
            {
                if (cur_s_cls_raw[i] > cur_s_cls_raw[cls_id])
                {
                    cls_id = i;
                }
            }

            // 7.1.6 不合格则直接跳过, 避免无用的反量化, DFL和dist2bbox计算
            // 7.1.6 If not qualified, skip to avoid unnecessary dequantization, DFL and dist2bbox calculation
            if (cur_s_cls_raw[cls_id] < CONF_THRES_RAW)
            {
                s_cls_raw += CLASSES_NUM;
                s_bbox_raw += REG * 4;
                continue;
            }

            // 7.1.7 计算这个目标的分数
            // 7.1.7 Calculate the score of the target
            float score = 1 / (1 + std::exp(-cur_s_cls_raw[cls_id]));

            // 7.1.8 对bbox_raw信息进行反量化, DFL计算
            // 7.1.8 Dequantize bbox_raw information, DFL calculation
            float ltrb[4], sum, dfl;
            for (int i = 0; i < 4; i++)
            {
                ltrb[i] = 0.;
                sum = 0.;
                for (int j = 0; j < REG; j++)
                {
                    int index_id = REG * i + j;
                    dfl = std::exp(float(cur_s_bbox_raw[index_id]) * s_bbox_scale[index_id]);
                    ltrb[i] += dfl * j;
                    sum += dfl;
                }
                ltrb[i] /= sum;
            }

            // 7.1.9 剔除不合格的框   if(x1 >= x2 || y1 >=y2) continue;
            // 7.1.9 Remove unqualified boxes
            if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0)
            {
                s_cls_raw += CLASSES_NUM;
                s_bbox_raw += REG * 4;
                continue;
            }

            // 7.1.10 dist 2 bbox (ltrb 2 xyxy)
            float x1 = (w + 0.5 - ltrb[0]) * 8.0;
            float y1 = (h + 0.5 - ltrb[1]) * 8.0;
            float x2 = (w + 0.5 + ltrb[2]) * 8.0;
            float y2 = (h + 0.5 + ltrb[3]) * 8.0;

            // 7.1.11 对应类别加入到对应的std::vector中
            // 7.1.11 Add the corresponding class to the corresponding std::vector.
            bboxes[cls_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
            scores[cls_id].push_back(score);

            // Mask的处理
            // int x1_corp = int(x1 * X_SCALE_CORP);
            // int x2_corp = int(x2 * X_SCALE_CORP);
            // int y1_corp = int(y1 * Y_SCALE_CORP);
            // int y2_corp = int(y2 * Y_SCALE_CORP);
            // cv::Mat mask(y2 - y1, x2 - x1, CV_8UC1, cv::Scalar(0));
            // for (int y = y1_corp; y < y2_corp; y++)
            // {
            //     for (int x = x1_corp; x < x2_corp; x++)
            //     {
            //         float sum = 0.0;
            //         for (int i = 0; i < MCES; i++)
            //         {
            //             int index = y * W_4 * MCES + x * MCES + i;
            //             sum += proto[index] * float(cur_s_mces_raw[i]) * s_mces_scale[i];
            //             if (sum > 0.5)
            //             {
            //                 mask.at<uint8_t>(y - y1_corp, x - x1_corp) = 1;
            //                 break;
            //             }
            //         }
            //         // if (sum > 0.5)
            //         // {
            //         //     mask.at<uint8_t>(y - y1_corp, x - x1_corp) = 1;
            //         // }
            //     }
            // }
            std::vector<float> mc(MCES);
            for (int i = 0; i < MCES; i++)
            {
                mc[i] = float(cur_s_mces_raw[i]) * s_mces_scale[i];
            }
            maskes[cls_id].push_back(mc);

            s_cls_raw += CLASSES_NUM;
            s_bbox_raw += REG * 4;
            s_mces_raw += MCES;
        }
    }

    // 7.2 中目标特征图
    // 7.2 Media Object Feature Map
    // output[order[3]]: (1, H // 16,  W // 16,  CLASSES_NUM)
    // output[order[4]]: (1, H // 16,  W // 16,  4 * REG)
    // output[order[5]]: (1, H // 16, W // 16, MCES)

    // 7.2.1 检查反量化类型是否符合RDK Model Zoo的README导出的bin模型规范
    // 7.2.1 Check if the dequantization type complies with the bin model specification exported in the RDK Model Zoo README.
    if (output[order[3]].properties.quantiType != NONE)
    {
        std::cout << "output[order[3]] QuantiType is not NONE, please check!" << std::endl;
        return -1;
    }
    if (output[order[4]].properties.quantiType != SCALE)
    {
        std::cout << "output[order[4]] QuantiType is not SCALE, please check!" << std::endl;
        return -1;
    }
    if (output[order[5]].properties.quantiType != SCALE)
    {
        std::cout << "output[order[5]] QuantiType is not SCALE, please check!" << std::endl;
        return -1;
    }

    // 7.2.2 对缓存的BPU内存进行刷新
    // 7.2.2 Flush the cached BPU memory
    hbSysFlushMem(&(output[order[3]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    hbSysFlushMem(&(output[order[4]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    hbSysFlushMem(&(output[order[5]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

    // 7.2.3 将BPU推理完的内存地址转换为对应类型的指针
    // 7.2.3 Convert the memory address of BPU inference to a pointer of the corresponding type
    auto *m_cls_raw = reinterpret_cast<float *>(output[order[3]].sysMem[0].virAddr);
    auto *m_bbox_raw = reinterpret_cast<int32_t *>(output[order[4]].sysMem[0].virAddr);
    auto *m_bbox_scale = reinterpret_cast<float *>(output[order[4]].properties.scale.scaleData);
    auto *m_mces_raw = reinterpret_cast<int32_t *>(output[order[5]].sysMem[0].virAddr);
    auto *m_mces_scale = reinterpret_cast<float *>(output[order[5]].properties.scale.scaleData);

    for (int h = 0; h < H_16; h++)
    {
        for (int w = 0; w < W_16; w++)
        {
            // 7.2.4 取对应H和W位置的C通道, 记为数组的形式
            // cls对应CLASSES_NUM个分数RAW值, 也就是Sigmoid计算之前的值，这里利用函数单调性先筛选, 再计算
            // bbox对应4个坐标乘以REG的RAW值, 也就是DFL计算之前的值, 仅仅分数合格了, 才会进行这部分的计算
            // 7.2.4 Get the C channel at the corresponding H and W positions, represented as an array.
            // cls corresponds to CLASSES_NUM raw score values, which are the values before Sigmoid calculation. Here, we use the monotonicity of the function to filter first, then calculate.
            // bbox corresponds to the raw values of 4 coordinates multiplied by REG, which are the values before DFL calculation. This part of the calculation is only performed if the score is qualified.
            float *cur_m_cls_raw = m_cls_raw;
            int32_t *cur_m_bbox_raw = m_bbox_raw;
            int32_t *cur_m_mces_raw = m_mces_raw;

            // 7.2.5 找到分数的最大值索引, 如果最大值小于阈值，则舍去
            // 7.2.5 Find the index of the maximum score value and discard if the maximum value is less than the threshold
            int cls_id = 0;
            for (int i = 1; i < CLASSES_NUM; i++)
            {
                if (cur_m_cls_raw[i] > cur_m_cls_raw[cls_id])
                {
                    cls_id = i;
                }
            }

            // 7.2.6 不合格则直接跳过, 避免无用的反量化, DFL和dist2bbox计算
            // 7.2.6 If not qualified, skip to avoid unnecessary dequantization, DFL and dist2bbox calculation
            if (cur_m_cls_raw[cls_id] < CONF_THRES_RAW)
            {
                m_cls_raw += CLASSES_NUM;
                m_bbox_raw += REG * 4;
                continue;
            }

            // 7.2.7 计算这个目标的分数
            // 7.2.7 Calculate the score of the target
            float score = 1 / (1 + std::exp(-cur_m_cls_raw[cls_id]));

            // 7.2.8 对bbox_raw信息进行反量化, DFL计算
            // 7.2.8 Dequantize bbox_raw information, DFL calculation
            float ltrb[4], sum, dfl;
            for (int i = 0; i < 4; i++)
            {
                ltrb[i] = 0.;
                sum = 0.;
                for (int j = 0; j < REG; j++)
                {
                    int index_id = REG * i + j;
                    dfl = std::exp(float(cur_m_bbox_raw[index_id]) * m_bbox_scale[index_id]);
                    ltrb[i] += dfl * j;
                    sum += dfl;
                }
                ltrb[i] /= sum;
            }

            // 7.2.9 剔除不合格的框   if(x1 >= x2 || y1 >=y2) continue;
            // 7.2.9 Remove unqualified boxes
            if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0)
            {
                m_cls_raw += CLASSES_NUM;
                m_bbox_raw += REG * 4;
                continue;
            }

            // 7.2.10 dist 2 bbox (ltrb 2 xyxy)
            float x1 = (w + 0.5 - ltrb[0]) * 16.0;
            float y1 = (h + 0.5 - ltrb[1]) * 16.0;
            float x2 = (w + 0.5 + ltrb[2]) * 16.0;
            float y2 = (h + 0.5 + ltrb[3]) * 16.0;

            // 7.2.11 对应类别加入到对应的std::vector中
            // 7.2.11 Add the corresponding class to the corresponding std::vector.
            bboxes[cls_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
            scores[cls_id].push_back(score);

            // // Mask的处理
            // int x1_corp = int(x1 * X_SCALE_CORP);
            // int x2_corp = int(x2 * X_SCALE_CORP);
            // int y1_corp = int(y1 * Y_SCALE_CORP);
            // int y2_corp = int(y2 * Y_SCALE_CORP);
            // cv::Mat mask(y2 - y1, x2 - x1, CV_8UC1, cv::Scalar(0));
            // for (int y = y1_corp; y < y2_corp; y++)
            // {
            //     for (int x = x1_corp; x < x2_corp; x++)
            //     {
            //         float sum = 0.0;
            //         for (int i = 0; i < MCES; i++)
            //         {
            //             int index = y * W_4 * MCES + x * MCES + i;
            //             sum += proto[index] * float(cur_m_mces_raw[i]) * m_mces_scale[i];
            //             if (sum > 0.5)
            //             {
            //                 mask.at<uint8_t>(y - y1_corp, x - x1_corp) = 1;
            //                 break;
            //             }
            //         }
            //         // if (sum > 0.5)
            //         // {
            //         //     mask.at<uint8_t>(y - y1_corp, x - x1_corp) = 1;
            //         // }
            //     }
            // }

            // maskes[cls_id].push_back(mask);

            std::vector<float> mc(MCES);
            for (int i = 0; i < MCES; i++)
            {
                mc[i] = float(cur_m_mces_raw[i]) * m_mces_scale[i];
            }
            maskes[cls_id].push_back(mc);

            m_cls_raw += CLASSES_NUM;
            m_bbox_raw += REG * 4;
            m_mces_raw += MCES;
        }
    }

    // 7.3 大目标特征图
    // 7.3 Big Object Feature Map
    // output[order[6]]: (1, H // 32,  W // 32,  CLASSES_NUM)
    // output[order[7]]: (1, H // 32,  W // 32,  4 * REG)
    // output[order[8]]: (1, H // 32, W // 16, MCES)

    // 7.3.1 检查反量化类型是否符合RDK Model Zoo的README导出的bin模型规范
    // 7.3.1 Check if the dequantization type complies with the bin model specification exported in the RDK Model Zoo README.
    if (output[order[6]].properties.quantiType != NONE)
    {
        std::cout << "output[order[6]] QuantiType is not NONE, please check!" << std::endl;
        return -1;
    }
    if (output[order[7]].properties.quantiType != SCALE)
    {
        std::cout << "output[order[7]] QuantiType is not SCALE, please check!" << std::endl;
        return -1;
    }
    if (output[order[8]].properties.quantiType != SCALE)
    {
        std::cout << "output[order[8]] QuantiType is not SCALE, please check!" << std::endl;
        return -1;
    }

    // 7.3.2 对缓存的BPU内存进行刷新
    // 7.3.2 Flush the cached BPU memory
    hbSysFlushMem(&(output[order[6]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    hbSysFlushMem(&(output[order[7]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    hbSysFlushMem(&(output[order[8]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

    // 7.3.3 将BPU推理完的内存地址转换为对应类型的指针
    // 7.3.3 Convert the memory address of BPU inference to a pointer of the corresponding type
    auto *l_cls_raw = reinterpret_cast<float *>(output[order[6]].sysMem[0].virAddr);
    auto *l_bbox_raw = reinterpret_cast<int32_t *>(output[order[7]].sysMem[0].virAddr);
    auto *l_bbox_scale = reinterpret_cast<float *>(output[order[7]].properties.scale.scaleData);
    auto *l_mces_raw = reinterpret_cast<int32_t *>(output[order[8]].sysMem[0].virAddr);
    auto *l_mces_scale = reinterpret_cast<float *>(output[order[8]].properties.scale.scaleData);

    // debug
    for (int i=0; i<H_32*W_32*MCES; i++){
        l_mces << l_mces_raw[i] << ", ";
    }
    l_mces << std::endl;
    std::cout << "\nl_mces: ";
    for(int i=0; i<MCES;i++){
        std::cout << std::fixed << std::setprecision(10)<< l_mces_scale[i] << ", ";
    }
    std::cout << std::endl << std::endl;

    for (int h = 0; h < H_32; h++)
    {
        for (int w = 0; w < W_32; w++)
        {
            // 7.3.4 取对应H和W位置的C通道, 记为数组的形式
            // cls对应CLASSES_NUM个分数RAW值, 也就是Sigmoid计算之前的值，这里利用函数单调性先筛选, 再计算
            // bbox对应4个坐标乘以REG的RAW值, 也就是DFL计算之前的值, 仅仅分数合格了, 才会进行这部分的计算
            // 7.3.4 Get the C channel at the corresponding H and W positions, represented as an array.
            // cls corresponds to CLASSES_NUM raw score values, which are the values before Sigmoid calculation. Here, we use the monotonicity of the function to filter first, then calculate.
            // bbox corresponds to the raw values of 4 coordinates multiplied by REG, which are the values before DFL calculation. This part of the calculation is only performed if the score is qualified.
            float *cur_l_cls_raw = l_cls_raw;
            int32_t *cur_l_bbox_raw = l_bbox_raw;
            int32_t *cur_l_mces_raw = l_mces_raw;

            // 7.3.5 找到分数的最大值索引, 如果最大值小于阈值，则舍去
            // 7.3.5 Find the index of the maximum score value and discard if the maximum value is less than the threshold
            int cls_id = 0;
            for (int i = 1; i < CLASSES_NUM; i++)
            {
                if (cur_l_cls_raw[i] > cur_l_cls_raw[cls_id])
                {
                    cls_id = i;
                }
            }

            // 7.3.6 不合格则直接跳过, 避免无用的反量化, DFL和dist2bbox计算
            // 7.3.6 If not qualified, skip to avoid unnecessary dequantization, DFL and dist2bbox calculation
            if (cur_l_cls_raw[cls_id] < CONF_THRES_RAW)
            {
                l_cls_raw += CLASSES_NUM;
                l_bbox_raw += REG * 4;
                continue;
            }

            // 7.3.7 计算这个目标的分数
            // 7.3.7 Calculate the score of the target
            float score = 1 / (1 + std::exp(-cur_l_cls_raw[cls_id]));

            // 7.3.8 对bbox_raw信息进行反量化, DFL计算
            // 7.3.8 Dequantize bbox_raw information, DFL calculation
            float ltrb[4], sum, dfl;
            for (int i = 0; i < 4; i++)
            {
                ltrb[i] = 0.;
                sum = 0.;
                for (int j = 0; j < REG; j++)
                {
                    int index_id = REG * i + j;
                    dfl = std::exp(float(cur_l_bbox_raw[index_id]) * l_bbox_scale[index_id]);
                    ltrb[i] += dfl * j;
                    sum += dfl;
                }
                ltrb[i] /= sum;
            }

            // 7.3.9 剔除不合格的框   if(x1 >= x2 || y1 >=y2) continue;
            // 7.3.9 Remove unqualified boxes
            if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0)
            {
                l_cls_raw += CLASSES_NUM;
                l_bbox_raw += REG * 4;
                continue;
            }

            // 7.3.10 dist 2 bbox (ltrb 2 xyxy)
            float x1 = (w + 0.5 - ltrb[0]) * 32.0;
            float y1 = (h + 0.5 - ltrb[1]) * 32.0;
            float x2 = (w + 0.5 + ltrb[2]) * 32.0;
            float y2 = (h + 0.5 + ltrb[3]) * 32.0;

            // 7.3.11 对应类别加入到对应的std::vector中
            // 7.3.11 Add the corresponding class to the corresponding std::vector.
            bboxes[cls_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
            scores[cls_id].push_back(score);

            // Mask的处理
            std::vector<float> mc(MCES);
            for (int i = 0; i < MCES; i++)
            {
                mc[i] = float(cur_l_mces_raw[i]) * l_mces_scale[i];
            }
            maskes[cls_id].push_back(mc);

                        // // debug
                        // std::cout << "mcs:" << std::endl;
                        // for (int i = 0; i < MCES; i++)
                        // {
                        //     std::cout << mc[i] << " ";
                        // }
                        // std::cout << std::endl;

            l_cls_raw += CLASSES_NUM;
            l_bbox_raw += REG * 4;
            l_mces_raw += MCES;
        }
    }

    // 7.4 对每一个类别进行NMS
    // 7.4 NMS
    std::vector<std::vector<int>> indices(CLASSES_NUM);
    for (int i = 0; i < CLASSES_NUM; i++)
    {
        cv::dnn::NMSBoxes(bboxes[i], scores[i], SCORE_THRESHOLD, NMS_THRESHOLD, indices[i], 1.f, NMS_TOP_K);
    }
    std::cout << "\033[31m Post Process time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 << " ms\033[0m" << std::endl;

    // 8. 渲染
    // 8. Render
    begin_time = std::chrono::system_clock::now();
    cv::Mat zeros(img.size(), img.type(), cv::Scalar(0, 0, 0));

    // debug
    std::cout << "img (cols, rows, channels): (";
    std::cout << img.rows << ", ";
    std::cout << img.cols << ", ";
    std::cout << img.channels() << ")" << std::endl;

    std::cout << "zeros (cols, rows, channels): (";
    std::cout << zeros.rows << ", ";
    std::cout << zeros.cols << ", ";
    std::cout << zeros.channels() << ")" << std::endl;

    for (int cls_id = 0; cls_id < CLASSES_NUM; cls_id++)
    {
        // 8.1 每一个类别分别渲染
        // 8.1 Render for each class
        for (std::vector<int>::iterator it = indices[cls_id].begin(); it != indices[cls_id].end(); ++it)
        {
            // 8.2 获取基本的 bbox 信息
            // 8.2 Get basic bbox information
            // float x1 = (bboxes[cls_id][*it].x - x_shift) / x_scale;
            // float y1 = (bboxes[cls_id][*it].y - y_shift) / y_scale;
            // float x2 = x1 + (bboxes[cls_id][*it].width) / x_scale;
            // float y2 = y1 + (bboxes[cls_id][*it].height) / y_scale;
            int x1 = int((bboxes[cls_id][*it].x - x_shift) / x_scale);
            int y1 = int((bboxes[cls_id][*it].y - y_shift) / y_scale);
            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            int x2 = int(x1 + (bboxes[cls_id][*it].width) / x_scale);
            int y2 = int(y1 + (bboxes[cls_id][*it].height) / y_scale);
            x2 = std::min(int(img.cols - 1), x2);
            y2 = std::min(int(img.rows - 1), y2);
            float score = scores[cls_id][*it];
            std::string name = object_names[cls_id % CLASSES_NUM];

            // 8.3 绘制矩形
            // 8.3 Draw rect
            cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), rdk_colors[cls_id % 20], LINE_SIZE);

            // 8.4 绘制字体
            // 8.4 Draw text
            std::string text = name + ": " + std::to_string(static_cast<int>(score * 100)) + "%";
            cv::putText(img, text, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, FONT_SIZE, rdk_colors[(cls_id + 1) % 20], FONT_THICKNESS + 5, cv::LINE_AA);
            cv::putText(img, text, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, FONT_SIZE, rdk_colors[cls_id % 20], FONT_THICKNESS, cv::LINE_AA);

            // 8.5 打印检测信息
            // 8.5 Print detection information
            std::cout << std::endl;
            std::cout << "(" << x1 << " " << y1 << " " << x2 << " " << y2 << "): \t" << text << std::endl;

            // 8.6 对Mask进行合成和缩放 [TODO: GPU 加速]
            int x1_corp = int(bboxes[cls_id][*it].x * X_SCALE_CORP);
            int y1_corp = int(bboxes[cls_id][*it].y * X_SCALE_CORP);
            int x2_corp = int((bboxes[cls_id][*it].x + bboxes[cls_id][*it].width) * Y_SCALE_CORP);
            int y2_corp = int((bboxes[cls_id][*it].y + bboxes[cls_id][*it].height) * Y_SCALE_CORP);

            std::cout << "x1_corp: " << x1_corp << ", x2_corp: " << x2_corp << ", y1_corp: " << y1_corp << ", y2_corp: " << y2_corp << std::endl;
            cv::Mat mask(y2 - y1, x2 - x1, CV_8UC1, cv::Scalar(0));
            // debug
            std::cout << "mcs:" << std::endl;
            for (int i = 0; i < MCES; i++)
            {
                std::cout << maskes[cls_id][*it][i] << " ";
            }
            std::cout << std::endl;
            double a[] = {-0.5078671, -1.7792891, 0.22384067, -0.031085026, 0.040541567, 0.42481288, 0.004164445, -0.019916827, 0.07834157, -1.5329591, -0.6799558, -1.3593721, 0.27600715, 0.7392442, 0.40653914, 0.094646804, -0.16810837, 0.85772616, -0.20207602, 0.0013240166, -0.06012171, 0.04421918, -0.023318088, 0.16049749, -0.07510812, -0.57381636, -0.27384806, -1.015032, 0.10577665, -1.3185172, -0.8278994, -0.019991808,};

            
            for (int x = x1_corp; x < x2_corp; x++)
            {
                for (int y = y1_corp; y < y2_corp; y++)
                {
                    float sum = 0.0;
                    for (int i = 0; i < MCES; i++)
                    {
                        // int index = y * W_4 * MCES + x * MCES + i;
                        int index = (x * H_4 * MCES) + (y  * MCES) + i;
                        // std::cout << proto[index] << " ";
                        // sum += proto[index] * maskes[cls_id][*it][i];
                        sum += a[i] * proto[index];
                    }
                    if (sum > 0.0)
                    {
                        mask.at<uint8_t>(y - y1_corp, x - x1_corp) = 1;
                    }
                }
            }
            cv::Mat mask_resized;
            cv::resize(mask, mask_resized, cv::Size(x2 - x1, y2 - y1));

            // debug
            std::cout << "mask_resized (cols, rows, channels): (";
            std::cout << mask_resized.rows << ", ";
            std::cout << mask_resized.cols << ", ";
            std::cout << mask_resized.channels() << ")" << std::endl;

            std::cout << "x1, y1, x2, y2: (" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << ")" << std::endl;
            std::cout << "(x2 - x1, y2 - y1)" << ": (" << (x2 - x1) << ", " << (y2 - y1) << ")" << std::endl;

            zeros(cv::Rect(x1, y1, x2 - x1, y2 - y1)).setTo(rdk_colors[(cls_id + 21) % 20], mask_resized == 1);
        }
    }
    cv::Mat add_result;
    cv::addWeighted(img, 1.0, zeros, 0.3, 0, add_result);
    cv::Mat concatenated_result;
    cv::hconcat(img, zeros, concatenated_result);
    cv::hconcat(concatenated_result, add_result, concatenated_result);
    std::cout << "\033[31m Draw Result time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 << " ms\033[0m" << std::endl;

    // 9. 保存
    // 9. Save
    cv::imwrite(IMG_SAVE_PATH, concatenated_result);

    // 10. 释放任务
    // 10. Release task
    hbDNNReleaseTask(task_handle);

    // 11. 释放内存
    // 11. Release memory
    hbSysFreeMem(&(input.sysMem[0]));
    for (int i = 0; i < 6; i++)
        hbSysFreeMem(&(output[i].sysMem[0]));

    // 12. 释放模型
    // 12. Release model
    hbDNNRelease(packed_dnn_handle);

    debug.close();

    return 0;
}