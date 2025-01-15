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

// D-Robotics *.bin 模型路径
// Path of D-Robotics *.bin model.
#define MODEL_PATH "../../ptq_models/yolov8s_detect_bayese_640x640_nchwrgb_mix_modified.bin"

// 推理结果的*.txt文件的存储路径
#define JSON_RESULT_PATH "yolov8s_detect_bayese_640x640_nchwrgb_mix_coco2017_val_pridect.json"

// 推理结果的*.txt文件的存储路径
//
#define TXT_RESULT_PATH "cpp_json_result.txt"


//
// COCO2017 val source image path.
#define IMAGE_PATH "../../../../datasets/COCO2017/val2017"

// 前处理方式选择, 0:Resize, 1:LetterBox
// Preprocessing method selection, 0: Resize, 1: LetterBox
#define RESIZE_TYPE 0
#define LETTERBOX_TYPE 1
#define PREPROCESS_TYPE LETTERBOX_TYPE

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

// 绘制标签的字体尺寸, 默认1.0
// Font size for drawing labels, default is 1.0.
#define FONT_SIZE 1.0

// 绘制标签的字体粗细, 默认 1.0
// Font thickness for drawing labels, default is 1.0.
#define FONT_THICKNESS 1.0

// 绘制矩形框的线宽, 默认2.0
// Line width for drawing bounding boxes, default is 2.0.
#define LINE_SIZE 2.0

// C/C++ Standard Librarys
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <cstdlib>

namespace fs = std::filesystem;

// Thrid Party Librarys
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

// RDK BPU libDNN API
#include "dnn/hb_dnn.h"
#include "dnn/hb_dnn_ext.h"
#include "dnn/plugin/hb_dnn_layer.h"
#include "dnn/plugin/hb_dnn_plugin.h"
#include "dnn/hb_sys.h"

#define RDK_CHECK_SUCCESS(value, errmsg)                        \
    do                                                          \
    {                                                           \
        /*value can be call of function*/                       \
        auto ret_code = value;                                  \
        if (ret_code != 0)                                      \
        {                                                       \
            std::cout << errmsg << ", error code:" << ret_code; \
            return ret_code;                                    \
        }                                                       \
    } while (0);

// COCO Names
std::vector<std::string> object_names = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

int main()
{
    // 0. 打印相关版本信息
    // std::cout << "OpenCV build details: " << cv::getBuildInformation() << std::endl;
    std::cout << "[OpenCV] Version: " << CV_VERSION << std::endl;

    // 1. 加载bin模型
    auto begin_time = std::chrono::system_clock::now();
    hbPackedDNNHandle_t packed_dnn_handle;
    const char *model_file_name = MODEL_PATH;
    RDK_CHECK_SUCCESS(
        hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1),
        "hbDNNInitializeFromFiles failed");
    std::cout << "\033[31m Load D-Robotics Quantize model time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 << " ms\033[0m" << std::endl;

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

    // 2.3.1 D-Robotics YOLOv8 *.bin 模型应该为单输入
    // 2.3.1 D-Robotics YOLOv8 *.bin model should have only one input
    if (input_count > 1)
    {
        std::cout << "Your Model have more than 1 input, please check!" << std::endl;
        return -1;
    }

    // 2.3.2 D-Robotics YOLOv8 *.bin 模型输入Tensor类型应为nv12
    // tensor type: HB_DNN_IMG_TYPE_NV12
    if (input_properties.tensorType == HB_DNN_IMG_TYPE_RGB)
    {
        std::cout << "input tensor type: HB_DNN_IMG_TYPE_RGB" << std::endl;
    }
    else
    {
        std::cout << "input tensor type is not HB_DNN_IMG_TYPE_RGB, please check!" << std::endl;
        return -1;
    }

    // 2.3.3 D-Robotics YOLOv8 *.bin 模型输入Tensor数据排布应为NCHW
    // tensor layout: HB_DNN_LAYOUT_NCHW
    std::cout << "input_properties.tensorType: " << input_properties.tensorType << std::endl;
    if (input_properties.tensorLayout == HB_DNN_LAYOUT_NCHW)
    {
        std::cout << "input tensor layout: HB_DNN_LAYOUT_NCHW" << std::endl;
    }
    else
    {
        std::cout << "input tensor layout is not HB_DNN_LAYOUT_NCHW, please check!" << std::endl;
        return -1;
    }

    // 2.3.4 D-Robotics YOLOv8 *.bin 模型输入Tensor数据的valid shape应为(1,3,H,W)
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

    // 2.4.1 D-Robotics YOLOv8 *.bin 模型应该有6个输出
    // 2.4.1 D-Robotics YOLOv8 *.bin model should have 6 outputs
    if (output_count == 6)
    {
        for (int i = 0; i < 6; i++)
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
        std::cout << "Your Model's outputs num is not 6, please check!" << std::endl;
        return -1;
    }

    // 2.4.2 调整输出头顺序的映射
    // 2.4.2 Adjust the mapping of output order
    int order[6] = {0, 1, 2, 3, 4, 5};
    int32_t H_8 = input_H / 8;
    int32_t H_16 = input_H / 16;
    int32_t H_32 = input_H / 32;
    int32_t W_8 = input_W / 8;
    int32_t W_16 = input_W / 16;
    int32_t W_32 = input_W / 32;
    int32_t order_we_want[6][3] = {
        {H_8, W_8, CLASSES_NUM},   // output[order[3]]: (1, H // 8,  W // 8,  CLASSES_NUM)
        {H_8, W_8, 64},            // output[order[0]]: (1, H // 8,  W // 8,  64)
        {H_16, W_16, CLASSES_NUM}, // output[order[4]]: (1, H // 16, W // 16, CLASSES_NUM)
        {H_16, W_16, 64},          // output[order[1]]: (1, H // 16, W // 16, 64)
        {H_32, W_32, CLASSES_NUM}, // output[order[5]]: (1, H // 32, W // 32, CLASSES_NUM)
        {H_32, W_32, 64},          // output[order[2]]: (1, H // 32, W // 32, 64)
    };
    for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j < 6; j++)
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
    if (order[0] + order[1] + order[2] + order[3] + order[4] + order[5] == 0 + 1 + 2 + 3 + 4 + 5)
    {
        std::cout << "Outputs order check SUCCESS, continue." << std::endl;
        std::cout << "order = {";
        for (int i = 0; i < 6; i++)
        {
            std::cout << order[i] << ", ";
        }
        std::cout << "}" << std::endl;
    }
    else
    {
        std::cout << "Outputs order check FAILED, use default" << std::endl;
        for (int i = 0; i < 6; i++)
            order[i] = i;
    }

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
            {
                file_paths.push_back(entry.path());
            }
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

    // 准备输入数据的hbDNNTensor内存空间
    hbDNNTensor input;
    input.properties = input_properties;
    hbSysAllocCachedMem(&input.sysMem[0], int(3 * input_H * input_W));

    // 准备模型输出数据的空间
    hbDNNTensor *output = new hbDNNTensor[output_count];
    for (int i = 0; i < 6; i++)
    {
        hbDNNTensorProperties &output_properties = output[i].properties;
        hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i);
        int out_aligned_size = output_properties.alignedByteSize;
        hbSysMem &mem = output[i].sysMem[0];
        hbSysAllocCachedMem(&mem, out_aligned_size);
    }

    // 利用反函数作用阈值，利用单调性筛选
    float CONF_THRES_RAW = -log(1 / SCORE_THRESHOLD - 1);

    // 板端直接推理所有验证集，并生成*.txt标签和渲染结果 (串行程序)
    size_t image_total_cnt = file_paths.size();
    std::stringstream cpp_results;

    for (size_t i = 0; i < image_total_cnt; ++i)
    {
        // if (i > 100)
        //     break;
        std::cout << "\033[1;32m" << "[" << (i + 1) << " / " << image_total_cnt << "] " << file_paths[i] << '\n'
                  << "\033[0m";

        cv::Mat img = cv::imread(file_paths[i].string());
        // std::cout << "img (cols, rows, channels): (";
        // std::cout << img.rows << ", ";
        // std::cout << img.cols << ", ";
        // std::cout << img.channels() << ")" << std::endl;

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
            // cv::cvtColor(resize_img, resize_img, cv::COLOR_BGR2RGB);

            std::cout << "\033[31m pre process (LetterBox) time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 << " ms\033[0m" << std::endl;
        }
        else if (PREPROCESS_TYPE == RESIZE_TYPE) // resize
        {
            begin_time = std::chrono::system_clock::now();

            cv::Size targetSize(input_W, input_H);
            cv::resize(img, resize_img, targetSize);
            // cv::cvtColor(resize_img, resize_img, cv::COLOR_BGR2RGB);

            y_scale = 1.0 * input_H / img.rows;
            x_scale = 1.0 * input_W / img.cols;
            y_shift = 0;
            x_shift = 0;

            std::cout << "\033[31m pre process (Resize) time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 << " ms\033[0m" << std::endl;
        }
        // std::cout << "y_scale = " << y_scale << ", ";
        // std::cout << "x_scale = " << x_scale << std::endl;
        // std::cout << "y_shift = " << y_shift << ", ";
        // std::cout << "x_shift = " << x_shift << std::endl;

        // 3.4 cv::Mat的NHWC-BGR888格式转为NCHW-RGB888格式
        // 3.4 Convert NHWC-BGR888 to NCHW-RGB888
        begin_time = std::chrono::system_clock::now();
        uint8_t *data_u8{reinterpret_cast<uint8_t *>(resize_img.ptr<uint8_t>())};
        int8_t *data_s8{reinterpret_cast<int8_t *>(input.sysMem[0].virAddr)};

        for (int h = 0; h < input_H; h++)
        {
            for (int w = 0; w < input_W; w++)
            {
                // BGR 到 RGB 的通道顺序转换，同时进行减均值操作 (-128)
                data_s8[(0 * input_H * input_W) + h * input_W + w] = static_cast<int8_t>(data_u8[h * input_W * 3 + w * 3 + 2] - 128); // R
                data_s8[(1 * input_H * input_W) + h * input_W + w] = static_cast<int8_t>(data_u8[h * input_W * 3 + w * 3 + 1] - 128); // G
                data_s8[(2 * input_H * input_W) + h * input_W + w] = static_cast<int8_t>(data_u8[h * input_W * 3 + w * 3 + 0] - 128); // B
            }
        }

        std::cout << "\033[31m (u8-128)->s8 time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 << " ms\033[0m" << std::endl;

        begin_time = std::chrono::system_clock::now();

        hbSysFlushMem(&input.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);

        // 推理模型
        hbDNNTaskHandle_t task_handle = nullptr;
        hbDNNInferCtrlParam infer_ctrl_param;
        HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
        hbDNNInfer(&task_handle, &output, &input, dnn_handle, &infer_ctrl_param);

        // 等待任务结束
        hbDNNWaitTaskDone(task_handle, 0);
        std::cout << "\033[31m forward time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 << " ms\033[0m" << std::endl;

        // YOLOv8-Detect 后处理
        std::vector<std::vector<cv::Rect2d>> bboxes(CLASSES_NUM); // 每个id的xyhw 信息使用一个std::vector<cv::Rect2d>存储
        std::vector<std::vector<float>> scores(CLASSES_NUM);      // 每个id的score信息使用一个std::vector<float>存储

        begin_time = std::chrono::system_clock::now();

        // 7.1 小目标特征图
        // 7.1 Small Object Feature Map
        // output[order[0]]: (1, H // 8,  W // 8,  CLASSES_NUM)
        // output[order[1]]: (1, H // 8,  W // 8,  4 * REG)

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

        // 7.1.2 对缓存的BPU内存进行刷新
        // 7.1.2 Flush the cached BPU memory
        hbSysFlushMem(&(output[order[0]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
        hbSysFlushMem(&(output[order[1]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

        // 7.1.3 将BPU推理完的内存地址转换为对应类型的指针
        // 7.1.3 Convert the memory address of BPU inference to a pointer of the corresponding type
        auto *s_cls_raw = reinterpret_cast<float *>(output[order[0]].sysMem[0].virAddr);
        auto *s_bbox_raw = reinterpret_cast<int32_t *>(output[order[1]].sysMem[0].virAddr);
        auto *s_bbox_scale = reinterpret_cast<float *>(output[order[1]].properties.scale.scaleData);
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

                s_cls_raw += CLASSES_NUM;
                s_bbox_raw += REG * 4;
            }
        }

        // 7.2 中目标特征图
        // 7.2 Media Object Feature Map
        // output[order[2]]: (1, H // 16,  W // 16,  CLASSES_NUM)
        // output[order[3]]: (1, H // 16,  W // 16,  4 * REG)

        // 7.2.1 检查反量化类型是否符合RDK Model Zoo的README导出的bin模型规范
        // 7.2.1 Check if the dequantization type complies with the bin model specification exported in the RDK Model Zoo README.
        if (output[order[2]].properties.quantiType != NONE)
        {
            std::cout << "output[order[2]] QuantiType is not NONE, please check!" << std::endl;
            return -1;
        }
        if (output[order[3]].properties.quantiType != SCALE)
        {
            std::cout << "output[order[3]] QuantiType is not SCALE, please check!" << std::endl;
            return -1;
        }

        // 7.2.2 对缓存的BPU内存进行刷新
        // 7.2.2 Flush the cached BPU memory
        hbSysFlushMem(&(output[order[2]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
        hbSysFlushMem(&(output[order[3]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

        // 7.2.3 将BPU推理完的内存地址转换为对应类型的指针
        // 7.2.3 Convert the memory address of BPU inference to a pointer of the corresponding type
        auto *m_cls_raw = reinterpret_cast<float *>(output[order[2]].sysMem[0].virAddr);
        auto *m_bbox_raw = reinterpret_cast<int32_t *>(output[order[3]].sysMem[0].virAddr);
        auto *m_bbox_scale = reinterpret_cast<float *>(output[order[3]].properties.scale.scaleData);
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

                m_cls_raw += CLASSES_NUM;
                m_bbox_raw += REG * 4;
            }
        }

        // 7.3 大目标特征图
        // 7.3 Big Object Feature Map
        // output[order[4]]: (1, H // 32,  W // 32,  CLASSES_NUM)
        // output[order[5]]: (1, H // 32,  W // 32,  4 * REG)

        // 7.3.1 检查反量化类型是否符合RDK Model Zoo的README导出的bin模型规范
        // 7.3.1 Check if the dequantization type complies with the bin model specification exported in the RDK Model Zoo README.
        if (output[order[4]].properties.quantiType != NONE)
        {
            std::cout << "output[order[4]] QuantiType is not NONE, please check!" << std::endl;
            return -1;
        }
        if (output[order[5]].properties.quantiType != SCALE)
        {
            std::cout << "output[order[5]] QuantiType is not SCALE, please check!" << std::endl;
            return -1;
        }

        // 7.3.2 对缓存的BPU内存进行刷新
        // 7.3.2 Flush the cached BPU memory
        hbSysFlushMem(&(output[order[4]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
        hbSysFlushMem(&(output[order[5]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

        // 7.3.3 将BPU推理完的内存地址转换为对应类型的指针
        // 7.3.3 Convert the memory address of BPU inference to a pointer of the corresponding type
        auto *l_cls_raw = reinterpret_cast<float *>(output[order[4]].sysMem[0].virAddr);
        auto *l_bbox_raw = reinterpret_cast<int32_t *>(output[order[5]].sysMem[0].virAddr);
        auto *l_bbox_scale = reinterpret_cast<float *>(output[order[5]].properties.scale.scaleData);
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

                l_cls_raw += CLASSES_NUM;
                l_bbox_raw += REG * 4;
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
        for (int cls_id = 0; cls_id < CLASSES_NUM; cls_id++)
        {
            // 8.1 每一个类别分别渲染
            // 8.1 Render for each class
            for (std::vector<int>::iterator it = indices[cls_id].begin(); it != indices[cls_id].end(); ++it)
            {
                // 8.2 获取基本的 bbox 信息
                // 8.2 Get basic bbox information
                float x1 = (bboxes[cls_id][*it].x - x_shift) / x_scale;
                float y1 = (bboxes[cls_id][*it].y - y_shift) / y_scale;
                float x2 = x1 + (bboxes[cls_id][*it].width) / x_scale;
                float y2 = y1 + (bboxes[cls_id][*it].height) / y_scale;
                float score = scores[cls_id][*it];
                std::string name = object_names[cls_id % CLASSES_NUM];

                // 8.5 打印检测信息
                // 8.5 Print detection information
                // std::cout << "(" << x1 << " " << y1 << " " << x2 << " " << y2 << "): \t" << text << std::endl;

                // 保存pycocotools所需要的标签信息
                cpp_results << file_paths[i].filename().string() << " ";
                cpp_results << cls_id << " " << score << " ";
                cpp_results << x1 << " " << y1 << " " << x2 << " " << y2 << std::endl;
                
            }
        }
        
        // 10. 释放任务
        // 10. Release task
        hbDNNReleaseTask(task_handle);
    }

    // 11. 释放内存
    // 11. Release memory
    hbSysFreeMem(&(input.sysMem[0]));
    hbSysFreeMem(&(output->sysMem[0]));

    // 12. 释放模型
    // 12. Release model
    hbDNNRelease(packed_dnn_handle);


    // 保存pycocotools所需要的标签信息
    std::ofstream file_stream(TXT_RESULT_PATH);

    if (!file_stream.is_open()) {
        std::cerr << "Failed to open/create the file: " << TXT_RESULT_PATH << '\n';
        return 1;
    }

    file_stream << cpp_results.rdbuf();
    file_stream.close();

    // 验证文件是否已经正确关闭
    if (file_stream.is_open()) {
        std::cerr << "File did not close properly.\n";
    } else {
        std::cout << "\033[1;32m" << "result label saved: \"" << TXT_RESULT_PATH << "\"" <<  "\033[0m" << std::endl;
    }

    // 调用Python脚本将*.txt标签转化为*.json标签
    std::string txt_result_path = TXT_RESULT_PATH;
    std::string json_result_path = JSON_RESULT_PATH;
    std::string command = "python3 ../eval_convert.py --txt-path " + txt_result_path + " --json-path " + json_result_path;
    std::cout << command << std::endl;
    int result = system(command.c_str());
        if (result == -1) {
        std::cerr << "Failed to execute command\n";
    }else{
        std::cout << "python run success!" << std::endl;
    }
    return 0;
}