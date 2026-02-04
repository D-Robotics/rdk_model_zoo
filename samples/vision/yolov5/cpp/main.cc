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
// #define MODEL_PATH "../../models/yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin"
#define MODEL_PATH "../../models/yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin"

// 推理使用的测试图片路径
// Path of the test image used for inference.
// #define TESR_IMG_PATH "../../../../../resource/assets/kite.jpg"
#define TESR_IMG_PATH "../../../../../resource/assets/bus.jpg"

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

// 模型的anchors
// anchors
#define ANCHORS 10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0

// NMS的阈值, 默认0.45
// Non-Maximum Suppression (NMS) threshold, default is 0.45
#define NMS_THRESHOLD 0.45

// 分数阈值, 默认0.25
// Score threshold, default is 0.25
#define SCORE_THRESHOLD 0.25

// NMS选取的前K个框数, 默认300
// Number of top-K boxes selected by NMS, default is 300
#define NMS_TOP_K 300

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

    // 2.3.1 D-Robotics YOLOv5-Detect *.bin 模型应该为单输入
    // 2.3.1 D-Robotics YOLOv5-Detect *.bin model should have only one input
    if (input_count > 1)
    {
        std::cout << "Your Model have more than 1 input, please check!" << std::endl;
        return -1;
    }

    // 2.3.2 D-Robotics YOLOv5-Detect *.bin 模型输入Tensor类型应为nv12
    // tensor type: HB_DNN_IMG_TYPE_NV12
    if (input_properties.validShape.numDimensions == 4)
    {
        std::cout << "input tensor type: HB_DNN_IMG_TYPE_NV12" << std::endl;
    }
    else
    {
        std::cout << "input tensor type is not HB_DNN_IMG_TYPE_NV12, please check!" << std::endl;
        return -1;
    }

    // 2.3.3 D-Robotics YOLOv5-Detect *.bin 模型输入Tensor数据排布应为NCHW
    // tensor layout: HB_DNN_LAYOUT_NCHW
    if (input_properties.tensorType == 1)
    {
        std::cout << "input tensor layout: HB_DNN_LAYOUT_NCHW" << std::endl;
    }
    else
    {
        std::cout << "input tensor layout is not HB_DNN_LAYOUT_NCHW, please check!" << std::endl;
        return -1;
    }

    // 2.3.4 D-Robotics YOLOv5-Detect *.bin 模型输入Tensor数据的valid shape应为(1,3,H,W)
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

    // 2.4.1 D-Robotics YOLOv5-Detect *.bin 模型应该有 3 个输出
    // 2.4.1 D-Robotics YOLOv5-Detect *.bin model should have 3 outputs
    if (output_count == 3)
    {
        for (int i = 0; i < 3; i++)
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
        std::cout << "Your Model's outputs num is not 3, please check!" << std::endl;
        return -1;
    }

    // 2.4.2 调整输出头顺序的映射
    // 2.4.2 Adjust the mapping of output order
    int order[3] = {0, 1, 2};
    int32_t H_8 = input_H / 8;
    int32_t H_16 = input_H / 16;
    int32_t H_32 = input_H / 32;
    int32_t W_8 = input_W / 8;
    int32_t W_16 = input_W / 16;
    int32_t W_32 = input_W / 32;
    int32_t order_we_want[6][3] = {
        {H_8, W_8, 3 * (5 * CLASSES_NUM)},   // output[order[0]]: (1, H // 8,  W // 8,  3 × (5 + CLASSES_NUM))
        {H_16, W_16, 3 * (5 * CLASSES_NUM)}, // output[order[1]]: (1, H // 16, W // 16, 3 × (5 + CLASSES_NUM))
        {H_32, W_32, 3 * (5 * CLASSES_NUM)}, // output[order[2]]: (1, H // 32, W // 32, 3 × (5 + CLASSES_NUM))
    };
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
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
    if (order[0] + order[1] + order[2] == 0 + 1 + 2)
    {
        std::cout << "Outputs order check SUCCESS, continue." << std::endl;
        std::cout << "order = {";
        for (int i = 0; i < 3; i++)
        {
            std::cout << order[i] << ", ";
        }
        std::cout << "}" << std::endl;
    }
    else
    {
        std::cout << "Outputs order check FAILED, use default" << std::endl;
        for (int i = 0; i < 3; i++)
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
    for (int i = 0; i < 3; i++)
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

    // 7. YOLOv5-Detect 后处理
    // 7. Postprocess
    float CONF_THRES_RAW = -log(1 / SCORE_THRESHOLD - 1);     // 利用反函数作用阈值，利用单调性筛选
    std::vector<std::vector<cv::Rect2d>> bboxes(CLASSES_NUM); // 每个id的 xyhw 信息使用一个std::vector<cv::Rect2d>存储
    std::vector<std::vector<float>> scores(CLASSES_NUM);      // 每个id的score信息使用一个std::vector<float>存储
    std::vector<float> anchors = {ANCHORS};                   // 锚框信息
    if (anchors.size() != 18)
    {
        std::cout << "Anchors size is not 18, please check!" << std::endl;
        return -1;
    }
    std::cout << "anchors: ";
    for (auto it = anchors.begin(); it != anchors.end(); ++it)
    {
        std::cout << *it << "  ";
    }
    std::cout << std::endl;
    std::vector<std::pair<double, double>> s_anchors = {{anchors[0], anchors[1]},
                                                        {anchors[2], anchors[3]},
                                                        {anchors[4], anchors[5]}};
    std::vector<std::pair<double, double>> m_anchors = {{anchors[6], anchors[7]},
                                                        {anchors[8], anchors[9]},
                                                        {anchors[10], anchors[11]}};
    std::vector<std::pair<double, double>> l_anchors = {{anchors[12], anchors[13]},
                                                        {anchors[14], anchors[15]},
                                                        {anchors[16], anchors[17]}};

    begin_time = std::chrono::system_clock::now();

    // 7.1 小目标特征图
    // 7.1 Small Object Feature Map
    // output[order[0]]: (1, H // 8,  W // 8,  3 × (5 + CLASSES_NUM))

    // 7.1.1 检查反量化类型是否符合RDK Model Zoo的README导出的bin模型规范
    // 7.1.1 Check if the dequantization type complies with the bin model specification exported in the RDK Model Zoo README.
    if (output[order[0]].properties.quantiType != NONE)
    {
        std::cout << "output[order[0]] QuantiType should be NONE, please check!" << std::endl;
        return -1;
    }

    // 7.1.2 对缓存的BPU内存进行刷新
    // 7.1.2 Flush the cached BPU memory
    hbSysFlushMem(&(output[order[0]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

    // 7.1.3 将BPU推理完的内存地址转换为对应类型的指针
    // 7.1.3 Convert the memory address of BPU inference to a pointer of the corresponding type
    auto *s_raw = reinterpret_cast<float *>(output[order[0]].sysMem[0].virAddr);
    for (int h = 0; h < H_8; h++)
    {
        for (int w = 0; w < W_8; w++)
        {

            for (auto anchor = s_anchors.begin(); anchor != s_anchors.end(); anchor++)
            {
                // 7.1.4 取对应H, W, anchor位置的C通道, 记为数组的形式
                // 7.1.4 Extract the C channel corresponding to the H, W, and anchor positions, represented as an array.
                float *cur_s_raw = s_raw;
                s_raw += (5 + CLASSES_NUM);

                // 7.1.5 如果条件概率小于conf, 则全概率一定小于conf, 舍去
                // 7.1.5 If the conditional probability is less than conf, all probabilities must be less than conf, so it is discarded
                if (cur_s_raw[4] < CONF_THRES_RAW)
                    continue;

                // 7.1.6 找到分数的最大值索引, 如果最大值小于阈值，则舍去
                // 7.1.6 Find the index of the maximum score value and discard if the maximum value is less than the threshold
                int cls_id = 5;
                int end = CLASSES_NUM + 5;
                for (int i = 6; i < end; i++)
                {
                    if (cur_s_raw[i] > cur_s_raw[cls_id])
                    {
                        cls_id = i;
                    }
                }
                float score = 1.0 / (1.0 + std::exp(-cur_s_raw[4])) / (1.0 + std::exp(-cur_s_raw[cls_id]));

                // 7.1.7 不合格则直接跳过, 避免无用的dist2bbox计算
                // 7.1.7 Skip if not qualified to avoid unnecessary dist2bbox calculation
                if (score < SCORE_THRESHOLD)
                    continue;
                cls_id -= 5;

                // 7.1.8 特征解码计算
                // 7.1.8 Feature decoding calculation
                float center_x = ((1.0 / (1.0 + std::exp(-cur_s_raw[0]))) * 2 - 0.5 + w) * 8;
                float center_y = ((1.0 / (1.0 + std::exp(-cur_s_raw[1]))) * 2 - 0.5 + h) * 8;
                float bbox_w = std::pow((1.0 / (1.0 + std::exp(-cur_s_raw[2]))) * 2, 2) * (*anchor).first;
                float bbox_h = std::pow((1.0 / (1.0 + std::exp(-cur_s_raw[3]))) * 2, 2) * (*anchor).second;
                float bbox_x = center_x - bbox_w / 2.0;
                float bbox_y = center_y - bbox_h / 2.0;

                // 7.1.9 对应类别加入到对应的std::vector中
                // 7.1.9 Add the corresponding class to the corresponding std::vector.
                bboxes[cls_id].push_back(cv::Rect2d(bbox_x, bbox_y, bbox_w, bbox_h));
                scores[cls_id].push_back(score);
            }
        }
    }

    // 7.2 中目标特征图
    // 7.2 Media Object Feature Map
    // output[order[0]]: (1, H // 16,  W // 16,  3 × (5 + CLASSES_NUM))

    // 7.2.1 检查反量化类型是否符合RDK Model Zoo的README导出的bin模型规范
    // 7.2.1 Check if the dequantization type complies with the bin model specification exported in the RDK Model Zoo README.
    if (output[order[1]].properties.quantiType != NONE)
    {
        std::cout << "output[order[0]] QuantiType should be NONE, please check!" << std::endl;
        return -1;
    }

    // 7.2.2 对缓存的BPU内存进行刷新
    // 7.2.2 Flush the cached BPU memory
    hbSysFlushMem(&(output[order[1]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

    // 7.2.3 将BPU推理完的内存地址转换为对应类型的指针
    // 7.2.3 Convert the memory address of BPU inference to a pointer of the corresponding type
    auto *m_raw = reinterpret_cast<float *>(output[order[1]].sysMem[0].virAddr);
    for (int h = 0; h < H_16; h++)
    {
        for (int w = 0; w < W_16; w++)
        {

            for (auto anchor = m_anchors.begin(); anchor != m_anchors.end(); anchor++)
            {
                // 7.2.4 取对应H, W, anchor位置的C通道, 记为数组的形式
                // 7.2.4 Extract the C channel corresponding to the H, W, and anchor positions, represented as an array.
                float *cur_m_raw = m_raw;
                m_raw += (5 + CLASSES_NUM);

                // 7.2.5 如果条件概率小于conf, 则全概率一定小于conf, 舍去
                // 7.2.5 If the conditional probability is less than conf, all probabilities must be less than conf, so it is discarded
                if (cur_m_raw[4] < CONF_THRES_RAW)
                    continue;

                // 7.2.6 找到分数的最大值索引, 如果最大值小于阈值，则舍去
                // 7.2.6 Find the index of the maximum score value and discard if the maximum value is less than the threshold
                int cls_id = 5;
                int end = CLASSES_NUM + 5;
                for (int i = 6; i < end; i++)
                {
                    if (cur_m_raw[i] > cur_m_raw[cls_id])
                    {
                        cls_id = i;
                    }
                }
                float score = 1.0 / (1.0 + std::exp(-cur_m_raw[4])) / (1.0 + std::exp(-cur_m_raw[cls_id]));

                // 7.2.7 不合格则直接跳过, 避免无用的dist2bbox计算
                // 7.2.7 Skip if not qualified to avoid unnecessary dist2bbox calculation
                if (score < SCORE_THRESHOLD)
                    continue;
                cls_id -= 5;

                // 7.2.8 特征解码计算
                // 7.2.8 Feature decoding calculation
                float center_x = ((1.0 / (1.0 + std::exp(-cur_m_raw[0]))) * 2 - 0.5 + w) * 16;
                float center_y = ((1.0 / (1.0 + std::exp(-cur_m_raw[1]))) * 2 - 0.5 + h) * 16;
                float bbox_w = std::pow((1.0 / (1.0 + std::exp(-cur_m_raw[2]))) * 2, 2) * (*anchor).first;
                float bbox_h = std::pow((1.0 / (1.0 + std::exp(-cur_m_raw[3]))) * 2, 2) * (*anchor).second;
                float bbox_x = center_x - bbox_w / 2.0;
                float bbox_y = center_y - bbox_h / 2.0;

                // 7.2.9 对应类别加入到对应的std::vector中
                // 7.2.9 Add the corresponding class to the corresponding std::vector.
                bboxes[cls_id].push_back(cv::Rect2d(bbox_x, bbox_y, bbox_w, bbox_h));
                scores[cls_id].push_back(score);
            }
        }
    }

    // 7.3 大目标特征图
    // 7.3 Large Object Feature Map
    // output[order[0]]: (1, H // 32,  W // 32,  3 × (5 + CLASSES_NUM))

    // 7.3.1 检查反量化类型是否符合RDK Model Zoo的README导出的bin模型规范
    // 7.3.1 Check if the dequantization type complies with the bin model specification exported in the RDK Model Zoo README.
    if (output[order[2]].properties.quantiType != NONE)
    {
        std::cout << "output[order[0]] QuantiType should be NONE, please check!" << std::endl;
        return -1;
    }

    // 7.3.2 对缓存的BPU内存进行刷新
    // 7.3.2 Flush the cached BPU memory
    hbSysFlushMem(&(output[order[2]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

    // 7.3.3 将BPU推理完的内存地址转换为对应类型的指针
    // 7.3.3 Convert the memory address of BPU inference to a pointer of the corresponding type
    auto *l_raw = reinterpret_cast<float *>(output[order[2]].sysMem[0].virAddr);
    for (int h = 0; h < H_32; h++)
    {
        for (int w = 0; w < W_32; w++)
        {

            for (auto anchor = l_anchors.begin(); anchor != l_anchors.end(); anchor++)
            {
                // 7.3.4 取对应H, W, anchor位置的C通道, 记为数组的形式
                // 7.1.4 Extract the C channel corresponding to the H, W, and anchor positions, represented as an array.
                float *cur_l_raw = l_raw;
                l_raw += (5 + CLASSES_NUM);

                // 7.3.5 如果条件概率小于conf, 则全概率一定小于conf, 舍去
                // 7.3.5 If the conditional probability is less than conf, all probabilities must be less than conf, so it is discarded
                if (cur_l_raw[4] < CONF_THRES_RAW)
                    continue;

                // 7.3.6 找到分数的最大值索引, 如果最大值小于阈值，则舍去
                // 7.3.6 Find the index of the maximum score value and discard if the maximum value is less than the threshold
                int cls_id = 5;
                int end = CLASSES_NUM + 5;
                for (int i = 6; i < end; i++)
                {
                    if (cur_l_raw[i] > cur_l_raw[cls_id])
                    {
                        cls_id = i;
                    }
                }
                float score = 1.0 / (1.0 + std::exp(-cur_l_raw[4])) / (1.0 + std::exp(-cur_l_raw[cls_id]));

                // 7.3.7 不合格则直接跳过, 避免无用的dist2bbox计算
                // 7.1.7 Skip if not qualified to avoid unnecessary dist2bbox calculation
                if (score < SCORE_THRESHOLD)
                    continue;
                cls_id -= 5;

                // 7.3.8 特征解码计算
                // 7.3.8 Feature decoding calculation
                float center_x = ((1.0 / (1.0 + std::exp(-cur_l_raw[0]))) * 2 - 0.5 + w) * 32;
                float center_y = ((1.0 / (1.0 + std::exp(-cur_l_raw[1]))) * 2 - 0.5 + h) * 32;
                float bbox_w = std::pow((1.0 / (1.0 + std::exp(-cur_l_raw[2]))) * 2, 2) * (*anchor).first;
                float bbox_h = std::pow((1.0 / (1.0 + std::exp(-cur_l_raw[3]))) * 2, 2) * (*anchor).second;
                float bbox_x = center_x - bbox_w / 2.0;
                float bbox_y = center_y - bbox_h / 2.0;

                // 7.3.9 对应类别加入到对应的std::vector中
                // 7.3.9 Add the corresponding class to the corresponding std::vector.
                bboxes[cls_id].push_back(cv::Rect2d(bbox_x, bbox_y, bbox_w, bbox_h));
                scores[cls_id].push_back(score);
            }
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

            // 8.3 绘制矩形
            // 8.3 Draw rect
            cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), LINE_SIZE);

            // 8.4 绘制字体
            // 8.4 Draw text
            std::string text = name + ": " + std::to_string(static_cast<int>(score * 100)) + "%";
            cv::putText(img, text, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, FONT_SIZE, cv::Scalar(0, 0, 255), FONT_THICKNESS, cv::LINE_AA);

            // 8.5 打印检测信息
            // 8.5 Print detection information
            std::cout << "(" << x1 << " " << y1 << " " << x2 << " " << y2 << "): \t" << text << std::endl;
        }
    }
    std::cout << "\033[31m Draw Result time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 << " ms\033[0m" << std::endl;

    // 9. 保存
    // 9. Save
    cv::imwrite("cpp_result.jpg", img);

    // 10. 释放任务
    // 10. Release task
    hbDNNReleaseTask(task_handle);

    // 11. 释放内存
    // 11. Release memory
    hbSysFreeMem(&(input.sysMem[0]));
    hbSysFreeMem(&(output->sysMem[0]));

    // 12. 释放模型
    // 12. Release model
    hbDNNRelease(packed_dnn_handle);

    return 0;
}