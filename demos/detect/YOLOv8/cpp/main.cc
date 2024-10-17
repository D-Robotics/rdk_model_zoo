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
#define MODEL_PATH "../../models/yolov8x_detect_bayese_640x640_nv12_modified.bin"
// #define TESR_IMG_PATH "../../../../../resource/assets/bus.jpg"
#define TESR_IMG_PATH "../../../../../resource/assets/kite.jpg"
#define IMG_SAVE_PATH "cpp_result.jpg"
#define CLASSES_NUM 80
#define NMS_THRESHOLD 0.45
#define SCORE_THRESHOLD 0.25
#define NMS_TOP_K 300
#define REG 16
#define FONT_SIZE 1.0
#define FONT_THINKNESS 1.0
#define LINE_SIZE 2.0

// C/C++ Standard Librarys
#include <iostream>
#include <vector>
#include <algorithm>

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
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

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
    std::cout << "\033[31m Load D-Robotics Quantize model time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count()/1000.0 << " ms\033[0m" << std::endl;


    // 2. 打印模型信息
    // 2.1 模型名称
    const char **model_name_list;
    int model_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle),
        "hbDNNGetModelNameList failed");
    if (model_count > 1)
    { // 如果这个bin模型有多个打包，则只使用第一个，一般只有一个
        std::cout << "This model file have more than 1 model, only use model 0.";
    }
    const char *model_name = model_name_list[0];
    std::cout << "[model name]: " << model_name << std::endl;

    // 2.2 获得Packed模型的第一个模型的handle
    hbDNNHandle_t dnn_handle;
    RDK_CHECK_SUCCESS(
        hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name),
        "hbDNNGetModelHandle failed");

    // 2.3 模型输入检查
    int32_t input_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetInputCount(&input_count, dnn_handle),
        "hbDNNGetInputCount failed");

    hbDNNTensorProperties input_properties;
    RDK_CHECK_SUCCESS(
        hbDNNGetInputTensorProperties(&input_properties, dnn_handle, 0),
        "hbDNNGetInputTensorProperties failed");

    // 2.3.1 D-Robotics YOLOv8 *.bin 模型应该为单输入
    if (input_count > 1)
    {
        std::cout << "Your Model have more than 1 input, please check!" << std::endl;
        return -1;
    }

    // 2.3.2 D-Robotics YOLOv8 *.bin 模型输入Tensor类型应为nv12
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

    // 2.3.3 D-Robotics YOLOv8 *.bin 模型输入Tensor数据排布应为NCHW
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
    int32_t output_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetOutputCount(&output_count, dnn_handle),
        "hbDNNGetOutputCount failed");

    // 2.4.1 D-Robotics YOLOv8 *.bin 模型应该有6个输出
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
    int order[6] = {0, 1, 2, 3, 4, 5};
    int32_t H_8 = input_H / 8;
    int32_t H_16 = input_H / 16;
    int32_t H_32 = input_H / 32;
    int32_t W_8 = input_W / 8;
    int32_t W_16 = input_W / 16;
    int32_t W_32 = input_W / 32;
    int32_t order_we_want[6][3] = {
        {H_8, W_8, 64},            // output[order[0]]: (1, H // 8,  W // 8,  64)
        {H_16, W_16, 64},          // output[order[1]]: (1, H // 16, W // 16, 64)
        {H_32, W_32, 64},          // output[order[2]]: (1, H // 32, W // 32, 64)
        {H_8, W_8, CLASSES_NUM},   // output[order[3]]: (1, H // 8,  W // 8,  CLASSES_NUM)
        {H_16, W_16, CLASSES_NUM}, // output[order[4]]: (1, H // 16, W // 16, CLASSES_NUM)
        {H_32, W_32, CLASSES_NUM}, // output[order[5]]: (1, H // 32, W // 32, CLASSES_NUM)
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

    // 3. 利用OpenCV准备nv12的输入数据
    // 注：实际量产中, 往往使用Codec, VPU, JPU等硬件来准备nv12输入数据
    // H264/H265 -> Decoder -> nv12 -> VPS/VSE -> BPU
    // V4L2 -> Decoder -> nv12 -> VPS/VSE -> BPU
    // MIPI -> nv12 -> VPS/VPS -> BPU
    // 3.1 利用OpenCV读取图像
    cv::Mat img = cv::imread(TESR_IMG_PATH);
    std::cout << "img path: " << TESR_IMG_PATH << std::endl;
    std::cout << "img (cols, rows, channels): (";
    std::cout << img.rows << ", ";
    std::cout << img.cols << ", ";
    std::cout << img.channels() << ")" << std::endl;

    // 3.2 使用Resize暴力拉伸的方式完成前处理
    begin_time = std::chrono::system_clock::now();
    cv::Size targetSize(input_W, input_H);
    cv::Mat resize_img;
    cv::resize(img, resize_img, targetSize);
    std::cout << "resize_img (cols, rows, channels): (";
    std::cout << resize_img.rows << ", ";
    std::cout << resize_img.cols << ", ";
    std::cout << resize_img.channels() << ")" << std::endl;
    float y_scale = 1.0 * img.rows / resize_img.rows;
    float x_scale = 1.0 * img.cols / resize_img.cols;
    std::cout << "y_scale = " << y_scale << ", ";
    std::cout << "x_scale = " << x_scale << std::endl;
    std::cout << "\033[31m pre process time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count()/1000.0 << " ms\033[0m" << std::endl;


    // 3.3 cv::Mat的BGR888格式转为YUV420SP格式
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
    std::cout << "\033[31m bgr8 to nv12 time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count()/1000.0 << " ms\033[0m" << std::endl;

    begin_time = std::chrono::system_clock::now();
    // 3.4 将准备好的输入数据放入hbDNNTensor
    hbDNNTensor input;
    input.properties = input_properties;
    hbSysAllocCachedMem(&input.sysMem[0], int(3 * input_H * input_W / 2));
    memcpy(input.sysMem[0].virAddr, ynv12, int(3 * input_H * input_W / 2));
    hbSysFlushMem(&input.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);

    // 4. 准备模型输出数据的空间
    hbDNNTensor *output = new hbDNNTensor[output_count];
    for (int i = 0; i < 6; i++)
    {
        hbDNNTensorProperties &output_properties = output[i].properties;
        hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i);
        int out_aligned_size = output_properties.alignedByteSize;
        hbSysMem &mem = output[i].sysMem[0];
        hbSysAllocCachedMem(&mem, out_aligned_size);
    }

    // 5. 推理模型
    hbDNNTaskHandle_t task_handle = nullptr;
    hbDNNInferCtrlParam infer_ctrl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
    hbDNNInfer(&task_handle,
               &output,
               &input,
               dnn_handle,
               &infer_ctrl_param);

    // 6. 等待任务结束
    hbDNNWaitTaskDone(task_handle, 0);
    std::cout << "\033[31m forward time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count()/1000.0 << " ms\033[0m" << std::endl;


    // 7. YOLOv8-Detect 后处理
    float CONF_THRES_RAW = -log(1 / SCORE_THRESHOLD - 1);     // 利用反函数作用阈值，利用单调性筛选
    std::vector<std::vector<cv::Rect2d>> bboxes(CLASSES_NUM); // 每个id的xyhw 信息使用一个std::vector<cv::Rect2d>存储
    std::vector<std::vector<float>> scores(CLASSES_NUM);      // 每个id的score信息使用一个std::vector<float>存储
    
    begin_time = std::chrono::system_clock::now();
    // 7.1 小目标特征图
    // output[order[0]]: (1, H // 8,  W // 8,  4 * REG)
    // output[order[3]]: (1, H // 8,  W // 8,  CLASSES_NUM)
    // 7.1.1 检查反量化类型是否符合RDK Model Zoo的README导出的bin模型规范
    if (output[order[0]].properties.quantiType != SCALE)
    {
        std::cout << "output[order[0]] QuantiType is not SCALE, please check!" << std::endl;
        return -1;
    }
    if (output[order[3]].properties.quantiType != NONE)
    {
        std::cout << "output[order[3]] QuantiType is not NONE, please check!" << std::endl;
        return -1;
    }

    // 7.1.2 对缓存的BPU内存进行刷新
    hbSysFlushMem(&(output[order[0]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    hbSysFlushMem(&(output[order[3]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

    // 7.1.3 将BPU推理完的内存地址转换为对应类型的指针
    auto *s_bbox_raw = reinterpret_cast<int32_t *>(output[order[0]].sysMem[0].virAddr);
    auto *s_bbox_scale = reinterpret_cast<float *>(output[order[0]].properties.scale.scaleData);
    auto *s_cls_raw = reinterpret_cast<float *>(output[order[3]].sysMem[0].virAddr);
    for (int h = 0; h < H_8; h++)
    {
        for (int w = 0; w < W_8; w++)
        {
            // 7.1.4 取对应H和W位置的C通道, 记为数组的形式
            // cls对应CLASSES_NUM个分数RAW值, 也就是Sigmoid计算之前的值，这里利用函数单调性先筛选, 再计算
            // bbox对应4个坐标乘以REG的RAW值, 也就是DFL计算之前的值, 仅仅分数合格了, 才会进行这部分的计算
            float *cur_s_cls_raw = s_cls_raw;
            int32_t *cur_s_bbox_raw = s_bbox_raw;
            s_cls_raw += CLASSES_NUM;
            s_bbox_raw += REG * 4;

            // 7.1.5 找到分数的最大值索引, 如果最大值小于阈值，则舍去
            int cls_id = 0;
            for (int i = 1; i < CLASSES_NUM; i++)
            {
                if (cur_s_cls_raw[i] > cur_s_cls_raw[cls_id])
                {
                    cls_id = i;
                }
            }

            // 7.1.6 不合格则直接跳过, 避免无用的反量化, DFL和dist2bbox计算
            if (cur_s_cls_raw[cls_id] < CONF_THRES_RAW)
            {
                continue;
            }

            // 7.1.7 计算这个目标的分数
            float score = 1 / (1 + std::exp(-cur_s_cls_raw[cls_id]));

            // 7.1.8 对bbox_raw信息进行反量化, DFL计算
            float ltrb[4], sum, dfl;
            for (int i = 0; i < 4; i++)
            {
                ltrb[i] = 0.;
                sum = 0.;
                for (int j = 0; j < REG; j++)
                {
                    dfl = std::exp(float(cur_s_bbox_raw[REG * i + j]) * s_bbox_scale[j]);
                    ltrb[i] += dfl * j;
                    sum += dfl;
                }
                ltrb[i] /= sum;
            }

            // 7.1.9 剔除不合格的框   if(x1 >= x2 || y1 >=y2) continue;
            if (ltrb[2] <= ltrb[0] || ltrb[3] <= ltrb[1])
                continue;

            // 7.1.10 dist 2 bbox (ltrb 2 xyxy)
            float x1 = (w + 0.5 - ltrb[0]) * 8.0;
            float y1 = (h + 0.5 - ltrb[1]) * 8.0;
            float x2 = (w + 0.5 + ltrb[2]) * 8.0;
            float y2 = (h + 0.5 + ltrb[3]) * 8.0;

            // 7.1.11 对应类别加入到对应的std::vector中
            bboxes[cls_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
            scores[cls_id].push_back(score);
        }
    }

    // 7.2 中目标特征图
    // output[order[1]]: (1, H // 16,  W // 16,  4 * REG)
    // output[order[4]]: (1, H // 16,  W // 16,  CLASSES_NUM)
    // 7.2.1 检查反量化类型是否符合RDK Model Zoo的README导出的bin模型规范
    if (output[order[1]].properties.quantiType != SCALE)
    {
        std::cout << "output[order[0]] QuantiType is not SCALE, please check!" << std::endl;
        return -1;
    }
    if (output[order[4]].properties.quantiType != NONE)
    {
        std::cout << "output[order[3]] QuantiType is not NONE, please check!" << std::endl;
        return -1;
    }
    // 7.2.2 对缓存的BPU内存进行刷新
    hbSysFlushMem(&(output[order[1]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    hbSysFlushMem(&(output[order[4]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    // 7.2.3 将BPU推理完的内存地址转换为对应类型的指针
    auto *m_bbox_raw = reinterpret_cast<int32_t *>(output[order[1]].sysMem[0].virAddr);
    auto *m_bbox_scale = reinterpret_cast<float *>(output[order[1]].properties.scale.scaleData);
    auto *m_cls_raw = reinterpret_cast<float *>(output[order[4]].sysMem[0].virAddr);
    for (int h = 0; h < H_16; h++)
    {
        for (int w = 0; w < W_16; w++)
        {
            // 7.2.4 取对应H和W位置的C通道, 记为数组的形式
            // cls对应CLASSES_NUM个分数RAW值, 也就是Sigmoid计算之前的值，这里利用函数单调性先筛选, 再计算
            // bbox对应4个坐标乘以REG的RAW值, 也就是DFL计算之前的值, 仅仅分数合格了, 才会进行这部分的计算
            float *cur_m_cls_raw = m_cls_raw;
            int32_t *cur_m_bbox_raw = m_bbox_raw;
            m_cls_raw += CLASSES_NUM;
            m_bbox_raw += REG * 4;

            // 7.2.5 找到分数的最大值索引, 如果最大值小于阈值，则舍去
            int cls_id = 0;
            for (int i = 1; i < CLASSES_NUM; i++)
            {
                if (cur_m_cls_raw[i] > cur_m_cls_raw[cls_id])
                {
                    cls_id = i;
                }
            }
            // 7.2.6 不合格则直接跳过, 避免无用的反量化, DFL和dist2bbox计算
            if (cur_m_cls_raw[cls_id] < CONF_THRES_RAW)
                continue;

            // 7.2.7 计算这个目标的分数
            float score = 1 / (1 + std::exp(-cur_m_cls_raw[cls_id]));

            // 7.2.8 对bbox_raw信息进行反量化, DFL计算
            float ltrb[4], sum, dfl;
            for (int i = 0; i < 4; i++)
            {
                ltrb[i] = 0.;
                sum = 0.;
                for (int j = 0; j < REG; j++)
                {
                    dfl = std::exp(float(cur_m_bbox_raw[REG * i + j]) * s_bbox_scale[j]);
                    ltrb[i] += dfl * j;
                    sum += dfl;
                }
                ltrb[i] /= sum;
            }

            // 7.2.9 剔除不合格的框   if(x1 >= x2 || y1 >=y2) continue;
            if (ltrb[2] <= ltrb[0] || ltrb[3] <= ltrb[1])
                continue;

            // 7.2.10 dist 2 bbox (ltrb 2 xyxy)
            float x1 = (w + 0.5 - ltrb[0]) * 16.0;
            float y1 = (h + 0.5 - ltrb[1]) * 16.0;
            float x2 = (w + 0.5 + ltrb[2]) * 16.0;
            float y2 = (h + 0.5 + ltrb[3]) * 16.0;

            // 7.2.11 对应类别加入到对应的std::vector中
            bboxes[cls_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
            scores[cls_id].push_back(score);
        }
    }

    // 7.3 大目标特征图
    // output[order[2]]: (1, H // 32,  W // 32,  4 * REG)
    // output[order[5]]: (1, H // 32,  W // 32,  CLASSES_NUM)
    // 7.3.1 检查反量化类型是否符合RDK Model Zoo的README导出的bin模型规范
    if (output[order[2]].properties.quantiType != SCALE)
    {
        std::cout << "output[order[0]] QuantiType is not SCALE, please check!" << std::endl;
        return -1;
    }
    if (output[order[5]].properties.quantiType != NONE)
    {
        std::cout << "output[order[3]] QuantiType is not NONE, please check!" << std::endl;
        return -1;
    }
    // 7.3.2 对缓存的BPU内存进行刷新
    hbSysFlushMem(&(output[order[2]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    hbSysFlushMem(&(output[order[5]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    // 7.3.3 将BPU推理完的内存地址转换为对应类型的指针
    auto *l_bbox_raw = reinterpret_cast<int32_t *>(output[order[2]].sysMem[0].virAddr);
    auto *l_bbox_scale = reinterpret_cast<float *>(output[order[2]].properties.scale.scaleData);
    auto *l_cls_raw = reinterpret_cast<float *>(output[order[5]].sysMem[0].virAddr);
    for (int h = 0; h < H_32; h++)
    {
        for (int w = 0; w < W_32; w++)
        {
            // 7.3.4 取对应H和W位置的C通道, 记为数组的形式
            // cls对应CLASSES_NUM个分数RAW值, 也就是Sigmoid计算之前的值，这里利用函数单调性先筛选, 再计算
            // bbox对应4个坐标乘以REG的RAW值, 也就是DFL计算之前的值, 仅仅分数合格了, 才会进行这部分的计算
            float *cur_l_cls_raw = l_cls_raw;
            int32_t *cur_l_bbox_raw = l_bbox_raw;
            l_cls_raw += CLASSES_NUM;
            l_bbox_raw += REG * 4;

            // 7.3.5 找到分数的最大值索引, 如果最大值小于阈值，则舍去
            int cls_id = 0;
            for (int i = 1; i < CLASSES_NUM; i++)
            {
                if (cur_l_cls_raw[i] > cur_l_cls_raw[cls_id])
                {
                    cls_id = i;
                }
            }
            // 7.3.6 不合格则直接跳过, 避免无用的反量化, DFL和dist2bbox计算
            if (cur_l_cls_raw[cls_id] < CONF_THRES_RAW)
                continue;

            // 7.3.7 计算这个目标的分数
            float score = 1 / (1 + std::exp(-cur_l_cls_raw[cls_id]));

            // 7.3.8 对bbox_raw信息进行反量化, DFL计算
            float ltrb[4], sum, dfl;
            for (int i = 0; i < 4; i++)
            {
                ltrb[i] = 0.;
                sum = 0.;
                for (int j = 0; j < REG; j++)
                {
                    dfl = std::exp(float(cur_l_bbox_raw[REG * i + j]) * s_bbox_scale[j]);
                    ltrb[i] += dfl * j;
                    sum += dfl;
                }
                ltrb[i] /= sum;
            }

            // 7.3.9 剔除不合格的框   if(x1 >= x2 || y1 >=y2) continue;
            if (ltrb[2] <= ltrb[0] || ltrb[3] <= ltrb[1])
                continue;

            // 7.3.10 dist 2 bbox (ltrb 2 xyxy)
            float x1 = (w + 0.5 - ltrb[0]) * 32.0;
            float y1 = (h + 0.5 - ltrb[1]) * 32.0;
            float x2 = (w + 0.5 + ltrb[2]) * 32.0;
            float y2 = (h + 0.5 + ltrb[3]) * 32.0;

            // 7.3.11 对应类别加入到对应的std::vector中
            bboxes[cls_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
            scores[cls_id].push_back(score);
        }
    }

    // 7.4 对每一个类别进行NMS
    std::vector<std::vector<int>> indices(CLASSES_NUM);
    for (int i = 0; i < CLASSES_NUM; i++)
    {
        cv::dnn::NMSBoxes(bboxes[i], scores[i], SCORE_THRESHOLD, NMS_THRESHOLD, indices[i], 1.f, NMS_TOP_K);
    }
    std::cout << "\033[31m Post Process time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count()/1000.0 << " ms\033[0m" << std::endl;


    // 8. 渲染
    begin_time = std::chrono::system_clock::now();
    for (int cls_id = 0; cls_id < CLASSES_NUM; cls_id++)
    {
        // 每一个类别分别渲染
        for (std::vector<int>::iterator it = indices[cls_id].begin(); it != indices[cls_id].end(); ++it)
        {
            // 获取基本的 bbox 信息
            float x1 = x_scale * (bboxes[cls_id][*it].x);
            float y1 = y_scale * (bboxes[cls_id][*it].y);
            float x2 = x1 + x_scale * (bboxes[cls_id][*it].width);
            float y2 = y1 + y_scale * (bboxes[cls_id][*it].height);
            float score = scores[cls_id][*it];
            std::string name = object_names[cls_id % CLASSES_NUM];
            cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), LINE_SIZE);

            // 绘制字体
            std::string text = name + ": " + std::to_string(static_cast<int>(score * 100)) + "%";
            cv::putText(img, text, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, FONT_SIZE, cv::Scalar(0, 0, 255), FONT_THINKNESS, cv::LINE_AA);

            // 打印检测信息
            std::cout << "(" << x1 << " " << y1 << " " << x2 << " " << y2 << "): \t" << text << std::endl;
        }
    }
    std::cout << "\033[31m Draw Result time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count()/1000.0 << " ms\033[0m" << std::endl;


    // 9. 保存
    cv::imwrite("cpp_result.jpg", img);

    // 10. 释放任务
    hbDNNReleaseTask(task_handle);

    // 11. 释放内存
    hbSysFreeMem(&(input.sysMem[0]));
    hbSysFreeMem(&(output->sysMem[0]));

    // 12. 释放模型
    hbDNNRelease(packed_dnn_handle);

    return 0;
}