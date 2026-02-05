/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

Copyright (c) 2024-2025, WuChao && MaChao D-Robotics.

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

// 注意: 此程序在RDK板端运行
// Attention: This program runs on RDK board.

// ============================================================================
// Configuration Parameters
// ============================================================================

// D-Robotics *.bin 模型路径
// Path to D-Robotics *.bin model
#define MODEL_PATH "source/reference_bin_models/seg/yolo11n_seg_bayese_640x640_nv12.bin"

// 测试图片路径
// Path to test image
#define TEST_IMG_PATH "../../../../datasets/coco/assets/bus.jpg"

// 前处理方式: 0=Resize, 1=LetterBox
// Preprocessing method: 0=Resize, 1=LetterBox
#define RESIZE_TYPE 0
#define LETTERBOX_TYPE 1
#define PREPROCESS_TYPE LETTERBOX_TYPE

// 推理结果保存路径
// Path where the inference result will be saved
#define IMG_SAVE_PATH "segment_result.jpg"

// 模型的类别数量
// Number of classes in the model
#define CLASSES_NUM 80

// NMS的阈值
// Non-Maximum Suppression (NMS) threshold
#define NMS_THRESHOLD 0.45

// 分数阈值
// Score threshold
#define SCORE_THRESHOLD 0.25

// 控制回归部分离散化程度的超参数, DFL
// A hyperparameter that controls the discretization level of the regression part
#define REG 16

// 掩码系数数量
// Mask Coefficients
#define MCES 32

// 掩码阈值
// Mask threshold
#define MASK_THRESHOLD 0.5

// ============================================================================
// Includes
// ============================================================================

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>

// OpenCV
#include <opencv2/opencv.hpp>

// RDK BPU libDNN API
#include "dnn/hb_dnn.h"
#include "dnn/hb_dnn_ext.h"
#include "dnn/plugin/hb_dnn_layer.h"
#include "dnn/plugin/hb_dnn_plugin.h"
#include "dnn/hb_sys.h"

// ============================================================================
// Macros
// ============================================================================

#define CHECK_SUCCESS(value, errmsg)                                         \
    do {                                                                     \
        auto ret_code = value;                                               \
        if (ret_code != 0) {                                                 \
            std::cerr << "\033[1;31m[ERROR]\033[0m " << __FILE__ << ":"     \
                      << __LINE__ << " " << errmsg                           \
                      << ", error code: " << ret_code << std::endl;          \
            return ret_code;                                                 \
        }                                                                    \
    } while (0)

#define LOG_INFO(msg) \
    std::cout << "\033[1;32m[INFO]\033[0m " << msg << std::endl

#define LOG_WARN(msg) \
    std::cout << "\033[1;33m[WARN]\033[0m " << msg << std::endl

#define LOG_ERROR(msg) \
    std::cerr << "\033[1;31m[ERROR]\033[0m " << msg << std::endl

#define LOG_TIME(msg, duration) \
    std::cout << "\033[1;31m" << msg << " = " << std::fixed            \
              << std::setprecision(2) << (duration) << " ms\033[0m"    \
              << std::endl

// ============================================================================
// COCO Names and Colors
// ============================================================================

const std::vector<std::string> COCO_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

const std::vector<cv::Scalar> RDK_COLORS = {
    cv::Scalar(56, 56, 255), cv::Scalar(151, 157, 255), cv::Scalar(31, 112, 255), cv::Scalar(29, 178, 255),
    cv::Scalar(49, 210, 207), cv::Scalar(10, 249, 72), cv::Scalar(23, 204, 146), cv::Scalar(134, 219, 61),
    cv::Scalar(52, 147, 26), cv::Scalar(187, 212, 0), cv::Scalar(168, 153, 44), cv::Scalar(255, 194, 0),
    cv::Scalar(147, 69, 52), cv::Scalar(255, 115, 100), cv::Scalar(236, 24, 0), cv::Scalar(255, 56, 132),
    cv::Scalar(133, 0, 82), cv::Scalar(255, 56, 203), cv::Scalar(200, 149, 255), cv::Scalar(199, 55, 255)
};

// ============================================================================
// Detection Result Structure
// ============================================================================

struct Detection {
    cv::Rect2d bbox;           // Bounding box
    float score;               // Confidence score
    int class_id;              // Class ID
    std::vector<float> mask_coeffs;  // Mask coefficients (32)
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Convert BGR image to NV12 format
 */
cv::Mat bgr2nv12(const cv::Mat& bgr_img) {
    auto start = std::chrono::high_resolution_clock::now();

    int height = bgr_img.rows;
    int width = bgr_img.cols;

    // BGR to YUV420P
    cv::Mat yuv_mat;
    cv::cvtColor(bgr_img, yuv_mat, cv::COLOR_BGR2YUV_I420);
    uint8_t* yuv = yuv_mat.ptr<uint8_t>();

    // Allocate NV12 image
    cv::Mat nv12_img(height * 3 / 2, width, CV_8UC1);
    uint8_t* nv12 = nv12_img.ptr<uint8_t>();

    // Copy Y plane
    int y_size = height * width;
    memcpy(nv12, yuv, y_size);

    // Convert UV planar to UV packed (NV12)
    int uv_height = height / 2;
    int uv_width = width / 2;
    uint8_t* nv12_uv = nv12 + y_size;
    uint8_t* u_data = yuv + y_size;
    uint8_t* v_data = u_data + uv_height * uv_width;

    for (int i = 0; i < uv_width * uv_height; i++) {
        *nv12_uv++ = *u_data++;
        *nv12_uv++ = *v_data++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    LOG_TIME("BGR to NV12 time", duration);

    return nv12_img;
}

/**
 * @brief Preprocess image with letterbox or resize
 */
cv::Mat preprocess_image(const cv::Mat& img, int input_h, int input_w,
                         float& x_scale, float& y_scale,
                         int& x_shift, int& y_shift) {
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat result;

    if (PREPROCESS_TYPE == LETTERBOX_TYPE) {
        // Letterbox preprocessing
        x_scale = std::min(1.0f * input_h / img.rows, 1.0f * input_w / img.cols);
        y_scale = x_scale;

        if (x_scale <= 0 || y_scale <= 0) {
            throw std::runtime_error("Invalid scale factor");
        }

        int new_w = static_cast<int>(img.cols * x_scale);
        int new_h = static_cast<int>(img.rows * y_scale);

        x_shift = (input_w - new_w) / 2;
        y_shift = (input_h - new_h) / 2;
        int x_other = input_w - new_w - x_shift;
        int y_other = input_h - new_h - y_shift;

        cv::resize(img, result, cv::Size(new_w, new_h));
        cv::copyMakeBorder(result, result, y_shift, y_other, x_shift, x_other,
                          cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        LOG_TIME("Preprocess (LetterBox) time", duration);

    } else if (PREPROCESS_TYPE == RESIZE_TYPE) {
        // Resize preprocessing
        cv::resize(img, result, cv::Size(input_w, input_h));

        x_scale = 1.0f * input_w / img.cols;
        y_scale = 1.0f * input_h / img.rows;
        x_shift = 0;
        y_shift = 0;

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        LOG_TIME("Preprocess (Resize) time", duration);
    }

    LOG_INFO("Scale: x=" << x_scale << ", y=" << y_scale);
    LOG_INFO("Shift: x=" << x_shift << ", y=" << y_shift);

    return result;
}

/**
 * @brief Softmax function for DFL
 */
void softmax(float* input, float* output, int length) {
    float max_val = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }

    for (int i = 0; i < length; i++) {
        output[i] /= sum;
    }
}

/**
 * @brief Draw detection results on image
 */
void draw_detection(cv::Mat& img, const Detection& det) {
    int x1 = static_cast<int>(det.bbox.x);
    int y1 = static_cast<int>(det.bbox.y);
    int x2 = static_cast<int>(det.bbox.x + det.bbox.width);
    int y2 = static_cast<int>(det.bbox.y + det.bbox.height);

    cv::Scalar color = RDK_COLORS[det.class_id % RDK_COLORS.size()];
    cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

    std::string label = COCO_NAMES[det.class_id] + ": " +
                       std::to_string(det.score).substr(0, 4);

    int baseline;
    cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

    int label_y = std::max(y1, label_size.height);
    cv::rectangle(img, cv::Point(x1, label_y - label_size.height),
                 cv::Point(x1 + label_size.width, label_y + baseline),
                 color, cv::FILLED);
    cv::putText(img, label, cv::Point(x1, label_y),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
}

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char** argv) {
    LOG_INFO("=== Ultralytics YOLO Segment Demo (C++) ===");
    LOG_INFO("OpenCV Version: " << CV_VERSION);

    // ========================================================================
    // 0. Parse command line arguments
    // ========================================================================

    std::string model_path = MODEL_PATH;
    std::string test_img_path = TEST_IMG_PATH;
    std::string save_path = IMG_SAVE_PATH;

    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) test_img_path = argv[2];
    if (argc >= 4) save_path = argv[3];

    // ========================================================================
    // 1. Load BPU model
    // ========================================================================

    LOG_INFO("Loading model: " << model_path);
    auto start_time = std::chrono::high_resolution_clock::now();

    hbPackedDNNHandle_t packed_dnn_handle;
    const char* model_file_name = model_path.c_str();
    CHECK_SUCCESS(
        hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1),
        "Failed to initialize model from file");

    auto load_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start_time).count() / 1000.0;
    LOG_TIME("Load model time", load_duration);

    // ========================================================================
    // 2. Get model handle
    // ========================================================================

    const char** model_name_list;
    int model_count = 0;
    CHECK_SUCCESS(
        hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle),
        "Failed to get model name list");

    const char* model_name = model_name_list[0];
    LOG_INFO("Model name: " << model_name);

    hbDNNHandle_t dnn_handle;
    CHECK_SUCCESS(
        hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name),
        "Failed to get model handle");

    // ========================================================================
    // 3. Check model input
    // ========================================================================

    int32_t input_count = 0;
    CHECK_SUCCESS(
        hbDNNGetInputCount(&input_count, dnn_handle),
        "Failed to get input count");

    if (input_count != 1) {
        LOG_ERROR("Model should have exactly 1 input, but has " << input_count);
        return -1;
    }

    hbDNNTensorProperties input_properties;
    CHECK_SUCCESS(
        hbDNNGetInputTensorProperties(&input_properties, dnn_handle, 0),
        "Failed to get input tensor properties");

    // Check tensor type
    if (input_properties.tensorType != HB_DNN_IMG_TYPE_NV12) {
        LOG_ERROR("Input tensor type is not HB_DNN_IMG_TYPE_NV12");
        return -1;
    }
    LOG_INFO("Input tensor type: HB_DNN_IMG_TYPE_NV12");

    // Check tensor layout
    if (input_properties.tensorLayout != HB_DNN_LAYOUT_NCHW) {
        LOG_ERROR("Input tensor layout is not HB_DNN_LAYOUT_NCHW");
        return -1;
    }
    LOG_INFO("Input tensor layout: HB_DNN_LAYOUT_NCHW");

    // Get input shape
    if (input_properties.validShape.numDimensions != 4) {
        LOG_ERROR("Input tensor should have 4 dimensions");
        return -1;
    }

    int32_t input_h = input_properties.validShape.dimensionSize[2];
    int32_t input_w = input_properties.validShape.dimensionSize[3];
    LOG_INFO("Input shape: (1, 3, " << input_h << ", " << input_w << ")");

    // ========================================================================
    // 4. Check model outputs
    // ========================================================================

    int32_t output_count = 0;
    CHECK_SUCCESS(
        hbDNNGetOutputCount(&output_count, dnn_handle),
        "Failed to get output count");

    if (output_count != 10) {
        LOG_ERROR("Segment model should have exactly 10 outputs, but has " << output_count);
        return -1;
    }

    LOG_INFO("Model has 10 outputs (3 scales × 3 types + 1 proto)");

    // Print output shapes
    for (int i = 0; i < output_count; i++) {
        hbDNNTensorProperties output_properties;
        CHECK_SUCCESS(
            hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i),
            "Failed to get output tensor properties");

        std::cout << "output[" << i << "] shape: ("
                 << output_properties.validShape.dimensionSize[0] << ", "
                 << output_properties.validShape.dimensionSize[1] << ", "
                 << output_properties.validShape.dimensionSize[2] << ", "
                 << output_properties.validShape.dimensionSize[3] << "), ";

        if (output_properties.quantiType == SHIFT)
            std::cout << "SHIFT";
        else if (output_properties.quantiType == SCALE)
            std::cout << "SCALE";
        else if (output_properties.quantiType == NONE)
            std::cout << "NONE";
        std::cout << std::endl;
    }

    // ========================================================================
    // 5. Load and preprocess image
    // ========================================================================

    LOG_INFO("Loading image: " << test_img_path);
    cv::Mat img = cv::imread(test_img_path);
    if (img.empty()) {
        LOG_ERROR("Failed to load image: " << test_img_path);
        return -1;
    }
    LOG_INFO("Image size: " << img.cols << "x" << img.rows);

    // Preprocess image
    float x_scale, y_scale;
    int x_shift, y_shift;
    cv::Mat preprocessed = preprocess_image(img, input_h, input_w,
                                           x_scale, y_scale, x_shift, y_shift);

    // Convert to NV12
    cv::Mat nv12_img = bgr2nv12(preprocessed);

    // ========================================================================
    // 6. Prepare input tensor
    // ========================================================================

    hbDNNTensor input;
    input.properties = input_properties;

    int input_memSize = input_h * input_w * 3 / 2;
    hbSysAllocCachedMem(&input.sysMem[0], input_memSize);
    memcpy(input.sysMem[0].virAddr, nv12_img.ptr<uint8_t>(), input_memSize);
    hbSysFlushMem(&input.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);

    // ========================================================================
    // 7. Prepare output tensors
    // ========================================================================

    hbDNNTensor* output = new hbDNNTensor[output_count];
    for (int i = 0; i < output_count; i++) {
        hbDNNTensorProperties& output_properties = output[i].properties;
        hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i);
        int out_aligned_size = output_properties.alignedByteSize;
        hbSysAllocCachedMem(&output[i].sysMem[0], out_aligned_size);
    }

    // ========================================================================
    // 8. Run inference
    // ========================================================================

    LOG_INFO("Running inference...");
    start_time = std::chrono::high_resolution_clock::now();

    hbDNNTaskHandle_t task_handle = nullptr;
    hbDNNInferCtrlParam infer_ctrl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);

    hbDNNInfer(&task_handle, &output, &input, dnn_handle, &infer_ctrl_param);
    hbDNNWaitTaskDone(task_handle, 0);

    auto infer_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start_time).count() / 1000.0;
    LOG_TIME("BPU inference time", infer_duration);

    // ========================================================================
    // 9. Post-process
    // ========================================================================

    LOG_INFO("Post-processing...");
    start_time = std::chrono::high_resolution_clock::now();

    float CONF_THRES_RAW = -std::log(1.0f / SCORE_THRESHOLD - 1.0f);

    std::vector<Detection> detections;

    // Process proto mask first (output[9])
    int proto_h = input_h / 4;
    int proto_w = input_w / 4;

    hbSysFlushMem(&output[9].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    float* proto_data = reinterpret_cast<float*>(output[9].sysMem[0].virAddr);

    // Create proto matrix (proto_h*proto_w × MCES)
    cv::Mat proto_mat(proto_h * proto_w, MCES, CV_32F, proto_data);

    // Process 3 scales
    const int strides[3] = {8, 16, 32};
    const int grid_sizes[3] = {input_h / 8, input_h / 16, input_h / 32};

    for (int scale = 0; scale < 3; scale++) {
        int cls_idx = scale * 3 + 0;  // cls output index
        int box_idx = scale * 3 + 1;  // bbox output index
        int mce_idx = scale * 3 + 2;  // mask coefficients index

        int grid_h = grid_sizes[scale];
        int grid_w = grid_sizes[scale];
        float stride = strides[scale];

        // Flush memory
        hbSysFlushMem(&output[cls_idx].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
        hbSysFlushMem(&output[box_idx].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
        hbSysFlushMem(&output[mce_idx].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);

        // Get data pointers (all are float32 NONE type)
        float* cls_raw = reinterpret_cast<float*>(output[cls_idx].sysMem[0].virAddr);
        float* box_raw = reinterpret_cast<float*>(output[box_idx].sysMem[0].virAddr);
        float* mce_raw = reinterpret_cast<float*>(output[mce_idx].sysMem[0].virAddr);

        // Process each grid cell
        for (int h = 0; h < grid_h; h++) {
            for (int w = 0; w < grid_w; w++) {
                int offset = h * grid_w + w;

                float* cur_cls = cls_raw + offset * CLASSES_NUM;
                float* cur_box = box_raw + offset * (4 * REG);
                float* cur_mce = mce_raw + offset * MCES;

                // Find max class score
                int cls_id = 0;
                for (int i = 1; i < CLASSES_NUM; i++) {
                    if (cur_cls[i] > cur_cls[cls_id]) {
                        cls_id = i;
                    }
                }

                // Check threshold (before sigmoid)
                if (cur_cls[cls_id] < CONF_THRES_RAW) {
                    continue;
                }

                // Apply sigmoid to get confidence score
                float score = 1.0f / (1.0f + std::exp(-cur_cls[cls_id]));

                // Decode bbox using DFL (Distribution Focal Loss)
                float ltrb[4] = {0.0f};  // left, top, right, bottom

                for (int i = 0; i < 4; i++) {
                    float dfl_values[REG];
                    float dfl_softmax[REG];

                    for (int j = 0; j < REG; j++) {
                        dfl_values[j] = cur_box[i * REG + j];
                    }

                    softmax(dfl_values, dfl_softmax, REG);

                    for (int j = 0; j < REG; j++) {
                        ltrb[i] += dfl_softmax[j] * j;
                    }
                }

                // Convert to bbox coordinates
                float cx = (w + 0.5f) * stride;
                float cy = (h + 0.5f) * stride;
                float x1 = cx - ltrb[0] * stride;
                float y1 = cy - ltrb[1] * stride;
                float x2 = cx + ltrb[2] * stride;
                float y2 = cy + ltrb[3] * stride;

                // Check validity
                if (x1 >= 0 && y1 >= 0 && x2 > x1 && y2 > y1 &&
                    x2 <= input_w && y2 <= input_h) {

                    Detection det;
                    det.bbox = cv::Rect2d(x1, y1, x2 - x1, y2 - y1);
                    det.score = score;
                    det.class_id = cls_id;
                    det.mask_coeffs.resize(MCES);

                    // Copy mask coefficients
                    for (int i = 0; i < MCES; i++) {
                        det.mask_coeffs[i] = cur_mce[i];
                    }

                    detections.push_back(det);
                }
            }
        }
    }

    LOG_INFO("Detections before NMS: " << detections.size());

    // ========================================================================
    // 10. NMS (Non-Maximum Suppression)
    // ========================================================================

    std::vector<cv::Rect2d> nms_boxes;
    std::vector<float> nms_scores;
    std::vector<int> nms_indices;

    for (const auto& det : detections) {
        nms_boxes.push_back(det.bbox);
        nms_scores.push_back(det.score);
    }

    if (!nms_boxes.empty()) {
        cv::dnn::NMSBoxes(nms_boxes, nms_scores, SCORE_THRESHOLD, NMS_THRESHOLD, nms_indices);
    }

    LOG_INFO("Detections after NMS: " << nms_indices.size());

    auto post_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start_time).count() / 1000.0;
    LOG_TIME("Post-processing time", post_duration);

    // ========================================================================
    // 11. Generate masks and draw results
    // ========================================================================

    LOG_INFO("Generating masks and drawing results...");
    start_time = std::chrono::high_resolution_clock::now();

    cv::Mat img_display = preprocessed.clone();
    cv::Mat mask_overlay = cv::Mat::zeros(input_h, input_w, CV_8UC3);

    // Process each detection after NMS
    for (int idx : nms_indices) {
        const Detection& det = detections[idx];

        LOG_INFO("Detection: " << COCO_NAMES[det.class_id]
                 << ", score=" << std::fixed << std::setprecision(3) << det.score);

        // Draw bounding box
        draw_detection(img_display, det);

        // Generate instance mask
        // 1. Matrix multiplication: (proto_h*proto_w × MCES) × (MCES × 1)
        cv::Mat mce_mat(MCES, 1, CV_32F, (void*)det.mask_coeffs.data());
        cv::Mat mask_flat = proto_mat * mce_mat;
        cv::Mat mask_low_res = mask_flat.reshape(1, proto_h);

        // 2. Apply sigmoid
        cv::Mat sigmoid_mask;
        cv::exp(-mask_low_res, sigmoid_mask);
        sigmoid_mask = 1.0 / (1.0 + sigmoid_mask);

        // 3. Resize to input size
        cv::Mat resized_mask;
        cv::resize(sigmoid_mask, resized_mask, cv::Size(input_w, input_h), 0, 0, cv::INTER_LINEAR);

        // 4. Binarize mask
        cv::Mat binary_mask;
        cv::threshold(resized_mask, binary_mask, MASK_THRESHOLD, 1.0, cv::THRESH_BINARY);
        binary_mask.convertTo(binary_mask, CV_8U, 255);

        // 5. Crop mask to bbox region
        int x1 = std::max(0.0, det.bbox.x);
        int y1 = std::max(0.0, det.bbox.y);
        int x2 = std::min(static_cast<double>(input_w), det.bbox.x + det.bbox.width);
        int y2 = std::min(static_cast<double>(input_h), det.bbox.y + det.bbox.height);

        int mask_w = x2 - x1;
        int mask_h = y2 - y1;

        if (mask_w <= 0 || mask_h <= 0) continue;

        cv::Rect roi(x1, y1, mask_w, mask_h);
        cv::Mat roi_mask = binary_mask(roi);

        // 6. Create colored mask
        cv::Scalar color = RDK_COLORS[det.class_id % RDK_COLORS.size()];
        cv::Mat color_mask(mask_h, mask_w, CV_8UC3, color);

        // 7. Apply mask
        cv::Mat masked_color;
        cv::bitwise_and(color_mask, color_mask, masked_color, roi_mask);

        // 8. Add to overlay
        cv::Mat overlay_roi = mask_overlay(roi);
        cv::addWeighted(overlay_roi, 1.0, masked_color, 0.6, 0, overlay_roi);
    }

    // Create final result
    cv::Mat final_result;
    cv::addWeighted(img_display, 0.7, mask_overlay, 0.3, 0, final_result);

    // Concatenate: detection | mask | combined
    cv::Mat concatenated;
    cv::hconcat(img_display, mask_overlay, concatenated);
    cv::hconcat(concatenated, final_result, concatenated);

    // Output size: input_h × (input_w * 3)
    cv::imwrite(save_path, concatenated);
    LOG_INFO("Result saved to: " << save_path);
    LOG_INFO("Output size: " << concatenated.cols << "x" << concatenated.rows);

    auto draw_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start_time).count() / 1000.0;
    LOG_TIME("Mask generation and drawing time", draw_duration);

    // ========================================================================
    // 12. Cleanup
    // ========================================================================

    hbDNNReleaseTask(task_handle);
    hbSysFreeMem(&input.sysMem[0]);
    for (int i = 0; i < output_count; i++) {
        hbSysFreeMem(&output[i].sysMem[0]);
    }
    delete[] output;
    hbDNNRelease(packed_dnn_handle);

    LOG_INFO("=== Demo completed successfully ===");
    return 0;
}
