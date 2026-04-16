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
#define MODEL_PATH "source/reference_bin_models/yolo13n_detect_bayese_640x640_nv12.bin"

// 测试图片路径
// Path to test image
#define TEST_IMG_PATH "../../../../datasets/coco/assets/bus.jpg"

// 结果保存路径
// Path to save result
#define IMG_SAVE_PATH "cpp_result.jpg"

// 前处理方式: 0=Resize, 1=LetterBox
// Preprocessing method: 0=Resize, 1=LetterBox
#define RESIZE_TYPE 0
#define LETTERBOX_TYPE 1
#define PREPROCESS_TYPE LETTERBOX_TYPE

// 模型参数 / Model Parameters
#define CLASSES_NUM 80       // 类别数量 / Number of classes
#define NMS_THRESHOLD 0.7    // NMS阈值 / NMS threshold
#define SCORE_THRESHOLD 0.25 // 分数阈值 / Score threshold
#define NMS_TOP_K 300        // NMS最多保留框数 / Max boxes for NMS
#define REG 16               // DFL 回归参数 / DFL regression parameter

// 可视化参数 / Visualization Parameters
#define FONT_SCALE 0.6
#define FONT_THICKNESS 2
#define BOX_THICKNESS 2

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
#include <opencv2/dnn/dnn.hpp>

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
// COCO Class Names
// ============================================================================

const std::vector<std::string> COCO_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

// ============================================================================
// Detection Result Structure
// ============================================================================

struct Detection {
    int class_id;
    float confidence;
    cv::Rect2d bbox;  // x, y, w, h (use double for OpenCV NMSBoxes compatibility)

    Detection(int id, float conf, const cv::Rect2d& box)
        : class_id(id), confidence(conf), bbox(box) {}
};

// ============================================================================
// Color Palette for Visualization
// ============================================================================

const std::vector<cv::Scalar> COLORS = {
    cv::Scalar(56, 56, 255),    cv::Scalar(151, 157, 255),
    cv::Scalar(31, 112, 255),   cv::Scalar(29, 178, 255),
    cv::Scalar(49, 210, 207),   cv::Scalar(10, 249, 72),
    cv::Scalar(23, 204, 146),   cv::Scalar(134, 219, 61),
    cv::Scalar(52, 147, 26),    cv::Scalar(187, 212, 0),
    cv::Scalar(168, 153, 44),   cv::Scalar(255, 194, 0),
    cv::Scalar(147, 69, 52),    cv::Scalar(255, 115, 100),
    cv::Scalar(236, 24, 0),     cv::Scalar(255, 56, 132),
    cv::Scalar(133, 0, 82),     cv::Scalar(255, 56, 203),
    cv::Scalar(200, 149, 255),  cv::Scalar(199, 55, 255)
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Convert BGR image to NV12 format
 * @param bgr_img Input BGR image
 * @return NV12 formatted image (height * 3/2, width, CV_8UC1)
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
 * @param img Input image
 * @param input_h Target height
 * @param input_w Target width
 * @param x_scale Output: x scale factor
 * @param y_scale Output: y scale factor
 * @param x_shift Output: x shift
 * @param y_shift Output: y shift
 * @return Preprocessed image
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
 * @brief Draw detection results on image
 * @param img Image to draw on
 * @param detections Detection results
 * @param x_scale X scale factor for coordinate transformation
 * @param y_scale Y scale factor for coordinate transformation
 * @param x_shift X shift for coordinate transformation
 * @param y_shift Y shift for coordinate transformation
 */
void draw_detections(cv::Mat& img, const std::vector<Detection>& detections,
                    float x_scale, float y_scale, int x_shift, int y_shift) {
    auto start = std::chrono::high_resolution_clock::now();

    for (const auto& det : detections) {
        // Transform coordinates back to original image space
        float x1 = (det.bbox.x - x_shift) / x_scale;
        float y1 = (det.bbox.y - y_shift) / y_scale;
        float x2 = x1 + det.bbox.width / x_scale;
        float y2 = y1 + det.bbox.height / y_scale;

        // Clamp coordinates
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(img.cols)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(img.rows)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(img.cols)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(img.rows)));

        // Get color
        cv::Scalar color = COLORS[det.class_id % COLORS.size()];

        // Draw box
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2),
                     color, BOX_THICKNESS);

        // Prepare label
        std::string label = COCO_NAMES[det.class_id] + ": " +
                           std::to_string(static_cast<int>(det.confidence * 100)) + "%";

        // Get text size
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                             FONT_SCALE, FONT_THICKNESS, &baseline);

        // Draw label background
        int label_y = std::max(static_cast<int>(y1), text_size.height + 10);
        cv::rectangle(img,
                     cv::Point(x1, label_y - text_size.height - 10),
                     cv::Point(x1 + text_size.width, label_y),
                     color, cv::FILLED);

        // Draw label text
        cv::putText(img, label, cv::Point(x1, label_y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                   cv::Scalar(255, 255, 255), FONT_THICKNESS, cv::LINE_AA);

        // Print detection info
        std::cout << "  (" << static_cast<int>(x1) << ", " << static_cast<int>(y1)
                  << ", " << static_cast<int>(x2) << ", " << static_cast<int>(y2)
                  << ") -> " << label << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    LOG_TIME("Draw results time", duration);
}

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char** argv) {
    LOG_INFO("=== Ultralytics YOLO Detect Demo (C++) ===");
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

    if (model_count > 1) {
        LOG_WARN("Model file contains " << model_count << " models, using the first one");
    }

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

    if (output_count != 6) {
        LOG_ERROR("Model should have 6 outputs, but has " << output_count);
        return -1;
    }

    LOG_INFO("Model outputs:");
    for (int i = 0; i < output_count; i++) {
        hbDNNTensorProperties output_properties;
        CHECK_SUCCESS(
            hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i),
            "Failed to get output tensor properties");

        std::cout << "  output[" << i << "] shape: ("
                  << output_properties.validShape.dimensionSize[0] << ", "
                  << output_properties.validShape.dimensionSize[1] << ", "
                  << output_properties.validShape.dimensionSize[2] << ", "
                  << output_properties.validShape.dimensionSize[3] << ")";

        if (output_properties.quantiType == SHIFT) std::cout << ", SHIFT";
        else if (output_properties.quantiType == SCALE) std::cout << ", SCALE";
        else if (output_properties.quantiType == NONE) std::cout << ", NONE";
        std::cout << std::endl;
    }

    // Map output order
    int32_t H_8 = input_h / 8;
    int32_t W_8 = input_w / 8;
    int32_t H_16 = input_h / 16;
    int32_t W_16 = input_w / 16;
    int32_t H_32 = input_h / 32;
    int32_t W_32 = input_w / 32;

    int order[6] = {0, 1, 2, 3, 4, 5};
    int32_t expected_shapes[6][3] = {
        {H_8, W_8, CLASSES_NUM},
        {H_8, W_8, 64},
        {H_16, W_16, CLASSES_NUM},
        {H_16, W_16, 64},
        {H_32, W_32, CLASSES_NUM},
        {H_32, W_32, 64}
    };

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            hbDNNTensorProperties props;
            hbDNNGetOutputTensorProperties(&props, dnn_handle, j);

            int32_t h = props.validShape.dimensionSize[1];
            int32_t w = props.validShape.dimensionSize[2];
            int32_t c = props.validShape.dimensionSize[3];

            if (h == expected_shapes[i][0] &&
                w == expected_shapes[i][1] &&
                c == expected_shapes[i][2]) {
                order[i] = j;
                break;
            }
        }
    }

    LOG_INFO("Output order mapping: [" << order[0] << ", " << order[1] << ", "
             << order[2] << ", " << order[3] << ", " << order[4] << ", "
             << order[5] << "]");

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
        hbDNNGetOutputTensorProperties(&output[i].properties, dnn_handle, i);
        int out_size = output[i].properties.alignedByteSize;
        hbSysAllocCachedMem(&output[i].sysMem[0], out_size);
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

    std::vector<std::vector<cv::Rect2d>> bboxes(CLASSES_NUM);
    std::vector<std::vector<float>> scores(CLASSES_NUM);

    // Process 3 scales: 8, 16, 32
    const int strides[3] = {8, 16, 32};
    const int heights[3] = {H_8, H_16, H_32};
    const int widths[3] = {W_8, W_16, W_32};

    for (int scale_idx = 0; scale_idx < 3; scale_idx++) {
        int cls_output_idx = order[scale_idx * 2];
        int bbox_output_idx = order[scale_idx * 2 + 1];
        int stride = strides[scale_idx];
        int h = heights[scale_idx];
        int w = widths[scale_idx];

        // Flush memory
        hbSysFlushMem(&output[cls_output_idx].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
        hbSysFlushMem(&output[bbox_output_idx].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);

        // Get pointers
        float* cls_raw = reinterpret_cast<float*>(output[cls_output_idx].sysMem[0].virAddr);
        float* bbox_raw = reinterpret_cast<float*>(output[bbox_output_idx].sysMem[0].virAddr);

        // Process each grid cell
        for (int gh = 0; gh < h; gh++) {
            for (int gw = 0; gw < w; gw++) {
                float* cur_cls = cls_raw;
                float* cur_bbox = bbox_raw;

                // Find max class score
                int cls_id = 0;
                for (int c = 1; c < CLASSES_NUM; c++) {
                    if (cur_cls[c] > cur_cls[cls_id]) {
                        cls_id = c;
                    }
                }

                // Skip if score is too low
                if (cur_cls[cls_id] < CONF_THRES_RAW) {
                    cls_raw += CLASSES_NUM;
                    bbox_raw += REG * 4;
                    continue;
                }

                // Compute score
                float score = 1.0f / (1.0f + std::exp(-cur_cls[cls_id]));

                // DFL decode (softmax + weighted sum)
                float ltrb[4] = {0};
                for (int i = 0; i < 4; i++) {
                    float sum = 0.0f;
                    for (int j = 0; j < REG; j++) {
                        int idx = REG * i + j;
                        float dfl = std::exp(cur_bbox[idx]);
                        ltrb[i] += dfl * j;
                        sum += dfl;
                    }
                    ltrb[i] /= sum;
                }

                // Skip invalid boxes
                if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0) {
                    cls_raw += CLASSES_NUM;
                    bbox_raw += REG * 4;
                    continue;
                }

                // Convert to xyxy
                float x1 = (gw + 0.5f - ltrb[0]) * stride;
                float y1 = (gh + 0.5f - ltrb[1]) * stride;
                float x2 = (gw + 0.5f + ltrb[2]) * stride;
                float y2 = (gh + 0.5f + ltrb[3]) * stride;

                // Store result
                bboxes[cls_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
                scores[cls_id].push_back(score);

                cls_raw += CLASSES_NUM;
                bbox_raw += REG * 4;
            }
        }
    }

    // NMS for each class
    std::vector<Detection> final_detections;
    for (int cls_id = 0; cls_id < CLASSES_NUM; cls_id++) {
        if (bboxes[cls_id].empty()) continue;

        std::vector<int> indices;
        cv::dnn::NMSBoxes(bboxes[cls_id], scores[cls_id],
                         SCORE_THRESHOLD, NMS_THRESHOLD,
                         indices, 1.0f, NMS_TOP_K);

        for (int idx : indices) {
            final_detections.emplace_back(cls_id, scores[cls_id][idx], bboxes[cls_id][idx]);
        }
    }

    auto post_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start_time).count() / 1000.0;
    LOG_TIME("Post-processing time", post_duration);

    LOG_INFO("Detected " << final_detections.size() << " objects");

    // ========================================================================
    // 10. Draw results
    // ========================================================================

    if (!final_detections.empty()) {
        LOG_INFO("Detection results:");
        draw_detections(img, final_detections, x_scale, y_scale, x_shift, y_shift);
    }

    // Save result
    cv::imwrite(save_path, img);
    LOG_INFO("Result saved to: " << save_path);

    // ========================================================================
    // 11. Cleanup
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
