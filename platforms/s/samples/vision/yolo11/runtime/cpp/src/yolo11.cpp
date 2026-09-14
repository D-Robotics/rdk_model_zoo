/*
 * Copyright (c) 2025 D-Robotics Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file yolo11.cpp
 * @brief Implement the YOLO11 inference pipeline using HB-DNN / UCP runtime APIs.
 *
 * This file contains the complete implementation of the YOLO11 wrapper and its
 * end-to-end inference flow on D-Robotics platforms:
 * - Initialize and load a packed *.hbm model, query tensor properties,
 *   and allocate stride-aware tensor buffers.
 * - Preprocess input images (letterbox resize and BGR->NV12 conversion).
 * - Execute synchronous BPU inference via hbDNNInferV2() and UCP APIs.
 * - Postprocess raw DFL outputs: logit filtering, DFL box decoding,
 *   NMS, and coordinate scaling back to original image space.
 *
 * @see yolo11.hpp
 */

#include "yolo11.hpp"

// ---------------------------------------------------------------------------
// Internal helper
// ---------------------------------------------------------------------------

/**
 * @brief Filter class logits and decode DFL box distributions for one detection head.
 *
 * For each spatial position (h, w) in the feature map:
 * 1) Find argmax over C class logits from the classification tensor.
 * 2) Compare the max logit with conf_thres_raw; skip if below threshold.
 * 3) Decode 4 * reg logits from the box tensor via softmax + expectation (DFL).
 * 4) Convert (anchor_x, anchor_y, l, t, r, b) -> (x1, y1, x2, y2) at input scale.
 *
 * @param[in]  cls_tensor      Classification output tensor, shape (1, H, W, C), dtype float.
 * @param[in]  bbox_tensor     Box distribution tensor, shape (1, H, W, 4*reg), dtype S32 or float.
 * @param[in]  conf_thres_raw  Logit-domain confidence threshold (inverse sigmoid of score_thresh).
 * @param[in]  grid_size       Feature map spatial dimension (e.g., 80, 40, or 20).
 * @param[in]  stride          Input stride for this head (e.g., 8, 16, or 32).
 * @param[in]  reg             Number of DFL bins per bounding-box side (default: 16).
 * @param[out] detections      Decoded detections appended to this vector.
 */
static void filter_and_decode(
    const hbDNNTensor& cls_tensor,
    const hbDNNTensor& bbox_tensor,
    float conf_thres_raw,
    int grid_size,
    int stride,
    int reg,
    std::vector<Detection>& detections)
{
    const hbDNNTensorShape& shape = cls_tensor.properties.validShape;
    int H = shape.dimensionSize[1];
    int W = shape.dimensionSize[2];
    int C = shape.dimensionSize[3];

    const int64_t* stride_cls  = cls_tensor.properties.stride;
    const int64_t* stride_bbox = bbox_tensor.properties.stride;

    const uint8_t* data_cls  = reinterpret_cast<const uint8_t*>(cls_tensor.sysMem.virAddr);
    const uint8_t* data_bbox = reinterpret_cast<const uint8_t*>(bbox_tensor.sysMem.virAddr);

    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            size_t base_cls  = h * stride_cls[1]  + w * stride_cls[2];
            size_t base_bbox = h * stride_bbox[1] + w * stride_bbox[2];

            // Argmax over class logits
            float max_val = -1e30f;
            int   max_id  = 0;
            for (int c = 0; c < C; ++c) {
                const float* ptr = reinterpret_cast<const float*>(
                    data_cls + base_cls + c * stride_cls[3]);
                if (*ptr > max_val) { max_val = *ptr; max_id = c; }
            }

            // Logit-domain threshold check
            if (max_val < conf_thres_raw) continue;

            Detection det{};
            det.score    = sigmoid(max_val);
            det.class_id = max_id;

            // DFL decoding: 4 sides × reg bins
            float anchor_x = 0.5f + w;
            float anchor_y = 0.5f + h;
            float ltrb[4]  = {0.f, 0.f, 0.f, 0.f};

            for (int side = 0; side < 4; ++side) {
                float bins[16] = {};
                for (int bin = 0; bin < reg; ++bin) {
                    int ch = side * reg + bin;
                    const uint8_t* ptr = data_bbox + base_bbox + ch * stride_bbox[3];
                    if (bbox_tensor.properties.quantiType == SCALE) {
                        bins[bin] = dequant_value(*reinterpret_cast<const int32_t*>(ptr), ch, bbox_tensor.properties);
                    } else {
                        bins[bin] = *reinterpret_cast<const float*>(ptr);
                    }
                }
                // Softmax + expectation
                float probs[16] = {};
                softmax(bins, probs, reg);
                for (int i = 0; i < reg; ++i) {
                    ltrb[side] += probs[i] * i;
                }
            }

            // (anchor, ltrb) -> (x1, y1, x2, y2) in input-image scale
            det.bbox[0] = (anchor_x - ltrb[0]) * stride;
            det.bbox[1] = (anchor_y - ltrb[1]) * stride;
            det.bbox[2] = (anchor_x + ltrb[2]) * stride;
            det.bbox[3] = (anchor_y + ltrb[3]) * stride;

            detections.push_back(det);
        }
    }
}

// ---------------------------------------------------------------------------
// YOLO11 class
// ---------------------------------------------------------------------------

/**
 * @brief Construct a YOLO11 instance in an uninitialized state.
 */
YOLO11::YOLO11()
{
    model_count        = 0;
    packed_dnn_handle  = nullptr;
    dnn_handle         = nullptr;
    input_count        = 0;
    output_count       = 0;
    input_h            = 0;
    input_w            = 0;
    inited             = false;
}

/**
 * @brief Initialize model resources from a *.hbm model file.
 *
 * @param[in] model_path Path to the quantized *.hbm model file.
 * @retval 0        Success.
 * @retval non-zero DNN or UCP API error.
 */
int32_t YOLO11::init(const char* model_path)
{
    const char** model_name_list = nullptr;

    if (inited) {
        fprintf(stderr, "YOLO11::init() called twice\n");
        return -1;
    }

    // Load model from file
    HBDNN_CHECK_SUCCESS(hbDNNInitializeFromFiles(&packed_dnn_handle, &model_path, 1),
                        "hbDNNInitializeFromFiles failed");

    // Retrieve model name list and select the first model
    HBDNN_CHECK_SUCCESS(hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle),
                        "hbDNNGetModelNameList failed");
    HBDNN_CHECK_SUCCESS(hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]),
                        "hbDNNGetModelHandle failed");

    // Query input/output tensor counts
    HBDNN_CHECK_SUCCESS(hbDNNGetInputCount(&input_count, dnn_handle),
                        "hbDNNGetInputCount failed");
    HBDNN_CHECK_SUCCESS(hbDNNGetOutputCount(&output_count, dnn_handle),
                        "hbDNNGetOutputCount failed");

    // Resize tensor storage
    input_tensors.resize(input_count);
    output_tensors.resize(output_count);

    // Query tensor properties
    for (int i = 0; i < input_count; i++) {
        HBDNN_CHECK_SUCCESS(hbDNNGetInputTensorProperties(&input_tensors[i].properties, dnn_handle, i),
                            "hbDNNGetInputTensorProperties failed");
    }
    for (int i = 0; i < output_count; i++) {
        HBDNN_CHECK_SUCCESS(hbDNNGetOutputTensorProperties(&output_tensors[i].properties, dnn_handle, i),
                            "hbDNNGetOutputTensorProperties failed");
    }

    // Cache expected input resolution from the first input tensor (Y plane)
    input_h = input_tensors[0].properties.validShape.dimensionSize[1];
    input_w = input_tensors[0].properties.validShape.dimensionSize[2];

    // Allocate tensor memory
    prepare_input_tensor(input_tensors);
    prepare_output_tensor(output_tensors);

    inited = true;
    return 0;
}

/**
 * @brief Destructor: release tensor memory and DNN model resources.
 */
YOLO11::~YOLO11()
{
    for (int i = 0; i < input_count; i++) {
        hbUCPFree(&(input_tensors[i].sysMem));
    }
    for (int i = 0; i < output_count; i++) {
        hbUCPFree(&(output_tensors[i].sysMem));
    }
    if (packed_dnn_handle) {
        hbDNNRelease(packed_dnn_handle);
    }
}

// ---------------------------------------------------------------------------
// Free pipeline functions
// ---------------------------------------------------------------------------

/**
 * @brief Preprocess an input BGR image into NV12 model input tensors.
 *
 * @param[in,out] input_tensors Model input tensors to be filled.
 * @param[in]     img           Input image in BGR format.
 * @param[in]     input_w       Model input width in pixels.
 * @param[in]     input_h       Model input height in pixels.
 * @param[in]     image_format  Input format string (only "BGR" supported).
 * @retval 0        Success.
 * @retval -1       Unsupported image format.
 */
int32_t pre_process(std::vector<hbDNNTensor>& input_tensors,
                    cv::Mat& img,
                    const int input_w, const int input_h,
                    const std::string& image_format)
{
    if (image_format != "BGR") {
        fprintf(stderr, "Unsupported image_format: %s\n", image_format.c_str());
        return -1;
    }

    cv::Mat resized_mat;
    resized_mat.create(input_h, input_w, img.type());
    letterbox_resize(img, resized_mat);

    return bgr_to_nv12_tensor(resized_mat, input_tensors, input_h, input_w);
}

/**
 * @brief Execute synchronous BPU inference on prepared input tensors.
 *
 * @param[in,out] output_tensors Output tensors to be filled by runtime.
 * @param[in]     input_tensors  Prepared input tensors.
 * @param[in]     dnn_handle     DNN model handle.
 * @param[in]     sched_param    Optional UCP scheduling parameters.
 * @retval 0        Success.
 * @retval non-zero DNN or UCP API error.
 */
int32_t infer(std::vector<hbDNNTensor>& output_tensors,
              std::vector<hbDNNTensor>& input_tensors,
              const hbDNNHandle_t dnn_handle,
              hbUCPSchedParam* sched_param)
{
    hbUCPTaskHandle_t task_handle{nullptr};

    // Create inference task
    HBDNN_CHECK_SUCCESS(hbDNNInferV2(&task_handle, output_tensors.data(), input_tensors.data(), dnn_handle),
                        "hbDNNInferV2 failed");

    // Configure and submit to BPU scheduler
    hbUCPSchedParam ctrl_param;
    if (sched_param) {
        ctrl_param = *sched_param;
    } else {
        HB_UCP_INITIALIZE_SCHED_PARAM(&ctrl_param);
        ctrl_param.backend = HB_UCP_BPU_CORE_ANY;
    }
    HBUCP_CHECK_SUCCESS(hbUCPSubmitTask(task_handle, &ctrl_param), "hbUCPSubmitTask failed");

    // Wait for inference to complete (blocking)
    HBUCP_CHECK_SUCCESS(hbUCPWaitTaskDone(task_handle, 0), "hbUCPWaitTaskDone failed");

    // Invalidate CPU cache for output tensors
    for (auto& t : output_tensors) {
        hbUCPMemFlush(&t.sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
    }

    HBUCP_CHECK_SUCCESS(hbUCPReleaseTask(task_handle), "hbUCPReleaseTask failed");
    return 0;
}

/**
 * @brief Postprocess YOLO11 DFL outputs into final detection results.
 *
 * @param[out] results        Final detections after NMS and coordinate rescaling.
 * @param[in]  output_tensors Raw output tensors from inference.
 * @param[in]  config         Inference configuration.
 * @param[in]  orig_img_w     Original image width (pixels).
 * @param[in]  orig_img_h     Original image height (pixels).
 * @param[in]  input_w        Model input width (pixels).
 * @param[in]  input_h        Model input height (pixels).
 */
void post_process(std::vector<Detection>& results,
                  std::vector<hbDNNTensor>& output_tensors,
                  const Yolo11Config& config,
                  int orig_img_w, int orig_img_h,
                  int input_w, int input_h)
{
    // Convert probability threshold to logit domain
    float conf_thres_raw = -std::log(1.0f / config.score_thresh - 1.0f);

    std::vector<Detection> all_detections;

    // Decode each detection head: output layout is [cls0, box0, cls1, box1, cls2, box2]
    for (size_t s = 0; s < config.strides.size(); ++s) {
        const hbDNNTensor& cls_tensor  = output_tensors[2 * s];
        const hbDNNTensor& bbox_tensor = output_tensors[2 * s + 1];

        std::vector<Detection> head_dets;
        filter_and_decode(cls_tensor, bbox_tensor, conf_thres_raw,
                          config.anchor_sizes[s], config.strides[s],
                          config.reg, head_dets);

        all_detections.insert(all_detections.end(),
                              std::make_move_iterator(head_dets.begin()),
                              std::make_move_iterator(head_dets.end()));
    }

    // NMS across all heads
    results = nms_bboxes(all_detections, config.nms_thresh);

    // Rescale boxes from model input space to original image coordinates
    scale_letterbox_bboxes_back(results, orig_img_w, orig_img_h, input_w, input_h);
}
