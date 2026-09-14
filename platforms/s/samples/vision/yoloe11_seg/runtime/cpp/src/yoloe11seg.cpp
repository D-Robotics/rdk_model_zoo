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
 * @file yoloe11seg.cpp
 * @brief Implement the YOLOe11-Seg open-vocabulary inference pipeline using HB-DNN / UCP runtime APIs.
 *
 * This file contains the complete implementation of the YOLOe11-Seg wrapper:
 * - Model loading, tensor allocation and deallocation.
 * - Letterbox resize + BGR->NV12 preprocessing.
 * - Synchronous BPU inference via hbDNNInferV2() and UCP APIs.
 * - Per-head DFL box decoding and MCES extraction with logit-domain filtering.
 *   OpenMP parallelism is used to handle the 4585-class argmax efficiently.
 * - Class-wise NMS with MCES vectors kept in sync.
 * - Prototype-based mask generation (linear combination + sigmoid threshold).
 * - Mask resizing to original bounding box coordinates.
 *
 * @see yoloe11seg.hpp
 */

#include "yoloe11seg.hpp"
#include <omp.h>
#include <opencv2/imgproc.hpp>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * @brief Filter class logits and decode DFL boxes + MCES for one detection head.
 *
 * For each spatial position (h, w):
 * 1) Argmax over C class logits; skip if below @p conf_thres_raw.
 * 2) Decode 4×reg DFL logits from bbox_tensor to (x1,y1,x2,y2) in input scale.
 * 3) Read mces_num MCES logits from mces_tensor (dequantized per-channel).
 *
 * OpenMP parallelism is applied over (h, w) to handle 4585-class argmax efficiently.
 * Results are accumulated in thread-local buffers and merged afterwards.
 *
 * @param[in]  cls_tensor      Classification float tensor (1, H, W, C).
 * @param[in]  bbox_tensor     Box distribution S32 tensor (1, H, W, 4*reg).
 * @param[in]  mces_tensor     MCES S32 tensor (1, H, W, mces_num).
 * @param[in]  conf_thres_raw  Logit-domain confidence threshold.
 * @param[in]  grid_size       Feature map spatial size for this head.
 * @param[in]  stride          Downsampling stride for this head.
 * @param[in]  reg             Number of DFL bins per bbox side.
 * @param[in]  mces_num        MCES vector dimension.
 * @param[out] detections      Decoded detections appended here.
 * @param[out] all_mces        MCES vectors aligned with @p detections.
 */
static void filter_and_decode_mces(
    const hbDNNTensor& cls_tensor,
    const hbDNNTensor& bbox_tensor,
    const hbDNNTensor& mces_tensor,
    float conf_thres_raw,
    int grid_size,
    int stride,
    int reg,
    int mces_num,
    std::vector<Detection>& detections,
    std::vector<std::vector<float>>& all_mces)
{
    const hbDNNTensorShape& shape = cls_tensor.properties.validShape;
    int H = shape.dimensionSize[1];
    int W = shape.dimensionSize[2];
    int C = shape.dimensionSize[3];

    const int64_t* s_cls  = cls_tensor.properties.stride;
    const int64_t* s_bbox = bbox_tensor.properties.stride;
    const int64_t* s_mces = mces_tensor.properties.stride;

    const uint8_t* d_cls  = reinterpret_cast<const uint8_t*>(cls_tensor.sysMem.virAddr);
    const uint8_t* d_bbox = reinterpret_cast<const uint8_t*>(bbox_tensor.sysMem.virAddr);
    const uint8_t* d_mces = reinterpret_cast<const uint8_t*>(mces_tensor.sysMem.virAddr);

    // Thread-local caches to avoid synchronization inside the parallel region
    int max_threads = omp_get_max_threads();
    std::vector<std::vector<Detection>>          thread_dets(max_threads);
    std::vector<std::vector<std::vector<float>>> thread_mces(max_threads);

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            int tid = omp_get_thread_num();

            size_t base_cls  = h * s_cls[1]  + w * s_cls[2];
            size_t base_bbox = h * s_bbox[1] + w * s_bbox[2];
            size_t base_mces = h * s_mces[1] + w * s_mces[2];

            // 1) Argmax over class logits (4585 classes — parallelised at outer level)
            float max_val = -1e30f;
            int   max_id  = 0;
            for (int c = 0; c < C; ++c) {
                float val = *reinterpret_cast<const float*>(d_cls + base_cls + c * s_cls[3]);
                if (val > max_val) { max_val = val; max_id = c; }
            }
            if (max_val < conf_thres_raw) continue;

            Detection det{};
            det.score    = sigmoid(max_val);
            det.class_id = max_id;

            // 2) DFL box decoding: 4 sides × reg bins
            float anchor_x = 0.5f + w;
            float anchor_y = 0.5f + h;
            float ltrb[4]  = {0.f, 0.f, 0.f, 0.f};

            for (int side = 0; side < 4; ++side) {
                float bins[16] = {};
                for (int bin = 0; bin < reg; ++bin) {
                    int ch = side * reg + bin;
                    const int32_t* ptr = reinterpret_cast<const int32_t*>(
                        d_bbox + base_bbox + ch * s_bbox[3]);
                    bins[bin] = dequant_value(*ptr, ch, bbox_tensor.properties);
                }
                float probs[16] = {};
                softmax(bins, probs, reg);
                for (int i = 0; i < reg; ++i) ltrb[side] += probs[i] * i;
            }

            det.bbox[0] = (anchor_x - ltrb[0]) * stride;
            det.bbox[1] = (anchor_y - ltrb[1]) * stride;
            det.bbox[2] = (anchor_x + ltrb[2]) * stride;
            det.bbox[3] = (anchor_y + ltrb[3]) * stride;

            // 3) Read MCES coefficients
            std::vector<float> mces_vec(mces_num);
            for (int d = 0; d < mces_num; ++d) {
                const int32_t* ptr = reinterpret_cast<const int32_t*>(
                    d_mces + base_mces + d * s_mces[3]);
                mces_vec[d] = dequant_value(*ptr, d, mces_tensor.properties);
            }

            thread_dets[tid].push_back(det);
            thread_mces[tid].push_back(std::move(mces_vec));
        }
    }

    // Merge thread-local results into output vectors
    for (int t = 0; t < max_threads; ++t) {
        detections.insert(detections.end(),
                          std::make_move_iterator(thread_dets[t].begin()),
                          std::make_move_iterator(thread_dets[t].end()));
        all_mces.insert(all_mces.end(),
                        std::make_move_iterator(thread_mces[t].begin()),
                        std::make_move_iterator(thread_mces[t].end()));
    }
}

/**
 * @brief Class-wise NMS that keeps MCES vectors aligned with detections.
 *
 * @param[in]  detections  All candidate detections.
 * @param[in]  mces        MCES vectors aligned with @p detections.
 * @param[in]  iou_thresh  IoU threshold for suppression.
 * @return Pair of (kept detections, kept MCES), order-aligned.
 */
static std::pair<std::vector<Detection>, std::vector<std::vector<float>>>
nms_with_mces(const std::vector<Detection>& detections,
              const std::vector<std::vector<float>>& mces,
              float iou_thresh)
{
    std::vector<Detection>             kept_dets;
    std::vector<std::vector<float>>    kept_mces;

    // Group indices by class
    std::unordered_map<int, std::vector<size_t>> class_map;
    for (size_t i = 0; i < detections.size(); ++i)
        class_map[detections[i].class_id].push_back(i);

    for (auto& [cls_id, idx_list] : class_map) {
        // Sort by score descending
        std::sort(idx_list.begin(), idx_list.end(),
                  [&](size_t a, size_t b) {
                      return detections[a].score > detections[b].score;
                  });

        std::vector<bool> suppressed(idx_list.size(), false);

        for (size_t i = 0; i < idx_list.size(); ++i) {
            if (suppressed[i]) continue;
            size_t ki = idx_list[i];
            kept_dets.push_back(detections[ki]);
            kept_mces.push_back(mces[ki]);

            for (size_t j = i + 1; j < idx_list.size(); ++j) {
                if (!suppressed[j] &&
                    iou(detections[ki], detections[idx_list[j]]) > iou_thresh)
                    suppressed[j] = true;
            }
        }
    }
    return {kept_dets, kept_mces};
}

/**
 * @brief Decode per-instance masks from prototype features and MCES coefficients.
 *
 * For each detection, crops the proto tensor to the scaled bounding box region,
 * computes the linear combination proto × mces, applies sigmoid, and binarizes.
 *
 * @param[in]  detections  Final detections (boxes in input scale at call time).
 * @param[in]  mces        MCES vectors aligned with @p detections.
 * @param[in]  protos      Prototype features, flattened NHWC (H=mask_h, W=mask_w, C=mces_num).
 * @param[in]  input_w     Model input width (pixels).
 * @param[in]  input_h     Model input height (pixels).
 * @param[in]  mask_w      Prototype feature map width.
 * @param[in]  mask_h      Prototype feature map height.
 * @param[in]  mask_thresh Threshold applied after sigmoid for binary mask.
 * @return Vector of CV_8UC1 binary masks cropped at proto scale (0 or 1 values).
 */
// Note: this variant thresholds the raw linear combination directly (no sigmoid),
// which is intentional for the open-vocabulary model. It differs from the
// decode_masks() in utils/c_utils which applies sigmoid before thresholding.
static std::vector<cv::Mat> decode_masks_linear(
    const std::vector<Detection>& detections,
    const std::vector<std::vector<float>>& mces,
    const std::vector<float>& protos,
    int input_w, int input_h,
    int mask_w, int mask_h,
    float mask_thresh)
{
    std::vector<cv::Mat> masks;
    if (detections.empty()) return masks;

    int C = static_cast<int>(mces[0].size());
    float x_scale = static_cast<float>(mask_w) / input_w;
    float y_scale = static_cast<float>(mask_h) / input_h;

    masks.reserve(detections.size());

    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        const auto& mc  = mces[i];

        // Scale bbox to proto resolution and clamp
        int x1 = std::max(0, std::min(mask_w, static_cast<int>(det.bbox[0] * x_scale)));
        int y1 = std::max(0, std::min(mask_h, static_cast<int>(det.bbox[1] * y_scale)));
        int x2 = std::max(0, std::min(mask_w, static_cast<int>(det.bbox[2] * x_scale)));
        int y2 = std::max(0, std::min(mask_h, static_cast<int>(det.bbox[3] * y_scale)));

        int crop_h = y2 - y1;
        int crop_w = x2 - x1;
        if (crop_h <= 0 || crop_w <= 0) { masks.emplace_back(); continue; }

        cv::Mat mask(crop_h, crop_w, CV_8UC1, cv::Scalar(0));

        for (int yy = 0; yy < crop_h; ++yy) {
            for (int xx = 0; xx < crop_w; ++xx) {
                const float* pixel = &protos[((y1 + yy) * mask_w + (x1 + xx)) * C];
                float val = 0.f;
                for (int c = 0; c < C; ++c) val += pixel[c] * mc[c];
                // Threshold raw linear combination directly (no sigmoid), matching Python utils
                mask.at<uint8_t>(yy, xx) = (val > mask_thresh) ? 1 : 0;
            }
        }
        masks.push_back(mask);
    }
    return masks;
}

// ---------------------------------------------------------------------------
// YOLO11ESeg class
// ---------------------------------------------------------------------------

/**
 * @brief Construct a YOLO11ESeg instance in an uninitialized state.
 */
YOLO11ESeg::YOLO11ESeg()
{
    model_count_       = 0;
    packed_dnn_handle_ = nullptr;
    dnn_handle         = nullptr;
    input_count_       = 0;
    output_count_      = 0;
    input_h            = 0;
    input_w            = 0;
    inited_            = false;
}

/**
 * @brief Initialize model resources from a *.hbm model file.
 *
 * @param[in] model_path Path to the quantized *.hbm model file.
 * @retval 0        Success.
 * @retval non-zero DNN or UCP API error.
 */
int32_t YOLO11ESeg::init(const char* model_path)
{
    const char** model_name_list = nullptr;

    if (inited_) {
        fprintf(stderr, "YOLO11ESeg::init() called twice\n");
        return -1;
    }

    HBDNN_CHECK_SUCCESS(hbDNNInitializeFromFiles(&packed_dnn_handle_, &model_path, 1),
                        "hbDNNInitializeFromFiles failed");
    HBDNN_CHECK_SUCCESS(hbDNNGetModelNameList(&model_name_list, &model_count_, packed_dnn_handle_),
                        "hbDNNGetModelNameList failed");
    HBDNN_CHECK_SUCCESS(hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle_, model_name_list[0]),
                        "hbDNNGetModelHandle failed");
    HBDNN_CHECK_SUCCESS(hbDNNGetInputCount(&input_count_, dnn_handle),
                        "hbDNNGetInputCount failed");
    HBDNN_CHECK_SUCCESS(hbDNNGetOutputCount(&output_count_, dnn_handle),
                        "hbDNNGetOutputCount failed");

    input_tensors.resize(input_count_);
    output_tensors.resize(output_count_);

    for (int i = 0; i < input_count_; i++)
        HBDNN_CHECK_SUCCESS(hbDNNGetInputTensorProperties(&input_tensors[i].properties, dnn_handle, i),
                            "hbDNNGetInputTensorProperties failed");
    for (int i = 0; i < output_count_; i++)
        HBDNN_CHECK_SUCCESS(hbDNNGetOutputTensorProperties(&output_tensors[i].properties, dnn_handle, i),
                            "hbDNNGetOutputTensorProperties failed");

    input_h = input_tensors[0].properties.validShape.dimensionSize[1];
    input_w = input_tensors[0].properties.validShape.dimensionSize[2];

    prepare_input_tensor(input_tensors);
    prepare_output_tensor(output_tensors);

    inited_ = true;
    return 0;
}

/**
 * @brief Destructor: release tensor memory and DNN model resources.
 */
YOLO11ESeg::~YOLO11ESeg()
{
    for (int i = 0; i < input_count_; i++)  hbUCPFree(&(input_tensors[i].sysMem));
    for (int i = 0; i < output_count_; i++) hbUCPFree(&(output_tensors[i].sysMem));
    if (packed_dnn_handle_) hbDNNRelease(packed_dnn_handle_);
}

// ---------------------------------------------------------------------------
// Free pipeline functions
// ---------------------------------------------------------------------------

/**
 * @brief Preprocess an input BGR image into NV12 model input tensors.
 *
 * @param[in,out] input_tensors Model input tensors to fill.
 * @param[in]     img           Input BGR image.
 * @param[in]     input_w       Model input width.
 * @param[in]     input_h       Model input height.
 * @param[in]     image_format  Format string (only "BGR" supported).
 * @retval 0  Success.
 * @retval -1 Unsupported format.
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
 * @param[in,out] output_tensors Output tensors filled by the runtime.
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

    HBDNN_CHECK_SUCCESS(hbDNNInferV2(&task_handle, output_tensors.data(), input_tensors.data(), dnn_handle),
                        "hbDNNInferV2 failed");

    hbUCPSchedParam ctrl_param;
    if (sched_param) {
        ctrl_param = *sched_param;
    } else {
        HB_UCP_INITIALIZE_SCHED_PARAM(&ctrl_param);
        ctrl_param.backend = HB_UCP_BPU_CORE_ANY;
    }
    HBUCP_CHECK_SUCCESS(hbUCPSubmitTask(task_handle, &ctrl_param), "hbUCPSubmitTask failed");
    HBUCP_CHECK_SUCCESS(hbUCPWaitTaskDone(task_handle, 0), "hbUCPWaitTaskDone failed");

    for (auto& t : output_tensors)
        hbUCPMemFlush(&t.sysMem, HB_SYS_MEM_CACHE_INVALIDATE);

    HBUCP_CHECK_SUCCESS(hbUCPReleaseTask(task_handle), "hbUCPReleaseTask failed");
    return 0;
}

/**
 * @brief Postprocess YOLOe11-Seg outputs into final detections and instance masks.
 *
 * @param[in]  output_tensors Raw output tensors from inference.
 * @param[in]  config         Inference configuration.
 * @param[in]  orig_img_w     Original image width.
 * @param[in]  orig_img_h     Original image height.
 * @param[in]  input_w        Model input width.
 * @param[in]  input_h        Model input height.
 * @return InstanceSegResult  Detections and per-instance masks (index-aligned).
 */
InstanceSegResult post_process(std::vector<hbDNNTensor>& output_tensors,
                                const YoloE11SegConfig& config,
                                int orig_img_w, int orig_img_h,
                                int input_w, int input_h)
{
    float conf_thres_raw = -std::log(1.0f / config.score_thresh - 1.0f);

    std::vector<Detection>           all_dets;
    std::vector<std::vector<float>>  all_mces;

    // Step 1 & 2: Decode each head (output layout: [cls, box, mces] × 3 scales + protos)
    // OpenMP parallelism inside filter_and_decode_mces handles the 4585-class argmax
    for (size_t s = 0; s < config.strides.size(); ++s) {
        const hbDNNTensor& cls_t  = output_tensors[3 * s + 0];
        const hbDNNTensor& bbox_t = output_tensors[3 * s + 1];
        const hbDNNTensor& mces_t = output_tensors[3 * s + 2];

        std::vector<Detection>          head_dets;
        std::vector<std::vector<float>> head_mces;

        filter_and_decode_mces(cls_t, bbox_t, mces_t, conf_thres_raw,
                               config.anchor_sizes[s], config.strides[s],
                               config.reg, config.mces_num,
                               head_dets, head_mces);

        all_dets.insert(all_dets.end(),
                        std::make_move_iterator(head_dets.begin()),
                        std::make_move_iterator(head_dets.end()));
        all_mces.insert(all_mces.end(),
                        std::make_move_iterator(head_mces.begin()),
                        std::make_move_iterator(head_mces.end()));
    }

    // Step 3: Class-wise NMS (keeps MCES aligned)
    auto [final_dets, final_mces] = nms_with_mces(all_dets, all_mces, config.nms_thresh);

    // Step 4: Dequantize prototype tensor (output index 9, int16 per-N scale)
    auto protos = dequantize_s16_axis0(output_tensors[9]);

    // Step 5: Decode per-instance binary masks in proto space (linear threshold, no sigmoid)
    auto raw_masks = decode_masks_linear(final_dets, final_mces, protos,
                                  input_w, input_h,
                                  config.proto_size, config.proto_size,
                                  config.mask_thresh);

    // Step 6: Rescale boxes to original image coordinates
    scale_letterbox_bboxes_back(final_dets, orig_img_w, orig_img_h, input_w, input_h);

    // Step 7: Resize masks to their boxes in original image space
    InstanceSegResult result;
    result.masks      = resize_masks_to_boxes(raw_masks, final_dets, orig_img_w, orig_img_h, config.do_morph);
    result.detections = std::move(final_dets);
    return result;
}
