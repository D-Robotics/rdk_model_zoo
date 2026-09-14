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
 * @file yolo11pose.cpp
 * @brief Implement the YOLO11-Pose inference pipeline using HB-DNN / UCP runtime APIs.
 *
 * This file contains the complete implementation of the YOLO11-Pose wrapper:
 * - Model loading, tensor allocation and deallocation.
 * - Letterbox resize + BGR->NV12 preprocessing.
 * - Synchronous BPU inference via hbDNNInferV2() and UCP APIs.
 * - Per-head DFL box decoding and keypoint extraction with logit-domain filtering.
 * - Class-wise NMS with keypoints kept in sync.
 * - Coordinate rescaling to original image space (inverse letterbox).
 *
 * OpenMP is used to parallelize the per-head decode step for improved throughput.
 *
 * @see yolo11pose.hpp
 */

#include "yolo11pose.hpp"
#include <omp.h>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * @brief Filter class logits, decode DFL boxes, and decode keypoints for one detection head.
 *
 * For each spatial position (h, w):
 * 1) Read single-channel confidence logit; skip if below @p conf_thres_raw.
 * 2) Decode 4×reg DFL logits from @p bbox_tensor to (x1,y1,x2,y2) in input scale.
 * 3) Decode 17 keypoints (dx, dy, score) from @p kpts_tensor and map to input scale.
 *
 * Parallelized across H×W with OpenMP; merges thread-local buffers at the end.
 *
 * @param[in]  cls_tensor      Classification float tensor (1, H, W, 1).
 * @param[in]  bbox_tensor     Box distribution float tensor (1, H, W, 4*reg).
 * @param[in]  kpts_tensor     Keypoints float tensor (1, H, W, 17*3).
 * @param[in]  conf_thres_raw  Logit-domain confidence threshold.
 * @param[in]  grid_size       Feature map spatial size for this head.
 * @param[in]  stride          Downsampling stride for this head.
 * @param[in]  reg             Number of DFL bins per bbox side.
 * @param[out] detections      Decoded detections appended here.
 * @param[out] all_kpts        Decoded keypoints aligned with @p detections.
 */
static void filter_and_decode_kpts(
    const hbDNNTensor& cls_tensor,
    const hbDNNTensor& bbox_tensor,
    const hbDNNTensor& kpts_tensor,
    float conf_thres_raw,
    int grid_size,
    int stride,
    int reg,
    std::vector<Detection>& detections,
    std::vector<std::vector<Keypoint>>& all_kpts)
{
    const hbDNNTensorShape& shape = cls_tensor.properties.validShape;
    int H = shape.dimensionSize[1];
    int W = shape.dimensionSize[2];

    const int64_t* s_cls  = cls_tensor.properties.stride;
    const int64_t* s_bbox = bbox_tensor.properties.stride;
    const int64_t* s_kpts = kpts_tensor.properties.stride;

    const uint8_t* d_cls  = reinterpret_cast<const uint8_t*>(cls_tensor.sysMem.virAddr);
    const uint8_t* d_bbox = reinterpret_cast<const uint8_t*>(bbox_tensor.sysMem.virAddr);
    const uint8_t* d_kpts = reinterpret_cast<const uint8_t*>(kpts_tensor.sysMem.virAddr);

    // Thread-local caches to reduce contention
    const int nthreads = omp_get_max_threads();
    std::vector<std::vector<Detection>>              thread_dets(nthreads);
    std::vector<std::vector<std::vector<Keypoint>>>  thread_kpts(nthreads);

    // Parallel over spatial grid
    #pragma omp parallel for collapse(2)
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            int tid = omp_get_thread_num();
            auto& dets_local = thread_dets[tid];
            auto& kpts_local = thread_kpts[tid];

            size_t base_cls  = h * s_cls[1]  + w * s_cls[2];
            size_t base_bbox = h * s_bbox[1] + w * s_bbox[2];
            size_t base_kpts = h * s_kpts[1] + w * s_kpts[2];

            // 1) Single-channel confidence logit (pose model uses 1-class head)
            const float* ptr_cls = reinterpret_cast<const float*>(d_cls + base_cls);
            float val = *ptr_cls;
            if (val < conf_thres_raw) continue;

            Detection det{};
            det.score    = sigmoid(val);
            det.class_id = 0;  // single-class: person

            // 2) DFL box decoding: 4 sides × reg bins
            float anchor_x = 0.5f + w;
            float anchor_y = 0.5f + h;
            float ltrb[4]  = {0.f, 0.f, 0.f, 0.f};

            for (int side = 0; side < 4; ++side) {
                float bins[16] = {};
                for (int bin = 0; bin < reg; ++bin) {
                    int ch = side * reg + bin;
                    const float* ptr_bbox = reinterpret_cast<const float*>(
                        d_bbox + base_bbox + ch * s_bbox[3]);
                    bins[bin] = *ptr_bbox;
                }
                // Softmax with max-trick
                float max_bin = bins[0];
                for (int i = 1; i < reg; ++i) if (bins[i] > max_bin) max_bin = bins[i];
                float sum = 0.f;
                float probs[16] = {};
                for (int i = 0; i < reg; ++i) { probs[i] = std::exp(bins[i] - max_bin); sum += probs[i]; }
                for (int i = 0; i < reg; ++i) ltrb[side] += probs[i] * i / sum;
            }

            det.bbox[0] = (anchor_x - ltrb[0]) * stride;
            det.bbox[1] = (anchor_y - ltrb[1]) * stride;
            det.bbox[2] = (anchor_x + ltrb[2]) * stride;
            det.bbox[3] = (anchor_y + ltrb[3]) * stride;

            // 3) Keypoints decode: 17 × (dx, dy, score)
            std::vector<Keypoint> kpts_vec(17);
            for (int k = 0; k < 17; ++k) {
                int base_ch = 3 * k;
                const float* ptr_k = reinterpret_cast<const float*>(
                    d_kpts + base_kpts + base_ch * s_kpts[3]);
                // Offsets are predicted on the grid; map to input-space pixels
                kpts_vec[k].x     = (ptr_k[0] * 2.0f + (anchor_x - 0.5f)) * stride;
                kpts_vec[k].y     = (ptr_k[1] * 2.0f + (anchor_y - 0.5f)) * stride;
                kpts_vec[k].score = ptr_k[2];  // raw logit, not sigmoid
            }

            dets_local.push_back(det);
            kpts_local.push_back(std::move(kpts_vec));
        }
    }

    // Merge thread-local results
    for (int t = 0; t < nthreads; ++t) {
        detections.insert(detections.end(),
                          std::make_move_iterator(thread_dets[t].begin()),
                          std::make_move_iterator(thread_dets[t].end()));
        all_kpts.insert(all_kpts.end(),
                        std::make_move_iterator(thread_kpts[t].begin()),
                        std::make_move_iterator(thread_kpts[t].end()));
    }
}

/**
 * @brief Class-wise NMS that keeps keypoints aligned with detections.
 *
 * @param[in]  detections  All candidate detections.
 * @param[in]  kpts        Keypoints aligned with @p detections.
 * @param[in]  iou_thresh  IoU threshold for suppression.
 * @return Pair of (kept detections, kept keypoints), order-aligned.
 */
static std::pair<std::vector<Detection>, std::vector<std::vector<Keypoint>>>
nms_with_kpts(const std::vector<Detection>& detections,
              const std::vector<std::vector<Keypoint>>& kpts,
              float iou_thresh)
{
    std::vector<Detection>              kept_dets;
    std::vector<std::vector<Keypoint>>  kept_kpts;

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
            kept_kpts.push_back(kpts[ki]);

            for (size_t j = i + 1; j < idx_list.size(); ++j) {
                if (!suppressed[j] &&
                    iou(detections[ki], detections[idx_list[j]]) > iou_thresh)
                    suppressed[j] = true;
            }
        }
    }
    return {kept_dets, kept_kpts};
}

// ---------------------------------------------------------------------------
// YOLO11Pose class
// ---------------------------------------------------------------------------

/**
 * @brief Construct a YOLO11Pose instance in an uninitialized state.
 */
YOLO11Pose::YOLO11Pose()
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
int32_t YOLO11Pose::init(const char* model_path)
{
    const char** model_name_list = nullptr;

    if (inited_) {
        fprintf(stderr, "YOLO11Pose::init() called twice\n");
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
YOLO11Pose::~YOLO11Pose()
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
 * @brief Postprocess YOLO11-Pose outputs into final detections and keypoints.
 *
 * @param[in]  output_tensors Raw output tensors from inference.
 * @param[in]  config         Inference configuration.
 * @param[in]  orig_img_w     Original image width.
 * @param[in]  orig_img_h     Original image height.
 * @param[in]  input_w        Model input width.
 * @param[in]  input_h        Model input height.
 * @return Pair of (detections, keypoints_per_detection) in original image coordinates.
 */
PoseResult post_process(std::vector<hbDNNTensor>& output_tensors,
                        const YOLO11PoseConfig& config,
                        int orig_img_w, int orig_img_h,
                        int input_w, int input_h)
{
    float conf_thres_raw = -std::log(1.0f / config.score_thresh - 1.0f);

    std::vector<Detection>             all_dets;
    std::vector<std::vector<Keypoint>> all_kpts;

    // Step 1 & 2: Decode each head (output layout: [cls, box, kpts] × 3 scales)
    for (size_t s = 0; s < config.strides.size(); ++s) {
        const hbDNNTensor& cls_t  = output_tensors[3 * s + 0];
        const hbDNNTensor& bbox_t = output_tensors[3 * s + 1];
        const hbDNNTensor& kpts_t = output_tensors[3 * s + 2];

        std::vector<Detection>             head_dets;
        std::vector<std::vector<Keypoint>> head_kpts;

        filter_and_decode_kpts(cls_t, bbox_t, kpts_t, conf_thres_raw,
                               config.anchor_sizes[s], config.strides[s],
                               config.reg,
                               head_dets, head_kpts);

        all_dets.insert(all_dets.end(),
                        std::make_move_iterator(head_dets.begin()),
                        std::make_move_iterator(head_dets.end()));
        all_kpts.insert(all_kpts.end(),
                        std::make_move_iterator(head_kpts.begin()),
                        std::make_move_iterator(head_kpts.end()));
    }

    // Step 3: Class-wise NMS (keeps keypoints aligned)
    auto [final_dets, final_kpts] = nms_with_kpts(all_dets, all_kpts, config.nms_thresh);

    // Step 4: Rescale boxes and keypoints to original image coordinates
    scale_letterbox_bboxes_back(final_dets, orig_img_w, orig_img_h, input_w, input_h);
    scale_keypoints_back_letterbox(final_kpts, orig_img_w, orig_img_h, input_w, input_h);

    PoseResult result;
    result.detections = std::move(final_dets);
    result.keypoints  = std::move(final_kpts);
    return result;
}
