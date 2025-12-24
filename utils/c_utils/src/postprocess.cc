/*
 * Copyright (c) 2025, XiangshunZhao D-Robotics.
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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <unordered_map>
#include <iostream>

#include "nn_math.hpp"
#include "postprocess.hpp"

/**
 * @brief Get top-k classifications from the first output tensor (float logits).
 * @param[in]  tensor      Output tensor list; uses tensor[0] as a contiguous float array of logits.
 * @param[out] top_k_cls   Collected top-k results sorted by descending score.
 * @param[in]  top_k       Number of classes to keep.
 *
 * @note This implementation assumes 1000-way classification and float32 output (quantiType == NONE).
 */
void get_topk_result(hbDNNTensor& tensor,
                     std::vector<Classification> &top_k_cls, int top_k) {
  std::priority_queue<Classification, std::vector<Classification>,
                      std::greater<Classification>>
      queue; // min-heap to keep top-k

  // The type reinterpret_cast should be determined according to the output type
  // For example: HB_DNN_TENSOR_TYPE_F32 is float
  auto data = reinterpret_cast<float *>(tensor.sysMem.virAddr);
  auto quanti_type{tensor.properties.quantiType};

  // For example model, quantiType is NONE and no dequantize processing is required.
  if (quanti_type != hbDNNQuantiType::NONE) {
    std::cout << "quanti_type is not NONE, and the output needs to be dequantized!";
  }

  // 1000 classification score values (assumption for ImageNet-like models)
  int tensor_len = 1000;
  for (auto i = 0; i < tensor_len; i++) {
    float score = data[i];
    queue.push(Classification(i, score, "")); // push (id, score)
    if (queue.size() > top_k) {
      queue.pop(); // pop smallest to keep heap size == top_k
    }
  }

  // dump heap (ascending) into vector then reverse
  while (!queue.empty()) {
    top_k_cls.emplace_back(queue.top());
    queue.pop();
  }
  std::reverse(top_k_cls.begin(), top_k_cls.end()); // highest first
}

/**
 * @brief Dequantize an int32 per-channel tensor (NHWC) to float32.
 * @param[in] tensor   DNN tensor with S32 data, per-channel scale/zero-point, and byte strides.
 * @return std::vector<float>  Flattened NHWC array of size N*H*W*C in row-major order.
 */
std::vector<float> dequantizeTensorS32(const hbDNNTensor& tensor) {
    const auto& props = tensor.properties;
    const auto& shape = props.validShape;
    const int N = shape.dimensionSize[0];
    const int H = shape.dimensionSize[1];
    const int W = shape.dimensionSize[2];
    const int C = shape.dimensionSize[3];

    int total = N * H * W * C;
    std::vector<float> result(total);

    const float* scale_data = props.scale.scaleData;    // per-channel scale
    const int32_t* zp_data = props.scale.zeroPointData; // per-channel zero-point (optional)
    uint8_t* base = static_cast<uint8_t*>(tensor.sysMem.virAddr);
    int stride = props.stride[3];  // byte stride between channels

    // Iterate over NHW and channels; respect byte strides
      for (int h = 0; h < H; ++h) {
        int offset_h = h * shape.dimensionSize[2];
        int idx_h = h * W ;
        for (int w = 0; w < W; ++w) {
            int offset_w = (offset_h + w) * props.stride[2];
            int idx_w = (idx_h + w) * C;
          for (int c = 0; c < C; ++c) {
            int idx = idx_w + c;

            // Compute byte offset using strides for H/W and channel stride for C
            int offset = offset_w + c * stride;

            float s = scale_data[c];
            int zp = zp_data ? zp_data[c] : 0;
            float val = static_cast<float>(*(int32_t*)(base + offset));
            result[idx] = (val - zp) * s; // dequantize: (q - zp) * scale
          }
        }
      }
    return result;
}


/**
 * @brief Decode YOLOv5 multi-level, anchor-based outputs to detections.
 *
 * @param[in] all_results  Per-level flat outputs; each level is na*h*w*(5+num_classes).
 * @param[in] hw_list      (h,w) for each level.
 * @param[in] strides      Stride for each level.
 * @param[in] all_anchors  Anchors per level: {{aw, ah}, ...}.
 * @param[in] score_thresh Keep boxes with sigmoid(obj)*sigmoid(max_class) >= threshold.
 * @param[in] num_classes  Number of classes.
 * @return std::vector<Detection> Detections as {x1,y1,x2,y2,score,class_id}.
 *
 * @note NMS is not applied here.
 */
std::vector<Detection> yolov5_decode_all_layers(
    const std::vector<std::vector<float>>& all_results,
    const std::vector<std::pair<int, int>>& hw_list,
    const std::vector<int>& strides,
    const std::vector<std::vector<std::array<float, 2>>>& all_anchors,
    float score_thresh,
    int num_classes)
{
    const int feature_size = 5 + num_classes;  // 4 box + 1 obj + K class

    std::vector<Detection> results;

    // Decode each level
    for (size_t l = 0; l < all_results.size(); ++l) {
        const auto& result = all_results[l];
        const int h = hw_list[l].first;
        const int w = hw_list[l].second;
        const int stride = strides[l];
        const auto& anchors = all_anchors[l];
        const int na = static_cast<int>(anchors.size());

        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                for (int a = 0; a < na; ++a) {
                    const int offset = ((i * w + j) * na + a) * feature_size;

                    // Argmax over class logits
                    double max_cls = std::numeric_limits<double>::lowest();
                    int class_id = 0;
                    for (int c = 0; c < num_classes; ++c) {
                        const float s = result[offset + 5 + c];
                        if (s > max_cls) { max_cls = s; class_id = c; }
                    }

                    // Confidence = sigmoid(obj) * sigmoid(max_class)
                    const float obj = sigmoid(result[offset + 4]);
                    const float cls = sigmoid(static_cast<float>(max_cls));
                    const float conf = obj * cls;
                    if (conf < score_thresh) continue;

                    // Decode (YOLOv5): center & size
                    const float dx = sigmoid(result[offset + 0]);
                    const float dy = sigmoid(result[offset + 1]);
                    const float dw = sigmoid(result[offset + 2]);
                    const float dh = sigmoid(result[offset + 3]);

                    const float aw = anchors[a][0], ah = anchors[a][1];
                    const float cx = (dx * 2.f - 0.5f + j) * stride;
                    const float cy = (dy * 2.f - 0.5f + i) * stride;
                    const float bw = std::pow(dw * 2.f, 2.f) * aw;
                    const float bh = std::pow(dh * 2.f, 2.f) * ah;

                    // xywh -> xyxy
                    const float x1 = cx - 0.5f * bw;
                    const float y1 = cy - 0.5f * bh;
                    const float x2 = cx + 0.5f * bw;
                    const float y2 = cy + 0.5f * bh;

                    results.push_back({x1, y1, x2, y2, conf, class_id});
                }
            }
        }
    }
    return results;
}

/**
 * @brief Compute IoU between two detections (xyxy).
 * @param[in] a  First detection.
 * @param[in] b  Second detection.
 * @return float Intersection-over-Union in [0,1].
 */
float iou(const Detection& a, const Detection& b) {
    float xx1 = std::max(a.bbox[0], b.bbox[0]); // x1
    float yy1 = std::max(a.bbox[1], b.bbox[1]); // y1
    float xx2 = std::min(a.bbox[2], b.bbox[2]); // x2
    float yy2 = std::min(a.bbox[3], b.bbox[3]); // y2

    float inter_w = std::max(0.0f, xx2 - xx1);
    float inter_h = std::max(0.0f, yy2 - yy1);
    float inter_area = inter_w * inter_h;

    float area_a = (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1]);
    float area_b = (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]);
    float union_area = area_a + area_b - inter_area;

    return inter_area / (union_area + 1e-9f); // avoid div-by-zero
}

/**
 * @brief Class-wise Non-Maximum Suppression (NMS) on a set of detections.
 * @param[in] detections  Input detections (xyxy + score + class_id).
 * @param[in] iou_thresh  IoU threshold to suppress.
 * @return std::vector<Detection>  Kept detections after NMS.
 */
std::vector<Detection> nms_bboxes(
    const std::vector<Detection>& detections,
    float iou_thresh)
{
    std::vector<Detection> result;
    std::unordered_map<int, std::vector<Detection>> class_map;

    // Group by class to perform class-wise NMS
    for (const auto& det : detections) {
        class_map[det.class_id].push_back(det);
    }

    for (auto& kv : class_map) {
        std::vector<Detection>& dets = kv.second;

        // Sort descending by score
        std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) {
            return a.score > b.score;
        });

        std::vector<bool> suppressed(dets.size(), false);

        for (size_t i = 0; i < dets.size(); ++i) {
            if (suppressed[i]) continue;

            result.push_back(dets[i]); // keep

            for (size_t j = i + 1; j < dets.size(); ++j) {
                if (suppressed[j]) continue;
                if (iou(dets[i], dets[j]) > iou_thresh)
                    suppressed[j] = true; // suppress overlapping lower-score box
            }
        }
    }

    return result;
}

/**
 * @brief Map letterboxed model coordinates back to original image space (for boxes).
 * @param[in,out] dets      Detections to be rescaled in-place.
 * @param[in]     img_w     Original image width.
 * @param[in]     img_h     Original image height.
 * @param[in]     input_w   Network input width (letterboxed canvas).
 * @param[in]     input_h   Network input height (letterboxed canvas).
 */
void scale_letterbox_bboxes_back(std::vector<Detection>& dets,
                                  int img_w, int img_h,
                                  int input_w, int input_h) {
    // Compute scale and padding used by letterbox
    float scale = std::min(static_cast<float>(input_w) / img_w,
                           static_cast<float>(input_h) / img_h);
    float pad_w = (input_w - img_w * scale) / 2.0f;
    float pad_h = (input_h - img_h * scale) / 2.0f;

    for (auto& det : dets) {
        // Remove padding and inverse-scale
        det.bbox[0] = (det.bbox[0] - pad_w) / scale; // x1
        det.bbox[1] = (det.bbox[1] - pad_h) / scale; // y1
        det.bbox[2] = (det.bbox[2] - pad_w) / scale; // x2
        det.bbox[3] = (det.bbox[3] - pad_h) / scale; // y2

        // Clamp to image bounds
        det.bbox[0] = std::clamp(det.bbox[0], 0.0f, static_cast<float>(img_w));
        det.bbox[1] = std::clamp(det.bbox[1], 0.0f, static_cast<float>(img_h));
        det.bbox[2] = std::clamp(det.bbox[2], 0.0f, static_cast<float>(img_w));
        det.bbox[3] = std::clamp(det.bbox[3], 0.0f, static_cast<float>(img_h));
    }
}

/**
 * @brief Map letterboxed model coordinates back to original image space (for keypoints).
 * @param[in,out] kpts      2D keypoints per detection; rescaled in-place.
 * @param[in]     img_w     Original image width.
 * @param[in]     img_h     Original image height.
 * @param[in]     input_w   Network input width (letterboxed canvas).
 * @param[in]     input_h   Network input height (letterboxed canvas).
 */
void scale_keypoints_back_letterbox(
    std::vector<std::vector<Keypoint>>& kpts,
    int img_w, int img_h,
    int input_w, int input_h)
{
    // Letterbox parameters
    float scale = std::min(static_cast<float>(input_w) / img_w,
                           static_cast<float>(input_h) / img_h);
    float pad_w = (input_w - img_w * scale) / 2.0f;
    float pad_h = (input_h - img_h * scale) / 2.0f;

    for (auto& det_kpts : kpts) {
        for (auto& kp : det_kpts) {
            // Remove padding and inverse-scale
            kp.x = (kp.x - pad_w) / scale;
            kp.y = (kp.y - pad_h) / scale;

            // Clamp to image bounds
            kp.x = std::clamp(kp.x, 0.0f, static_cast<float>(img_w));
            kp.y = std::clamp(kp.y, 0.0f, static_cast<float>(img_h));
        }
    }
}
