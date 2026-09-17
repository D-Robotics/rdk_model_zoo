// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file yoloe26seg.cpp
 * @brief Implement the YOLOE-26 staged UCP inference pipeline.
 */

#include "yoloe26seg.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#ifndef YOLOE26_SOURCE_DIR
#define YOLOE26_SOURCE_DIR "."
#endif

namespace yoloe26 {

namespace {

constexpr int kModelWidth = 640;
constexpr int kModelHeight = 640;
constexpr int kClasses = 4585;
constexpr int kMaskChannels = 32;
constexpr int kMaskSize = 160;
constexpr std::array<int, 3> kStrides{8, 16, 32};

/** @brief Internal geometry shared by preprocessing and mask restoration. */
struct LetterboxTransform {
    double gain{1.0};
    int width{0};
    int height{0};
    int left{0};
    int top{0};
    int input_width{kModelWidth};
    int input_height{kModelHeight};
};

/** @brief Internal tensor view used by the static raw-v1 decoder. */
struct Tensor {
    const float* data{nullptr};
    int h{0};
    int w{0};
    int channels{0};

    float at(int anchor, int channel) const {
        return data[static_cast<size_t>(anchor) * channels + channel];
    }
};

/** @brief Internal decoded candidate retaining its 32 mask coefficients. */
struct RawDetection {
    std::array<float, 4> box{};
    float score{0.0f};
    int label{0};
    std::array<float, kMaskChannels> coefficients{};
};

void check_runtime(int code, const char* operation) {
    if (code != 0) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 std::to_string(code));
    }
}

std::string normalized_word(const char* path) {
    std::ifstream stream(path);
    std::string value;
    std::getline(stream, value);
    std::string result;
    for (unsigned char c : value) {
        if (std::isalnum(c)) result.push_back(static_cast<char>(std::tolower(c)));
    }
    return result;
}

struct BoardInfo {
    std::string soc;
    std::string board;
    std::string march;
    bool known{false};
};

BoardInfo board_info() {
    BoardInfo result;
    result.soc = normalized_word("/sys/class/boardinfo/soc_name");
    result.board = normalized_word("/sys/class/boardinfo/board_type");
    if (result.soc != "s100" && result.soc != "s100p") return result;

    const bool is_plus = result.soc == "s100p" || result.board == "p" ||
                         result.board == "nashm" ||
                         result.board.rfind("s100p", 0) == 0 ||
                         result.board.rfind("rdks100p", 0) == 0;
    result.march = is_plus ? "nash-m" : "nash-e";
    result.known = true;
    return result;
}

std::string canonical_suffix(const std::string& march) {
    return march == "nash-m" ? "_nashm_640x640_nv12.hbm"
                              : "_nashe_640x640_nv12.hbm";
}

bool has_suffix(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

LetterboxTransform make_transform(int source_width, int source_height,
                                  int input_width, int input_height) {
    if (source_width <= 0 || source_height <= 0 || input_width <= 0 ||
        input_height <= 0) {
        throw std::invalid_argument("Image and model dimensions must be positive");
    }
    LetterboxTransform transform;
    transform.input_width = input_width;
    transform.input_height = input_height;
    transform.gain = std::min(static_cast<double>(input_width) / source_width,
                              static_cast<double>(input_height) / source_height);
    transform.width = std::max(
        1, static_cast<int>(std::nearbyint(source_width * transform.gain)));
    transform.height = std::max(
        1, static_cast<int>(std::nearbyint(source_height * transform.gain)));
    transform.width = std::min(transform.width, input_width);
    transform.height = std::min(transform.height, input_height);
    transform.left = (input_width - transform.width) / 2;
    transform.top = (input_height - transform.height) / 2;
    return transform;
}

/**
 * @brief Decode static YOLOE-26 raw-v1 outputs without IoU suppression.
 *
 * This is deliberately kept close to the released protocol implementation:
 * per-scale top-K selection happens before optional multi-label expansion, and
 * tie ordering is deterministic across scales and anchors.
 */
std::vector<RawDetection> decode(const std::vector<Tensor>& tensors,
                                 float threshold,
                                 int max_det,
                                 bool single_label) {
    if (!(threshold > 0.0f && threshold < 1.0f) || max_det < 1 ||
        max_det > 8400 || tensors.size() != 10) {
        throw std::invalid_argument("Invalid threshold, max_det, or output count");
    }
    for (int i = 0; i < 10; ++i) {
        const auto& tensor = tensors[i];
        const int hw = i == 9 ? kMaskSize : kModelWidth / kStrides[i / 3];
        const int channels = i == 9
                                 ? kMaskChannels
                                 : (i % 3 == 0 ? kClasses
                                               : (i % 3 == 1 ? 4 : kMaskChannels));
        if (!tensor.data || tensor.h != hw || tensor.w != hw ||
            tensor.channels != channels) {
            throw std::invalid_argument("Invalid raw-v1 output shape");
        }
        const size_t count = static_cast<size_t>(hw) * hw * channels;
        for (size_t j = 0; j < count; ++j) {
            if (!std::isfinite(tensor.data[j])) {
                throw std::invalid_argument("Non-finite model output");
            }
        }
    }

    struct Candidate {
        float value;
        int scale;
        int anchor;
        int label;
        int rank{0};
    };

    std::vector<Candidate> anchors;
    anchors.reserve(8400);
    for (int scale = 0; scale < 3; ++scale) {
        const auto& cls = tensors[scale * 3];
        for (int anchor = 0; anchor < cls.h * cls.w; ++anchor) {
            int label = 0;
            for (int c = 1; c < kClasses; ++c) {
                if (cls.at(anchor, c) > cls.at(anchor, label)) label = c;
            }
            anchors.push_back({cls.at(anchor, label), scale, anchor, label});
        }
    }

    auto order = [](const Candidate& a, const Candidate& b) {
        if (a.value != b.value) return a.value > b.value;
        if (a.scale != b.scale) return a.scale < b.scale;
        if (a.anchor != b.anchor) return a.anchor < b.anchor;
        return a.label < b.label;
    };
    const size_t anchor_count = std::min<size_t>(max_det, anchors.size());
    std::partial_sort(anchors.begin(), anchors.begin() + anchor_count,
                      anchors.end(), order);
    anchors.resize(anchor_count);

    if (!single_label) {
        std::vector<Candidate> classes;
        classes.reserve(anchor_count * kClasses);
        for (int rank = 0; rank < static_cast<int>(anchor_count); ++rank) {
            const auto& anchor = anchors[rank];
            for (int c = 0; c < kClasses; ++c) {
                classes.push_back({tensors[3 * anchor.scale].at(anchor.anchor, c),
                                   anchor.scale, anchor.anchor, c, rank});
            }
        }
        const size_t class_count = std::min<size_t>(max_det, classes.size());
        std::partial_sort(classes.begin(), classes.begin() + class_count,
                          classes.end(), [](const Candidate& a, const Candidate& b) {
                              if (a.value != b.value) return a.value > b.value;
                              if (a.rank != b.rank) return a.rank < b.rank;
                              return a.label < b.label;
                          });
        classes.resize(class_count);
        anchors = std::move(classes);
    }

    const float raw_threshold = std::log(threshold / (1.0f - threshold));
    std::vector<RawDetection> result;
    result.reserve(anchors.size());
    for (const auto& candidate : anchors) {
        if (candidate.value <= raw_threshold) continue;
        const int stride = kStrides[candidate.scale];
        const int grid = kModelWidth / stride;
        const float x = candidate.anchor % grid + 0.5f;
        const float y = candidate.anchor / grid + 0.5f;
        const auto& box = tensors[3 * candidate.scale + 1];
        RawDetection detection;
        detection.box = {(x - box.at(candidate.anchor, 0)) * stride,
                         (y - box.at(candidate.anchor, 1)) * stride,
                         (x + box.at(candidate.anchor, 2)) * stride,
                         (y + box.at(candidate.anchor, 3)) * stride};
        detection.score = 1.0f / (1.0f +
                                  std::exp(-std::clamp(candidate.value, -80.0f, 80.0f)));
        detection.label = candidate.label;
        for (int c = 0; c < kMaskChannels; ++c) {
            detection.coefficients[c] =
                tensors[3 * candidate.scale + 2].at(candidate.anchor, c);
        }
        result.push_back(detection);
    }
    return result;
}

int tensor_item_size(const hbDNNTensorProperties& properties) {
    switch (properties.tensorType) {
        case HB_DNN_TENSOR_TYPE_F32:
        case HB_DNN_TENSOR_TYPE_S32:
            return 4;
        case HB_DNN_TENSOR_TYPE_S16:
        case HB_DNN_TENSOR_TYPE_U16:
            return 2;
        case HB_DNN_TENSOR_TYPE_S8:
        case HB_DNN_TENSOR_TYPE_U8:
            return 1;
        default:
            return 0;
    }
}

void allocate_tensor(hbDNNTensor& tensor, int item_size) {
    auto& properties = tensor.properties;
    if (properties.validShape.numDimensions != 4 || item_size <= 0) {
        throw std::runtime_error("Unsupported tensor rank or type");
    }
    for (int dim = 3; dim >= 0; --dim) {
        if (properties.stride[dim] == -1) {
            properties.stride[dim] =
                dim == 3 ? item_size
                         : properties.stride[dim + 1] *
                               properties.validShape.dimensionSize[dim + 1];
            if (dim < 2) properties.stride[dim] =
                (properties.stride[dim] + 31) / 32 * 32;
        }
        if (properties.stride[dim] <= 0) {
            throw std::runtime_error("Unsupported tensor stride");
        }
    }
    size_t bytes = static_cast<size_t>(properties.stride[0]) *
                   properties.validShape.dimensionSize[0];
    if (properties.alignedByteSize > 0) {
        bytes = std::max(bytes, static_cast<size_t>(properties.alignedByteSize));
    }
    check_runtime(hbUCPMallocCached(&tensor.sysMem, bytes, 0),
                  "allocate tensor");
    std::memset(tensor.sysMem.virAddr, 0, bytes);
}

void release_tensors(std::vector<hbDNNTensor>& tensors) noexcept {
    for (auto& tensor : tensors) {
        if (tensor.sysMem.virAddr) {
            hbUCPFree(&tensor.sysMem);
            tensor.sysMem.virAddr = nullptr;
        }
    }
    tensors.clear();
}

std::vector<float> unpack(const hbDNNTensor& tensor) {
    const auto& properties = tensor.properties;
    if (!tensor.sysMem.virAddr || properties.validShape.numDimensions != 4) {
        throw std::runtime_error("Output tensor has no valid memory or shape");
    }
    const int h = properties.validShape.dimensionSize[1];
    const int w = properties.validShape.dimensionSize[2];
    const int c = properties.validShape.dimensionSize[3];
    if (h <= 0 || w <= 0 || c <= 0) throw std::runtime_error("Invalid output shape");

    std::vector<float> values(static_cast<size_t>(h) * w * c);
    const auto* data = static_cast<const unsigned char*>(tensor.sysMem.virAddr);
    const bool quantized = properties.tensorType != HB_DNN_TENSOR_TYPE_F32;
    int axis = properties.quantizeAxis;
    if (axis < 0) axis += 4;
    if (properties.scale.scaleLen <= 1 && properties.scale.zeroPointLen <= 1) axis = 0;
    if (quantized && (properties.scale.scaleLen < 1 || !properties.scale.scaleData ||
                      axis < 0 || axis > 3)) {
        throw std::runtime_error("Invalid output quantization metadata");
    }
    if (quantized &&
        ((properties.scale.scaleLen != 1 &&
          properties.scale.scaleLen != properties.validShape.dimensionSize[axis]) ||
         (properties.scale.zeroPointLen > 1 &&
          properties.scale.zeroPointLen != properties.validShape.dimensionSize[axis]) ||
         (properties.scale.zeroPointLen > 0 && !properties.scale.zeroPointData))) {
        throw std::runtime_error("Quantization length mismatch");
    }

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int z = 0; z < c; ++z) {
                const auto* address = data + y * properties.stride[1] +
                                      x * properties.stride[2] +
                                      z * properties.stride[3];
                float value = 0.0f;
                switch (properties.tensorType) {
                    case HB_DNN_TENSOR_TYPE_F32:
                        std::memcpy(&value, address, sizeof(value));
                        break;
                    case HB_DNN_TENSOR_TYPE_S32: {
                        int32_t v;
                        std::memcpy(&v, address, sizeof(v));
                        value = static_cast<float>(v);
                        break;
                    }
                    case HB_DNN_TENSOR_TYPE_S16: {
                        int16_t v;
                        std::memcpy(&v, address, sizeof(v));
                        value = static_cast<float>(v);
                        break;
                    }
                    case HB_DNN_TENSOR_TYPE_U16: {
                        uint16_t v;
                        std::memcpy(&v, address, sizeof(v));
                        value = static_cast<float>(v);
                        break;
                    }
                    case HB_DNN_TENSOR_TYPE_S8: {
                        int8_t v;
                        std::memcpy(&v, address, sizeof(v));
                        value = static_cast<float>(v);
                        break;
                    }
                    case HB_DNN_TENSOR_TYPE_U8:
                        value = *address;
                        break;
                    default:
                        throw std::runtime_error("Unsupported output dtype");
                }
                if (quantized) {
                    const int coordinates[4] = {0, y, x, z};
                    const int scale_index = properties.scale.scaleLen == 1
                                                ? 0
                                                : coordinates[axis];
                    const float scale = properties.scale.scaleData[scale_index];
                    if (!(scale > 0.0f) || !std::isfinite(scale)) {
                        throw std::runtime_error("Invalid quantization scale");
                    }
                    const int zero_index = properties.scale.zeroPointLen == 1
                                               ? 0
                                               : coordinates[axis];
                    const float zero = properties.scale.zeroPointLen == 0
                                           ? 0.0f
                                           : properties.scale.zeroPointData[zero_index];
                    value = (value - zero) * scale;
                }
                values[(static_cast<size_t>(y) * w + x) * c + z] = value;
            }
        }
    }
    return values;
}

/** @brief Convert one raw model box from canvas to clipped source coordinates. */
std::array<float, 4> restore_box(const std::array<float, 4>& box,
                                 const LetterboxTransform& transform,
                                 int source_width,
                                 int source_height) {
    return {std::clamp(static_cast<float>((box[0] - transform.left) /
                                           transform.gain),
                       0.0f, static_cast<float>(source_width)),
            std::clamp(static_cast<float>((box[1] - transform.top) /
                                           transform.gain),
                       0.0f, static_cast<float>(source_height)),
            std::clamp(static_cast<float>((box[2] - transform.left) /
                                           transform.gain),
                       0.0f, static_cast<float>(source_width)),
            std::clamp(static_cast<float>((box[3] - transform.top) /
                                           transform.gain),
                       0.0f, static_cast<float>(source_height))};
}

/**
 * @brief Restore one binary mask and crop it to its source-image ROI.
 *
 * The order matches the Python runtime: threshold and crop in model canvas,
 * remove letterbox padding, resize to the full source image, then use clipped
 * integer-truncated source coordinates for the local ROI.
 */
cv::Mat restore_mask(const cv::Mat& raw_mask,
                    const std::array<float, 4>& model_box,
                    const LetterboxTransform& transform,
                    const std::array<float, 4>& source_box,
                    int source_width,
                    int source_height) {
    cv::Mat full_canvas;
    cv::resize(raw_mask, full_canvas,
               cv::Size(transform.input_width, transform.input_height), 0, 0,
               cv::INTER_LINEAR);
    cv::Mat binary = full_canvas > 0.0f;
    for (int y = 0; y < binary.rows; ++y) {
        auto* row = binary.ptr<unsigned char>(y);
        for (int x = 0; x < binary.cols; ++x) row[x] = row[x] ? 1 : 0;
    }

    const int canvas_left = std::clamp(transform.left, 0, transform.input_width);
    const int canvas_top = std::clamp(transform.top, 0, transform.input_height);
    const int canvas_width = std::clamp(transform.width, 0,
                                        transform.input_width - canvas_left);
    const int canvas_height = std::clamp(transform.height, 0,
                                         transform.input_height - canvas_top);
    if (canvas_width <= 0 || canvas_height <= 0) return cv::Mat();

    for (int y = 0; y < binary.rows; ++y) {
        auto* row = binary.ptr<unsigned char>(y);
        for (int x = 0; x < binary.cols; ++x) {
            if (x < model_box[0] || x >= model_box[2] || y < model_box[1] ||
                y >= model_box[3]) {
                row[x] = 0;
            }
        }
    }

    const cv::Mat unpadded = binary(
        cv::Rect(canvas_left, canvas_top, canvas_width, canvas_height));
    cv::Mat restored;
    cv::resize(unpadded, restored, cv::Size(source_width, source_height), 0, 0,
               cv::INTER_NEAREST);
    for (int y = 0; y < restored.rows; ++y) {
        auto* row = restored.ptr<unsigned char>(y);
        for (int x = 0; x < restored.cols; ++x) row[x] = row[x] ? 1 : 0;
    }

    // Clip first, then truncate. This is intentionally equivalent to int(box)
    // in the shared Python visualization contract.
    const int x1 = static_cast<int>(std::clamp(source_box[0], 0.0f,
                                              static_cast<float>(source_width)));
    const int y1 = static_cast<int>(std::clamp(source_box[1], 0.0f,
                                              static_cast<float>(source_height)));
    const int x2 = static_cast<int>(std::clamp(source_box[2], 0.0f,
                                              static_cast<float>(source_width)));
    const int y2 = static_cast<int>(std::clamp(source_box[3], 0.0f,
                                              static_cast<float>(source_height)));
    if (x2 <= x1 || y2 <= y1) return cv::Mat();
    return restored(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
}

void validate_output_tensors(const std::vector<hbDNNTensor>& output_tensors) {
    if (output_tensors.size() != 10) {
        throw std::invalid_argument("Expected ten YOLOE-26 output tensors");
    }
    for (int i = 0; i < 10; ++i) {
        const auto& tensor = output_tensors[i];
        const auto& shape = tensor.properties.validShape;
        const int hw = i == 9 ? kMaskSize : kModelWidth / kStrides[i / 3];
        const int channels = i == 9
                                 ? kMaskChannels
                                 : (i % 3 == 0 ? kClasses
                                               : (i % 3 == 1 ? 4 : kMaskChannels));
        if (shape.numDimensions != 4 || shape.dimensionSize[0] != 1 ||
            shape.dimensionSize[1] != hw || shape.dimensionSize[2] != hw ||
            shape.dimensionSize[3] != channels || tensor_item_size(tensor.properties) == 0 ||
            !tensor.sysMem.virAddr) {
            throw std::invalid_argument("Invalid raw-v1 output tensor shape or memory");
        }
        for (int dim = 0; dim < 4; ++dim) {
            if (tensor.properties.stride[dim] <= 0) {
                throw std::invalid_argument("Invalid raw-v1 output tensor stride");
            }
        }
    }
}

}  // namespace

int32_t pre_process(std::vector<hbDNNTensor>& input_tensors,
                    const cv::Mat& image,
                    int input_w,
                    int input_h,
                    const std::string& image_format) {
    if (image_format != "BGR") return -1;
    if (image.empty() || image.type() != CV_8UC3 || input_w <= 0 ||
        input_h <= 0 || input_w % 2 != 0 || input_h % 2 != 0) {
        return -2;
    }
    if (input_tensors.size() != 2 || !input_tensors[0].sysMem.virAddr ||
        !input_tensors[1].sysMem.virAddr) {
        return -3;
    }

    try {
        const LetterboxTransform transform =
            make_transform(image.cols, image.rows, input_w, input_h);
        cv::Mat resized;
        cv::resize(image, resized,
                   cv::Size(transform.width, transform.height), 0, 0,
                   cv::INTER_LINEAR);
        cv::Mat padded;
        cv::copyMakeBorder(resized, padded, transform.top,
                           input_h - transform.height - transform.top,
                           transform.left,
                           input_w - transform.width - transform.left,
                           cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

        cv::Mat i420;
        cv::cvtColor(padded, i420, cv::COLOR_BGR2YUV_I420);
        const auto* source = i420.ptr<unsigned char>();
        auto* y_plane = static_cast<unsigned char*>(input_tensors[0].sysMem.virAddr);
        auto* uv_plane = static_cast<unsigned char*>(input_tensors[1].sysMem.virAddr);
        const auto& y_props = input_tensors[0].properties;
        const auto& uv_props = input_tensors[1].properties;
        if (y_props.validShape.dimensionSize[1] != input_h ||
            y_props.validShape.dimensionSize[2] != input_w ||
            uv_props.validShape.dimensionSize[1] != input_h / 2 ||
            uv_props.validShape.dimensionSize[2] != input_w / 2) {
            return -4;
        }

        for (int row = 0; row < input_h; ++row) {
            for (int col = 0; col < input_w; ++col) {
                y_plane[row * y_props.stride[1] + col * y_props.stride[2]] =
                    source[row * input_w + col];
            }
        }
        const int uv_offset = input_h * input_w;
        const int v_offset = uv_offset + (input_h / 2) * (input_w / 2);
        for (int row = 0; row < input_h / 2; ++row) {
            for (int col = 0; col < input_w / 2; ++col) {
                const auto offset = row * uv_props.stride[1] +
                                    col * uv_props.stride[2];
                uv_plane[offset] = source[uv_offset + row * (input_w / 2) + col];
                uv_plane[offset + uv_props.stride[3]] =
                    source[v_offset + row * (input_w / 2) + col];
            }
        }
        check_runtime(hbUCPMemFlush(&input_tensors[0].sysMem,
                                     HB_SYS_MEM_CACHE_CLEAN),
                      "flush Y input");
        check_runtime(hbUCPMemFlush(&input_tensors[1].sysMem,
                                     HB_SYS_MEM_CACHE_CLEAN),
                      "flush UV input");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "pre_process: " << error.what() << '\n';
        return -5;
    }
}

int32_t infer(std::vector<hbDNNTensor>& output_tensors,
              std::vector<hbDNNTensor>& input_tensors,
              hbDNNHandle_t dnn_handle,
              hbUCPSchedParam* sched_param) {
    if (!dnn_handle || input_tensors.empty() || output_tensors.empty()) return -1;

    for (auto& tensor : input_tensors) {
        const int code = hbUCPMemFlush(&tensor.sysMem, HB_SYS_MEM_CACHE_CLEAN);
        if (code != 0) return code;
    }

    if (!sched_param) {
        // nullptr is the documented synchronous path for a single inference.
        const int code = hbDNNInferV2(nullptr, output_tensors.data(),
                                      input_tensors.data(), dnn_handle);
        if (code != 0) return code;
    } else {
        hbUCPTaskHandle_t task{nullptr};
        int code = hbDNNInferV2(&task, output_tensors.data(),
                                input_tensors.data(), dnn_handle);
        if (code != 0) return code;
        if (!task) return -2;
        code = hbUCPSubmitTask(task, sched_param);
        if (code == 0) code = hbUCPWaitTaskDone(task, 0);
        const int release_code = hbUCPReleaseTask(task);
        if (code != 0) return code;
        if (release_code != 0) return release_code;
    }

    for (auto& tensor : output_tensors) {
        const int code = hbUCPMemFlush(&tensor.sysMem,
                                       HB_SYS_MEM_CACHE_INVALIDATE);
        if (code != 0) return code;
    }
    return 0;
}

static InstanceSegResult post_process_with_transform(
    const std::vector<hbDNNTensor>& output_tensors,
    const YoloE26SegConfig& config,
    int source_width,
    int source_height,
    const LetterboxTransform& transform) {
    if (source_width <= 0 || source_height <= 0 || transform.gain <= 0.0 ||
        transform.input_width <= 0 || transform.input_height <= 0 ||
        transform.width <= 0 || transform.height <= 0) {
        throw std::invalid_argument("Invalid source image or letterbox transform");
    }
    validate_output_tensors(output_tensors);

    std::vector<std::vector<float>> buffers;
    buffers.reserve(output_tensors.size());
    for (const auto& output : output_tensors) buffers.push_back(unpack(output));

    std::vector<Tensor> tensors;
    tensors.reserve(buffers.size());
    for (int i = 0; i < 10; ++i) {
        const int hw = i == 9 ? kMaskSize : kModelWidth / kStrides[i / 3];
        const int channels = i == 9
                                 ? kMaskChannels
                                 : (i % 3 == 0 ? kClasses
                                               : (i % 3 == 1 ? 4 : kMaskChannels));
        tensors.push_back({buffers[i].data(), hw, hw, channels});
    }

    const std::vector<RawDetection> decoded =
        decode(tensors, config.score_threshold, config.max_det,
               config.single_label);
    InstanceSegResult result;
    result.detections.reserve(decoded.size());
    result.masks.reserve(decoded.size());

    for (const auto& raw_detection : decoded) {
        cv::Mat raw_mask(kMaskSize, kMaskSize, CV_32F, cv::Scalar(0));
        for (int y = 0; y < kMaskSize; ++y) {
            float* row = raw_mask.ptr<float>(y);
            for (int x = 0; x < kMaskSize; ++x) {
                float value = 0.0f;
                const float* proto = tensors[9].data +
                                     (static_cast<size_t>(y) * kMaskSize + x) *
                                         kMaskChannels;
                for (int c = 0; c < kMaskChannels; ++c) {
                    value += proto[c] * raw_detection.coefficients[c];
                }
                row[x] = value;
            }
        }

        const auto source_box = restore_box(raw_detection.box, transform,
                                            source_width, source_height);
        result.detections.push_back(
            Detection{{source_box[0], source_box[1], source_box[2], source_box[3]},
                      raw_detection.score, raw_detection.label});
        result.masks.push_back(restore_mask(raw_mask, raw_detection.box,
                                            transform, source_box, source_width,
                                            source_height));
    }
    return result;
}

InstanceSegResult post_process(const std::vector<hbDNNTensor>& output_tensors,
                               const YoloE26SegConfig& config,
                               int source_width,
                               int source_height,
                               int input_width,
                               int input_height) {
    return post_process_with_transform(
        output_tensors, config, source_width, source_height,
        make_transform(source_width, source_height, input_width, input_height));
}

YoloE26Seg::YoloE26Seg(YoloE26SegConfig config) : config_(std::move(config)) {}

YoloE26Seg::~YoloE26Seg() { release(); }

void YoloE26Seg::release() noexcept {
    release_tensors(input_tensors);
    release_tensors(output_tensors);
    if (packed_dnn_handle_) {
        hbDNNRelease(packed_dnn_handle_);
        packed_dnn_handle_ = nullptr;
    }
    dnn_handle = nullptr;
    input_count_ = 0;
    output_count_ = 0;
    input_h = 0;
    input_w = 0;
    inited_ = false;
}

int32_t YoloE26Seg::init(const char* requested_model_path) noexcept {
    if (inited_) return -1;
    release();

    try {
        const std::string model_size = config_.model_size;
        if (model_size != "n" && model_size != "s" && model_size != "m" &&
            model_size != "l" && model_size != "x") {
            throw std::invalid_argument("model_size must be one of n/s/m/l/x");
        }
        const std::string path = requested_model_path && requested_model_path[0]
                                     ? requested_model_path
                                     : (config_.model_path.empty()
                                            ? default_model_path(model_size)
                                            : config_.model_path);
        if (path.empty()) throw std::invalid_argument("model_path is empty");

        const BoardInfo board = board_info();
        if (!board.known) {
            throw std::invalid_argument("Only S100 and S100P boards are supported");
        }
        const std::string name = std::filesystem::path(path).filename().string();
        if (!has_suffix(name, "_nashe_640x640_nv12.hbm") &&
            !has_suffix(name, "_nashm_640x640_nv12.hbm")) {
            throw std::invalid_argument("Use a canonical YOLOE-26 HBM filename");
        }
        if (board.known && !has_suffix(name, canonical_suffix(board.march))) {
            throw std::invalid_argument(
                "HBM filename does not match this board; expected " +
                canonical_suffix(board.march));
        }

        const char* model_files[] = {path.c_str()};
        check_runtime(hbDNNInitializeFromFiles(&packed_dnn_handle_, model_files, 1),
                      "load HBM");
        const char** model_names = nullptr;
        int model_count = 0;
        check_runtime(hbDNNGetModelNameList(&model_names, &model_count,
                                            packed_dnn_handle_),
                      "model names");
        if (!model_names || model_count != 1) {
            throw std::runtime_error("Expected one model in HBM");
        }
        check_runtime(hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle_,
                                           model_names[0]),
                      "model handle");
        check_runtime(hbDNNGetInputCount(&input_count_, dnn_handle),
                      "input count");
        check_runtime(hbDNNGetOutputCount(&output_count_, dnn_handle),
                      "output count");
        if (input_count_ != 2 || output_count_ != 10) {
            throw std::runtime_error("Expected 2 NV12 inputs and 10 outputs");
        }

        input_tensors.resize(input_count_);
        output_tensors.resize(output_count_);
        for (int i = 0; i < input_count_; ++i) {
            auto& tensor = input_tensors[i];
            check_runtime(hbDNNGetInputTensorProperties(&tensor.properties,
                                                        dnn_handle, i),
                          "input properties");
            const auto& shape = tensor.properties.validShape;
            const std::array<int, 4> expected =
                i == 0 ? std::array<int, 4>{1, kModelHeight, kModelWidth, 1}
                       : std::array<int, 4>{1, kModelHeight / 2, kModelWidth / 2, 2};
            if (shape.numDimensions != 4 ||
                tensor.properties.tensorType != HB_DNN_TENSOR_TYPE_U8) {
                throw std::runtime_error("Expected uint8 NHWC NV12 inputs");
            }
            for (int d = 0; d < 4; ++d) {
                if (shape.dimensionSize[d] != expected[d]) {
                    throw std::runtime_error("Invalid NV12 input shape/order");
                }
            }
            allocate_tensor(tensor, 1);
        }
        for (int i = 0; i < output_count_; ++i) {
            auto& tensor = output_tensors[i];
            check_runtime(hbDNNGetOutputTensorProperties(&tensor.properties,
                                                         dnn_handle, i),
                          "output properties");
            if (tensor.properties.tensorType != HB_DNN_TENSOR_TYPE_F32 &&
                tensor.properties.quantiType != SCALE) {
                throw std::runtime_error(
                    "Integer output requires SCALE quantization metadata");
            }
            const int hw = i == 9 ? kMaskSize : kModelWidth / kStrides[i / 3];
            const int channels = i == 9
                                     ? kMaskChannels
                                     : (i % 3 == 0 ? kClasses
                                                   : (i % 3 == 1 ? 4 : kMaskChannels));
            const auto& shape = tensor.properties.validShape;
            if (shape.numDimensions != 4 || shape.dimensionSize[0] != 1 ||
                shape.dimensionSize[1] != hw || shape.dimensionSize[2] != hw ||
                shape.dimensionSize[3] != channels) {
                throw std::runtime_error("Invalid raw-v1 output shape/order");
            }
            const int item_size = tensor_item_size(tensor.properties);
            allocate_tensor(tensor, item_size);
        }
        input_h = kModelHeight;
        input_w = kModelWidth;
        inited_ = true;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "YoloE26Seg::init: " << error.what() << '\n';
        release();
        return -1;
    } catch (...) {
        std::cerr << "YoloE26Seg::init: unknown error\n";
        release();
        return -1;
    }
}

InstanceSegResult YoloE26Seg::predict(const cv::Mat& image) {
    if (!inited_ || !dnn_handle) {
        throw std::runtime_error("YoloE26Seg::init() must succeed before predict()");
    }
    if (image.empty() || image.type() != CV_8UC3) {
        throw std::invalid_argument("predict expects a nonempty BGR CV_8UC3 image");
    }
    if (!(config_.score_threshold > 0.0f && config_.score_threshold < 1.0f) ||
        config_.max_det < 1 || config_.max_det > 8400) {
        throw std::invalid_argument("Invalid threshold or max_det");
    }

    const int preprocess_code = pre_process(input_tensors, image, input_w, input_h);
    if (preprocess_code != 0) {
        throw std::runtime_error("pre_process failed: " +
                                 std::to_string(preprocess_code));
    }
    const int infer_code = infer(output_tensors, input_tensors, dnn_handle);
    if (infer_code != 0) {
        throw std::runtime_error("infer failed: " + std::to_string(infer_code));
    }
    return post_process(output_tensors, config_, image.cols, image.rows,
                        input_w, input_h);
}

std::string YoloE26Seg::default_model_path(const std::string& model_size) {
    if (model_size != "n" && model_size != "s" && model_size != "m" &&
        model_size != "l" && model_size != "x") {
        throw std::invalid_argument("model_size must be one of n/s/m/l/x");
    }
    const BoardInfo board = board_info();
    if (!board.known) {
        throw std::invalid_argument("Only S100 and S100P boards are supported");
    }
    const std::string march = board.march;
    const std::string stem = "yoloe_26" + model_size + "_seg_pf_" +
                             (march == "nash-m" ? "nashm" : "nashe") +
                             "_640x640_nv12.hbm";
    const std::filesystem::path source_dir(YOLOE26_SOURCE_DIR);
    const std::filesystem::path repository_model =
        source_dir / ".." / ".." / "model" / march / stem;
    std::error_code error;
    if (std::filesystem::exists(repository_model, error)) {
        return repository_model.string();
    }
    return repository_model.string();
}

}  // namespace yoloe26
