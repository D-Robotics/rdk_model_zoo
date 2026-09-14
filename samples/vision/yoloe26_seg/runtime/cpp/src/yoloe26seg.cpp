// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#include "yoloe26seg.hpp"
#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace yoloe26 {
constexpr int kClasses = 4585;
constexpr std::array<int, 3> kStrides{8, 16, 32};

struct Tensor {
    const float* data;
    int h, w, channels;
    float at(int anchor, int channel) const { return data[anchor * channels + channel]; }
};


inline std::vector<Detection> decode(const std::vector<Tensor>& tensors, float threshold = .25f,
                                     int max_det = 300, bool single_label = true) {
    if (!(threshold > 0 && threshold < 1) || max_det < 1 || max_det > 8400 || tensors.size() != 10)
        throw std::invalid_argument("Invalid threshold, max_det, or output count");
    for (int i = 0; i < 10; ++i) {
        const auto& t = tensors[i];
        const int hw = i == 9 ? 160 : 640 / kStrides[i / 3];
        const int channels = i == 9 ? 32 : (i % 3 == 0 ? kClasses : (i % 3 == 1 ? 4 : 32));
        if (!t.data || t.h != hw || t.w != hw || t.channels != channels)
            throw std::invalid_argument("Invalid raw-v1 output shape");
        for (int j = 0; j < hw * hw * channels; ++j)
            if (!std::isfinite(t.data[j])) throw std::invalid_argument("Non-finite output");
    }
    struct Candidate { float value; int scale, anchor, label; int rank = 0; };
    std::vector<Candidate> anchors;
    for (int scale = 0; scale < 3; ++scale) {
        const auto& cls = tensors[scale * 3];
        for (int anchor = 0; anchor < cls.h * cls.w; ++anchor) {
            int label = 0;
            for (int c = 1; c < kClasses; ++c)
                if (cls.at(anchor, c) > cls.at(anchor, label)) label = c;
            anchors.push_back({cls.at(anchor, label), scale, anchor, label});
        }
    }
    auto order = [](const Candidate& a, const Candidate& b) {
        if (a.value != b.value) return a.value > b.value;
        if (a.scale != b.scale) return a.scale < b.scale;
        if (a.anchor != b.anchor) return a.anchor < b.anchor;
        return a.label < b.label;
    };
    std::partial_sort(anchors.begin(), anchors.begin() + max_det, anchors.end(), order);
    anchors.resize(max_det);
    if (!single_label) {
        std::vector<Candidate> classes;
        for (int rank = 0; rank < max_det; ++rank) {
            const auto& a = anchors[rank];
            for (int c = 0; c < kClasses; ++c)
                classes.push_back({tensors[3 * a.scale].at(a.anchor, c), a.scale, a.anchor, c, rank});
        }
        std::partial_sort(classes.begin(), classes.begin() + max_det, classes.end(),
                          [](const Candidate& a, const Candidate& b) {
                              if (a.value != b.value) return a.value > b.value;
                              if (a.rank != b.rank) return a.rank < b.rank;
                              return a.label < b.label;
                          });
        classes.resize(max_det);
        anchors = std::move(classes);
    }
    const float raw_threshold = std::log(threshold / (1 - threshold));
    std::vector<Detection> result;
    for (const auto& a : anchors) {
        if (a.value <= raw_threshold) continue;
        int stride = kStrides[a.scale], grid = 640 / stride;
        float x = a.anchor % grid + .5f, y = a.anchor / grid + .5f;
        const auto& box = tensors[3 * a.scale + 1];
        Detection d{{(x - box.at(a.anchor, 0)) * stride, (y - box.at(a.anchor, 1)) * stride,
                     (x + box.at(a.anchor, 2)) * stride, (y + box.at(a.anchor, 3)) * stride},
                    1.f / (1.f + std::exp(-std::clamp(a.value, -80.f, 80.f))), a.label, {}};
        for (int c = 0; c < 32; ++c) d.coefficients[c] = tensors[3 * a.scale + 2].at(a.anchor, c);
        result.push_back(d);
    }
    return result;
}
}  // namespace yoloe26

namespace {
void check(int code, const char* operation) {
    if (code) throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(code));
}

class Runtime {
public:
    hbDNNPackedHandle_t packed = nullptr;
    hbDNNHandle_t model = nullptr;
    std::vector<hbDNNTensor> inputs, outputs;
    Runtime() = default;
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    ~Runtime() {
        for (auto& t : inputs) if (t.sysMem.virAddr) hbUCPFree(&t.sysMem);
        for (auto& t : outputs) if (t.sysMem.virAddr) hbUCPFree(&t.sysMem);
        if (packed) hbDNNRelease(packed);
    }
    void load(const char* path) {
        check(hbDNNInitializeFromFiles(&packed, &path, 1), "load HBM");
        const char** names = nullptr;
        int count = 0;
        check(hbDNNGetModelNameList(&names, &count, packed), "model names");
        if (count != 1) throw std::runtime_error("Expected one model");
        check(hbDNNGetModelHandle(&model, packed, names[0]), "model handle");
        int ni = 0, no = 0;
        check(hbDNNGetInputCount(&ni, model), "input count");
        check(hbDNNGetOutputCount(&no, model), "output count");
        if (ni != 2 || no != 10)
            throw std::runtime_error("Expected 2 NV12 inputs and 10 raw outputs");
        inputs.resize(ni);
        outputs.resize(no);
        for (int i = 0; i < ni; ++i) {
            auto& t = inputs[i];
            check(hbDNNGetInputTensorProperties(&t.properties, model, i), "input properties");
            const auto& shape = t.properties.validShape;
            const std::array<int, 4> expected = i == 0 ? std::array<int, 4>{1,640,640,1} : std::array<int, 4>{1,320,320,2};
            if (shape.numDimensions != 4 || t.properties.tensorType != HB_DNN_TENSOR_TYPE_U8)
                throw std::runtime_error("Expected uint8 NHWC NV12 inputs");
            for (int d = 0; d < 4; ++d)
                if (shape.dimensionSize[d] != expected[d]) throw std::runtime_error("Invalid NV12 input shape/order");
            allocate(t, 1);
        }
        for (int i = 0; i < no; ++i) {
            auto& t = outputs[i];
            check(hbDNNGetOutputTensorProperties(&t.properties, model, i), "output properties");
            if (t.properties.tensorType != HB_DNN_TENSOR_TYPE_F32 && t.properties.quantiType != SCALE)
                throw std::runtime_error("Integer output requires SCALE quantization metadata");
            int hw = i == 9 ? 160 : 640 / yoloe26::kStrides[i / 3];
            int c = i == 9 ? 32 : (i % 3 == 0 ? 4585 : (i % 3 == 1 ? 4 : 32));
            const auto& shape = t.properties.validShape;
            if (shape.numDimensions != 4 || shape.dimensionSize[0] != 1 ||
                shape.dimensionSize[1] != hw || shape.dimensionSize[2] != hw || shape.dimensionSize[3] != c)
                throw std::runtime_error("Invalid raw-v1 output shape/order");
            int item_size = 0;
            switch (t.properties.tensorType) {
                case HB_DNN_TENSOR_TYPE_F32: case HB_DNN_TENSOR_TYPE_S32: item_size = 4; break;
                case HB_DNN_TENSOR_TYPE_S16: case HB_DNN_TENSOR_TYPE_U16: item_size = 2; break;
                case HB_DNN_TENSOR_TYPE_S8: case HB_DNN_TENSOR_TYPE_U8: item_size = 1; break;
                default: throw std::runtime_error("Unsupported output dtype");
            }
            allocate(t, item_size);
        }
    }
    void allocate(hbDNNTensor& t, int item_size) {
        auto& p = t.properties;
        for (int d = 3; d >= 0; --d) {
            if (p.stride[d] == -1) {
                p.stride[d] = d == 3 ? item_size : p.stride[d + 1] * p.validShape.dimensionSize[d + 1];
                if (d < 2) p.stride[d] = (p.stride[d] + 31) / 32 * 32;
            }
            if (p.stride[d] <= 0) throw std::runtime_error("Unsupported tensor stride");
        }
        auto bytes = p.stride[0] * p.validShape.dimensionSize[0];
        if (p.alignedByteSize > bytes) bytes = p.alignedByteSize;
        check(hbUCPMallocCached(&t.sysMem, bytes, 0), "allocate tensor");
        std::memset(t.sysMem.virAddr, 0, bytes);
    }
    void run(const cv::Mat& padded) {
        cv::Mat i420;
        cv::cvtColor(padded, i420, cv::COLOR_BGR2YUV_I420);
        const auto* src = i420.ptr<unsigned char>();
        auto* y = static_cast<unsigned char*>(inputs[0].sysMem.virAddr);
        auto* uv = static_cast<unsigned char*>(inputs[1].sysMem.virAddr);
        for (int row = 0; row < 640; ++row)
            for (int col = 0; col < 640; ++col)
                y[row * inputs[0].properties.stride[1] + col * inputs[0].properties.stride[2]] = src[row * 640 + col];
        for (int row = 0; row < 320; ++row) for (int col = 0; col < 320; ++col) {
            auto offset = row * inputs[1].properties.stride[1] + col * inputs[1].properties.stride[2];
            uv[offset] = src[640 * 640 + row * 320 + col];
            uv[offset + inputs[1].properties.stride[3]] = src[640 * 640 * 5 / 4 + row * 320 + col];
        }
        for (auto& t : inputs) check(hbUCPMemFlush(&t.sysMem, HB_SYS_MEM_CACHE_CLEAN), "flush input");
        // Synchronous mode ensures tensor buffers are no longer in use on return.
        check(hbDNNInferV2(nullptr, outputs.data(), inputs.data(), model), "infer");
        for (auto& t : outputs) check(hbUCPMemFlush(&t.sysMem, HB_SYS_MEM_CACHE_INVALIDATE), "invalidate output");
    }
};

std::vector<float> unpack(const hbDNNTensor& t) {
    const auto& p = t.properties;
    int h = p.validShape.dimensionSize[1], w = p.validShape.dimensionSize[2], c = p.validShape.dimensionSize[3];
    std::vector<float> values(h * w * c);
    const auto* data = static_cast<const unsigned char*>(t.sysMem.virAddr);
    const bool quantized = p.tensorType != HB_DNN_TENSOR_TYPE_F32;
    int axis = p.quantizeAxis;
    if (axis < 0) axis += 4;
    if (p.scale.scaleLen <= 1 && p.scale.zeroPointLen <= 1) axis = 0;
    if (quantized && (p.scale.scaleLen < 1 || !p.scale.scaleData || axis < 0 || axis > 3))
        throw std::runtime_error("Invalid output quantization metadata");
    if (quantized && ((p.scale.scaleLen != 1 && p.scale.scaleLen != p.validShape.dimensionSize[axis]) ||
                     (p.scale.zeroPointLen > 1 && p.scale.zeroPointLen != p.validShape.dimensionSize[axis]) ||
                     (p.scale.zeroPointLen > 0 && !p.scale.zeroPointData)))
        throw std::runtime_error("Quantization length mismatch");
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) for (int z = 0; z < c; ++z) {
        const auto* address = data + y * p.stride[1] + x * p.stride[2] + z * p.stride[3];
        float value = 0;
        switch (p.tensorType) {
            case HB_DNN_TENSOR_TYPE_F32: std::memcpy(&value, address, 4); break;
            case HB_DNN_TENSOR_TYPE_S32: { int32_t v; std::memcpy(&v, address, 4); value = v; break; }
            case HB_DNN_TENSOR_TYPE_S16: { int16_t v; std::memcpy(&v, address, 2); value = v; break; }
            case HB_DNN_TENSOR_TYPE_U16: { uint16_t v; std::memcpy(&v, address, 2); value = v; break; }
            case HB_DNN_TENSOR_TYPE_S8: { int8_t v; std::memcpy(&v, address, 1); value = v; break; }
            case HB_DNN_TENSOR_TYPE_U8: value = *address; break;
            default: throw std::runtime_error("Unsupported output dtype");
        }
        if (quantized) {
            int coordinates[] = {0, y, x, z};
            float scale = p.scale.scaleData[p.scale.scaleLen == 1 ? 0 : coordinates[axis]];
            if (!(scale > 0) || !std::isfinite(scale)) throw std::runtime_error("Invalid scale");
            float zero = p.scale.zeroPointLen == 0 ? 0 : p.scale.zeroPointData[p.scale.zeroPointLen == 1 ? 0 : coordinates[axis]];
            value = (value - zero) * scale;
        }
        values[(y * w + x) * c + z] = value;
    }
    return values;
}

struct LetterboxTransform {
    double gain = 1.0;
    int width = 0;
    int height = 0;
    int left = 0;
    int top = 0;
};

using FrameResult = yoloe26::Result;


cv::Mat letterbox(const cv::Mat& image, LetterboxTransform& transform) {
    if (image.empty() || image.type() != CV_8UC3)
        throw std::runtime_error("Expected a nonempty uint8 BGR image");
    transform.gain = std::min(640.0 / image.rows, 640.0 / image.cols);
    transform.width = std::max(1, static_cast<int>(std::nearbyint(image.cols * transform.gain)));
    transform.height = std::max(1, static_cast<int>(std::nearbyint(image.rows * transform.gain)));
    transform.left = (640 - transform.width) / 2;
    transform.top = (640 - transform.height) / 2;

    cv::Mat resized, padded;
    cv::resize(image, resized, cv::Size(transform.width, transform.height), 0, 0, cv::INTER_LINEAR);
    cv::copyMakeBorder(resized, padded, transform.top, 640 - transform.height - transform.top,
                       transform.left, 640 - transform.width - transform.left,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return padded;
}

FrameResult decode_frame(Runtime& runtime, const cv::Mat& source, const LetterboxTransform& transform,
                         float score_threshold, int max_det, bool single_label) {
    std::vector<std::vector<float>> physical;
    physical.reserve(runtime.outputs.size());
    for (auto& output : runtime.outputs) physical.push_back(unpack(output));

    std::vector<std::vector<float>> buffers;
    buffers.reserve(10);
    std::vector<yoloe26::Tensor> tensors;
    tensors.reserve(10);
    buffers = std::move(physical);
    for (int i = 0; i < 10; ++i) {
        const int hw = i == 9 ? 160 : 640 / yoloe26::kStrides[i / 3];
        const int channels = i == 9 ? 32 : (i % 3 == 0 ? yoloe26::kClasses : (i % 3 == 1 ? 4 : 32));
        tensors.push_back({buffers[i].data(), hw, hw, channels});
    }

    FrameResult result;
    const auto decoded = yoloe26::decode(tensors, score_threshold, max_det, single_label);
    result.detections.reserve(decoded.size());
    result.masks.reserve(decoded.size());
    for (auto detection : decoded) {
        cv::Mat raw(160, 160, CV_32F, cv::Scalar(0));
        for (int y = 0; y < 160; ++y) {
            float* row = raw.ptr<float>(y);
            for (int x = 0; x < 160; ++x) {
                float value = 0.0f;
                for (int c = 0; c < 32; ++c)
                    value += tensors[9].at(y * 160 + x, c) * detection.coefficients[c];
                row[x] = value;
            }
        }
        cv::Mat full;
        cv::resize(raw, full, cv::Size(640, 640), 0, 0, cv::INTER_LINEAR);
        cv::Mat binary = full > 0;
        for (int y = 0; y < 640; ++y) {
            unsigned char* row = binary.ptr<unsigned char>(y);
            for (int x = 0; x < 640; ++x) {
                if (x < detection.box[0] || x >= detection.box[2] ||
                    y < detection.box[1] || y >= detection.box[3])
                    row[x] = 0;
            }
        }
        const cv::Mat unpadded = binary(cv::Rect(transform.left, transform.top,
                                                 transform.width, transform.height));
        cv::Mat restored_mask;
        cv::resize(unpadded, restored_mask, source.size(), 0, 0, cv::INTER_NEAREST);
        result.masks.push_back(std::move(restored_mask));

        detection.box[0] = std::clamp(static_cast<float>((detection.box[0] - transform.left) / transform.gain),
                                      0.0f, static_cast<float>(source.cols));
        detection.box[1] = std::clamp(static_cast<float>((detection.box[1] - transform.top) / transform.gain),
                                      0.0f, static_cast<float>(source.rows));
        detection.box[2] = std::clamp(static_cast<float>((detection.box[2] - transform.left) / transform.gain),
                                      0.0f, static_cast<float>(source.cols));
        detection.box[3] = std::clamp(static_cast<float>((detection.box[3] - transform.top) / transform.gain),
                                      0.0f, static_cast<float>(source.rows));
        result.detections.push_back(detection);
    }
    return result;
}

}  // namespace

namespace yoloe26 {
struct YoloE26Seg::Impl { Runtime runtime; };

YoloE26Seg::YoloE26Seg(const std::string& model_path) : impl_(std::make_unique<Impl>()) {
    auto read_word = [](const char* path) {
        std::ifstream stream(path);
        std::string value;
        std::getline(stream, value);
        std::string result;
        for (unsigned char c : value) if (std::isalnum(c)) result += std::tolower(c);
        return result;
    };
    const std::string soc = read_word("/sys/class/boardinfo/soc_name");
    const std::string board = read_word("/sys/class/boardinfo/board_type");
    if (soc != "s100" && soc != "s100p") throw std::runtime_error("Only S100/S100P are supported");
    const bool plus = soc == "s100p" || board == "p" || board == "nashm" ||
                      board.rfind("s100p", 0) == 0 || board.rfind("rdks100p", 0) == 0;
    const std::string suffix = plus ? "_nashm_640x640_nv12.hbm" : "_nashe_640x640_nv12.hbm";
    if (model_path.size() < suffix.size() ||
        model_path.compare(model_path.size() - suffix.size(), suffix.size(), suffix) != 0)
        throw std::runtime_error("Use the canonical HBM filename matching this board: " + suffix);
    impl_->runtime.load(model_path.c_str());
}

YoloE26Seg::~YoloE26Seg() = default;

Result YoloE26Seg::predict(const cv::Mat& image, float threshold, int max_det, bool single_label) {
    if (!(threshold > 0 && threshold < 1) || max_det < 1 || max_det > 8400)
        throw std::invalid_argument("Invalid threshold or max_det");
    LetterboxTransform transform;
    cv::Mat padded = letterbox(image, transform);
    impl_->runtime.run(padded);
    return decode_frame(impl_->runtime, image, transform, threshold, max_det, single_label);
}
}  // namespace yoloe26
