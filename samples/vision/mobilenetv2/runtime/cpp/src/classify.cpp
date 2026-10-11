// Copyright (c) 2025 D-Robotics Corporation
// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file classify.cpp
 * @brief MobileNetV2 model implementation: runtime lifecycle and stage math.
 *
 * The DNN/UCP handles, tensor buffers and task lifetime live in the private
 * Impl in this file; the public header exposes only owned stage-data types.
 * preprocess produces the center-cropped NV12 planes (Pillow-compatible
 * shorter-edge resize and center crop, BGR→I420 conversion, interleaved UV
 * packing); infer uploads the planes into the reusable SDK tensors
 * row-stride aware, runs one BPU task and copies the F32 output into an owned
 * logits vector; postprocess applies a stable softmax and Top-K selection.
 */

#include "classify.hpp"
#include "geometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
// hbDNNGetErrorDesc/hbUCPGetErrorDesc: the board SDK headers above do not include these.
#include "hobot/dnn/hb_dnn_status.h"
#include "hobot/hb_ucp_status.h"

#include <opencv2/imgproc.hpp>

#include "postprocess.hpp"
#include "preprocess.hpp"

namespace {

/** Build an exception carrying the DNN error description. */
[[noreturn]] void throw_dnn(int32_t code, const char* context)
{
    const char* desc = hbDNNGetErrorDesc(code);
    throw std::runtime_error(std::string(context) + " failed: " +
                             std::to_string(code) +
                             (desc ? std::string(" (") + desc + ")" : std::string()));
}

/** Build an exception carrying the UCP error description. */
[[noreturn]] void throw_ucp(int32_t code, const char* context)
{
    const char* desc = hbUCPGetErrorDesc(code);
    throw std::runtime_error(std::string(context) + " failed: " +
                             std::to_string(code) +
                             (desc ? std::string(" (") + desc + ")" : std::string()));
}

/** Release the inference task handle on every exit path, failure included. */
struct TaskGuard
{
    TaskGuard() = default;
    hbUCPTaskHandle_t handle{nullptr};

    ~TaskGuard()
    {
        if (handle)
            hbUCPReleaseTask(handle);
    }

    TaskGuard(const TaskGuard&) = delete;
    TaskGuard& operator=(const TaskGuard&) = delete;
};

}  // namespace

/** SDK state: packed/model handles plus the reusable tensor buffers. */
struct MobileNetV2::Impl
{
    hbDNNPackedHandle_t packed_dnn_handle{nullptr};  ///< Packed model handle.
    hbDNNHandle_t dnn_handle{nullptr};               ///< Selected model handle.
    int model_count{0};                              ///< Models in the pack.
    std::vector<hbDNNTensor> input_tensors;          ///< Reusable input buffers.
    std::vector<hbDNNTensor> output_tensors;         ///< Reusable output buffers.
    int input_h{0};                                  ///< Input height (pixels).
    int input_w{0};                                  ///< Input width (pixels).
    int output_len{0};                               ///< F32 values in output 0.
    int resize_shorter{256};                         ///< Shorter edge before the crop.

    ~Impl()
    {
        for (auto& tensor : input_tensors)
            if (tensor.sysMem.virAddr)
                hbUCPFree(&tensor.sysMem);
        for (auto& tensor : output_tensors)
            if (tensor.sysMem.virAddr)
                hbUCPFree(&tensor.sysMem);
        if (packed_dnn_handle != nullptr)
            hbDNNRelease(packed_dnn_handle);
    }
};

MobileNetV2::MobileNetV2(const std::string& model_path, int resize_shorter) : impl_(new Impl)
{
    impl_->resize_shorter = resize_shorter;
    // Load the packed model from disk; everything after this point is owned
    // by Impl and released by its destructor on any failure.
    const char* path = model_path.c_str();
    int32_t rc = hbDNNInitializeFromFiles(&impl_->packed_dnn_handle, &path, 1);
    if (rc != 0)
        throw_dnn(rc, "hbDNNInitializeFromFiles");

    const char** model_name_list = nullptr;
    rc = hbDNNGetModelNameList(&model_name_list, &impl_->model_count,
                               impl_->packed_dnn_handle);
    if (rc != 0)
        throw_dnn(rc, "hbDNNGetModelNameList");
    if (impl_->model_count < 1 || model_name_list == nullptr ||
        model_name_list[0] == nullptr)
        throw std::runtime_error("hbDNNGetModelNameList: model pack has no model");

    rc = hbDNNGetModelHandle(&impl_->dnn_handle, impl_->packed_dnn_handle,
                             model_name_list[0]);
    if (rc != 0)
        throw_dnn(rc, "hbDNNGetModelHandle");

    int32_t input_count = 0;
    int32_t output_count = 0;
    rc = hbDNNGetInputCount(&input_count, impl_->dnn_handle);
    if (rc != 0)
        throw_dnn(rc, "hbDNNGetInputCount");
    rc = hbDNNGetOutputCount(&output_count, impl_->dnn_handle);
    if (rc != 0)
        throw_dnn(rc, "hbDNNGetOutputCount");
    if (input_count < 1 || output_count < 1)
        throw std::runtime_error("model exposes no input or no output tensor");
    if (input_count != 2)
        throw std::runtime_error(
            "expected the S-series NV12 model contract: exactly Y and UV input "
            "tensors, got " + std::to_string(input_count));

    impl_->input_tensors.resize(input_count);
    impl_->output_tensors.resize(output_count);

    // Zero the sysMem fields so the destructor only frees allocated memory.
    for (int i = 0; i < input_count; ++i)
        std::memset(&impl_->input_tensors[i].sysMem, 0, sizeof(hbUCPSysMem));
    for (int i = 0; i < output_count; ++i)
        std::memset(&impl_->output_tensors[i].sysMem, 0, sizeof(hbUCPSysMem));

    for (int i = 0; i < input_count; ++i) {
        rc = hbDNNGetInputTensorProperties(&impl_->input_tensors[i].properties,
                                           impl_->dnn_handle, i);
        if (rc != 0)
            throw_dnn(rc, "hbDNNGetInputTensorProperties");
    }
    for (int i = 0; i < output_count; ++i) {
        rc = hbDNNGetOutputTensorProperties(
            &impl_->output_tensors[i].properties, impl_->dnn_handle, i);
        if (rc != 0)
            throw_dnn(rc, "hbDNNGetOutputTensorProperties");
    }

    // Validate the supported model contract before indexing or copying any
    // tensor buffer: an NV12 Y + UV input pair with even, positive geometry
    // and an F32 classification head whose declared values fit the allocated
    // output buffer. Malformed models are rejected instead of misread.
    const auto& y_shape = impl_->input_tensors[0].properties.validShape;
    const auto& uv_shape = impl_->input_tensors[1].properties.validShape;
    // The shape arrays hold 8 entries; a declared count outside 3..8 would
    // read out of bounds or cannot carry the NV12 layout.
    if (y_shape.numDimensions < 3 || y_shape.numDimensions > 8)
        throw std::runtime_error(
            "Y input tensor must have between 3 and 8 dimensions");
    impl_->input_h = y_shape.dimensionSize[1];
    impl_->input_w = y_shape.dimensionSize[2];
    if (impl_->input_h <= 0 || impl_->input_w <= 0)
        throw std::runtime_error("Y input tensor has a non-positive extent");
    if (impl_->input_h % 2 || impl_->input_w % 2)
        throw std::runtime_error("NV12 input height and width must be even");
    if (uv_shape.numDimensions < 3 || uv_shape.numDimensions > 8)
        throw std::runtime_error(
            "UV input tensor must have between 3 and 8 dimensions");
    if (uv_shape.dimensionSize[1] != impl_->input_h / 2 ||
        uv_shape.dimensionSize[2] != impl_->input_w / 2)
        throw std::runtime_error(
            "UV input extents must be half the Y input extents");

    const auto& out_props = impl_->output_tensors[0].properties;
    if (out_props.tensorType != HB_DNN_TENSOR_TYPE_F32)
        throw std::runtime_error("output tensor must be F32 for this runtime");
    if (out_props.validShape.numDimensions <= 0 ||
        out_props.validShape.numDimensions > 8)
        throw std::runtime_error("output tensor dimension count out of range");
    // Bound the product by the values the allocated buffer can hold at every
    // step, so a malformed shape cannot overflow the multiplication itself.
    const int64_t max_values =
        static_cast<int64_t>(out_props.alignedByteSize) /
        static_cast<int64_t>(sizeof(float));
    if (max_values <= 0)
        throw std::runtime_error(
            "declared output values exceed the allocated output buffer");
    int64_t output_len = 1;
    for (int i = 0; i < out_props.validShape.numDimensions; ++i) {
        const int32_t dim = out_props.validShape.dimensionSize[i];
        if (dim <= 0)
            throw std::runtime_error("output tensor has a non-positive extent");
        if (output_len > max_values / dim)
            throw std::runtime_error(
                "declared output values exceed the allocated output buffer");
        output_len *= dim;
    }
    impl_->output_len = static_cast<int>(output_len);

    if (prepare_input_tensor(impl_->input_tensors) != 0)
        throw std::runtime_error("input tensor preparation failed");
    if (prepare_output_tensor(impl_->output_tensors) != 0)
        throw std::runtime_error("output tensor preparation failed");
    for (const auto& tensor : impl_->input_tensors)
        if (tensor.sysMem.virAddr == nullptr)
            throw std::runtime_error("input tensor allocation returned null");
    for (const auto& tensor : impl_->output_tensors)
        if (tensor.sysMem.virAddr == nullptr)
            throw std::runtime_error("output tensor allocation returned null");

    // The upload writes one row of plane bytes per image row at the resolved
    // row stride. Dynamic (-1) strides were resolved by the allocation above
    // into aligned pitches that carry the plane; a fixed stride smaller than
    // the row, or a plane span beyond the allocation, would overlap rows or
    // overrun the buffer, so reject it before any copy can run.
    {
        const int64_t row_bytes = impl_->input_w;
        const int64_t row_stride =
            impl_->input_tensors[0].properties.stride[1];
        const int64_t span =
            static_cast<int64_t>(impl_->input_h - 1) * row_stride + row_bytes;
        if (row_stride < row_bytes ||
            span > static_cast<int64_t>(impl_->input_tensors[0].sysMem.memSize))
            throw std::runtime_error(
                "Y input row stride cannot carry the input plane");
    }
    {
        const auto& uv_props = impl_->input_tensors[1].properties;
        const int64_t row_bytes =
            static_cast<int64_t>(uv_props.validShape.dimensionSize[2]) * 2;
        const int64_t row_stride = uv_props.stride[1];
        const int64_t rows = uv_props.validShape.dimensionSize[1];
        const int64_t span = (rows - 1) * row_stride + row_bytes;
        if (row_stride < row_bytes ||
            span > static_cast<int64_t>(
                       impl_->input_tensors[1].sysMem.memSize))
            throw std::runtime_error(
                "UV input row stride cannot carry the input plane");
    }
}

MobileNetV2::~MobileNetV2() = default;

MobileNetV2Prepared MobileNetV2::preprocess(const cv::Mat& image) const
{
    if (image.empty() || image.type() != CV_8UC3)
        throw std::invalid_argument("preprocess expects a nonempty BGR uint8 image");
    if (impl_->input_h % 2 || impl_->input_w % 2)
        throw std::invalid_argument("model input height and width must be even");

    // Shorter-edge antialiased bicubic resize and center crop to the model
    // input: the geometry the published models were calibrated and evaluated
    // with, identical to the Python runtime.
    if (impl_->input_h != impl_->input_w || impl_->resize_shorter < impl_->input_w)
        throw std::invalid_argument(
            "center crop needs a square input and resize_shorter >= input size");
    const cv::Mat resized = mobilenet::center_crop(image, impl_->input_w, impl_->resize_shorter);

    // Convert BGR -> I420 (Y + U + V planar) and keep the planes as owned
    // NV12 bytes.
    cv::Mat yuv_mat;
    cv::cvtColor(resized, yuv_mat, cv::COLOR_BGR2YUV_I420);
    const uint8_t* yuv_data = yuv_mat.ptr<uint8_t>();

    const int input_h = impl_->input_h;
    const int input_w = impl_->input_w;
    const int uv_height = input_h / 2;
    const int uv_width = input_w / 2;

    MobileNetV2Prepared prepared;
    prepared.y.assign(yuv_data, yuv_data + input_h * input_w);
    prepared.uv.resize(static_cast<size_t>(uv_height) * uv_width * 2);

    const uint8_t* u_data_src = yuv_data + input_h * input_w;
    const uint8_t* v_data_src = u_data_src + uv_height * uv_width;
    uint8_t* uv_dst = prepared.uv.data();
    for (int h = 0; h < uv_height; ++h) {
        for (int w = 0; w < uv_width; ++w) {
            *uv_dst++ = *u_data_src++;  // U
            *uv_dst++ = *v_data_src++;  // V
        }
    }
    return prepared;
}

MobileNetV2Raw MobileNetV2::infer(const MobileNetV2Prepared& input)
{
    auto& y_tensor = impl_->input_tensors[0];
    auto& uv_tensor = impl_->input_tensors[1];
    const int input_h = impl_->input_h;
    const int input_w = impl_->input_w;
    const int uv_height = static_cast<int>(uv_tensor.properties.validShape.dimensionSize[1]);
    const int uv_width = static_cast<int>(uv_tensor.properties.validShape.dimensionSize[2]);

    if (static_cast<int>(input.y.size()) != input_h * input_w ||
        static_cast<int>(input.uv.size()) != uv_height * uv_width * 2)
        throw std::invalid_argument("prepared input does not match the model geometry");

    // Upload Y row by row, honoring the resolved row byte stride of the
    // tensor layout.
    uint8_t* y_dst = reinterpret_cast<uint8_t*>(y_tensor.sysMem.virAddr);
    const uint8_t* y_src = input.y.data();
    for (int h = 0; h < input_h; ++h) {
        std::memcpy(y_dst, y_src, input_w);
        y_src += input_w;
        y_dst += y_tensor.properties.stride[1];
    }

    // Upload the interleaved UV plane with per-row byte strides.
    uint8_t* uv_dst = reinterpret_cast<uint8_t*>(uv_tensor.sysMem.virAddr);
    const uint8_t* uv_src = input.uv.data();
    for (int h = 0; h < uv_height; ++h) {
        std::memcpy(uv_dst, uv_src, static_cast<size_t>(uv_width) * 2);
        uv_src += static_cast<size_t>(uv_width) * 2;
        uv_dst += uv_tensor.properties.stride[1];
    }

    // Ensure the uploaded data is visible to the BPU.
    int32_t rc = hbUCPMemFlush(&y_tensor.sysMem, HB_SYS_MEM_CACHE_CLEAN);
    if (rc != 0)
        throw_ucp(rc, "hbUCPMemFlush(input Y)");
    rc = hbUCPMemFlush(&uv_tensor.sysMem, HB_SYS_MEM_CACHE_CLEAN);
    if (rc != 0)
        throw_ucp(rc, "hbUCPMemFlush(input UV)");

    TaskGuard task;
    rc = hbDNNInferV2(&task.handle, impl_->output_tensors.data(),
                      impl_->input_tensors.data(), impl_->dnn_handle);
    if (rc != 0)
        throw_dnn(rc, "hbDNNInferV2");

    hbUCPSchedParam sched_param;
    HB_UCP_INITIALIZE_SCHED_PARAM(&sched_param);
    sched_param.backend = HB_UCP_BPU_CORE_ANY;
    sched_param.priority = 0;
    rc = hbUCPSubmitTask(task.handle, &sched_param);
    if (rc != 0)
        throw_ucp(rc, "hbUCPSubmitTask");

    rc = hbUCPWaitTaskDone(task.handle, 0);
    if (rc != 0)
        throw_ucp(rc, "hbUCPWaitTaskDone");

    // Invalidate the output caches so the CPU reads fresh BPU results.
    for (std::size_t i = 0; i < impl_->output_tensors.size(); ++i) {
        rc = hbUCPMemFlush(&impl_->output_tensors[i].sysMem,
                           HB_SYS_MEM_CACHE_INVALIDATE);
        if (rc != 0)
            throw_ucp(rc, "hbUCPMemFlush(output)");
    }

    // Release the finished task and report a failed release instead of
    // ignoring it; the guard still covers every exceptional path above.
    rc = hbUCPReleaseTask(task.handle);
    task.handle = nullptr;  // released or unreleasable — never release twice
    if (rc != 0)
        throw_ucp(rc, "hbUCPReleaseTask");

    // Copy the F32 output out of the reused buffer so the returned raw values
    // stay owned by the caller across subsequent inferences. The constructor
    // validated the length against the allocated buffer size. A quantized
    // output (quantiType != NONE) is outside this runtime's contract.
    MobileNetV2Raw raw;
    const float* data =
        static_cast<const float*>(impl_->output_tensors[0].sysMem.virAddr);
    raw.logits.assign(data, data + impl_->output_len);
    return raw;
}

std::vector<Classification> MobileNetV2::postprocess(const MobileNetV2Raw& raw,
                                                     int top_k) const
{
    // The model outputs logits: a numerically stable softmax turns them into
    // probabilities before the classes are ranked.
    const int tensor_len = static_cast<int>(raw.logits.size());
    if (tensor_len <= 0)
        throw std::invalid_argument("postprocess requires a nonempty output vector");

    const float max_logit = *std::max_element(raw.logits.begin(), raw.logits.end());
    double sum = 0.0;
    for (float value : raw.logits)
        sum += std::exp(static_cast<double>(value - max_logit));

    std::vector<Classification> results;
    results.reserve(tensor_len);
    for (int i = 0; i < tensor_len; ++i) {
        Classification cls;
        cls.class_id = i;
        cls.probability = static_cast<float>(
            std::exp(static_cast<double>(raw.logits[i] - max_logit)) / sum);
        results.emplace_back(cls);
    }

    std::vector<Classification> topk_results;
    get_topk_result(results, topk_results, top_k);
    return topk_results;
}

std::vector<Classification> MobileNetV2::predict(const cv::Mat& image,
                                                 int top_k)
{
    return postprocess(infer(preprocess(image)), top_k);
}

int MobileNetV2::input_width() const { return impl_->input_w; }

int MobileNetV2::input_height() const { return impl_->input_h; }
