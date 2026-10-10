// Copyright (c) 2025 D-Robotics Corporation
// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file ocr.cpp
 * @brief PaddleOCR det/rec model implementation: lifecycle and stage math.
 *
 * The DNN/UCP handles, tensor buffers and task lifetime live in the private
 * Impl structs in this file; the public header exposes only owned stage-data
 * types. Detector preprocessing resizes with INTER_AREA (no letterbox — the
 * DB detector was validated on directly resized input) and packs NV12
 * planes; detector inference copies the prediction map out in the map's own
 * domain (float32 for PP-OCRv6, int16 + scale for the legacy PP-OCRv3
 * export). Detector postprocessing thresholds the map, dilates the contours
 * with a ClipperLib offset (D' = area * ratio_prime / perimeter) and crops
 * the rectified text regions. Recognizer preprocessing produces RGB float32
 * CHW planes in [0, 1] (no ImageNet normalization — the model was
 * calibrated on raw [0, 1] crops); recognizer postprocessing is the greedy
 * CTC decode over the owned logits.
 */

#include "ocr.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"

#include <opencv2/imgproc.hpp>
#include <polyclipping/clipper.hpp>

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

/** Clamp a ClipperLib 64-bit coordinate back into the cv::Point int range. */
int safe_cast_coord(long long v)
{
    if (v > std::numeric_limits<int>::max()) return std::numeric_limits<int>::max();
    if (v < std::numeric_limits<int>::min()) return std::numeric_limits<int>::min();
    return static_cast<int>(v);
}

/**
 * Dilate (offset) contours by a data-driven scale using ClipperOffset.
 *
 * For each polygon, computes offset distance D' = area * ratio_prime /
 * perimeter, then performs polygon offsetting via ClipperLib with round
 * joins. Results that are empty or produce multiple polygons are skipped.
 */
std::vector<std::vector<cv::Point>> dilate_contours(
    const std::vector<std::vector<cv::Point>>& contours,
    float ratio_prime)
{
    std::vector<std::vector<cv::Point>> dilated_polys;

    for (size_t idx = 0; idx < contours.size(); ++idx) {
        const auto& poly = contours[idx];

        const double arc_length = cv::arcLength(poly, true);
        if (arc_length == 0)
            continue;  // degenerate contour: skipped without a result

        const double area = cv::contourArea(poly);
        const double d_prime = area * ratio_prime / arc_length;

        ClipperLib::Path path;
        path.reserve(poly.size());
        for (const auto& pt : poly)
            path.push_back(ClipperLib::IntPoint(pt.x, pt.y));

        ClipperLib::ClipperOffset pco;
        pco.AddPath(path, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

        ClipperLib::Paths solution;
        pco.Execute(solution, d_prime);

        if (solution.size() != 1)
            continue;  // offset did not produce exactly one polygon: skipped
        const auto& sol_path = solution[0];
        if (sol_path.empty())
            continue;  // offset produced an empty polygon: skipped

        std::vector<cv::Point> cv_poly;
        cv_poly.reserve(sol_path.size());
        for (const auto& ipt : sol_path)
            cv_poly.emplace_back(safe_cast_coord(ipt.X), safe_cast_coord(ipt.Y));

        dilated_polys.push_back(std::move(cv_poly));
    }

    return dilated_polys;
}

}  // namespace

// ===========================================================================
// PaddleOCRDet
// ===========================================================================

/** SDK state of the detector: handles plus the reusable tensor buffers. */
struct PaddleOCRDet::Impl
{
    hbDNNPackedHandle_t packed_dnn_handle{nullptr};  ///< Packed model handle.
    hbDNNHandle_t dnn_handle{nullptr};               ///< Selected model handle.
    int model_count{0};                              ///< Models in the pack.
    std::vector<hbDNNTensor> input_tensors;          ///< Reusable input buffers.
    std::vector<hbDNNTensor> output_tensors;         ///< Reusable output buffers.
    int input_h{0};                                  ///< Input height (pixels).
    int input_w{0};                                  ///< Input width (pixels).
    bool quantized_output{false};                    ///< int16 + scale map (PP-OCRv3).
    int map_h{0};                                    ///< Prediction-map height.
    int map_w{0};                                    ///< Prediction-map width.

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

PaddleOCRDet::PaddleOCRDet(const std::string& model_path) : impl_(new Impl)
{
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
        throw std::runtime_error("detector exposes no input or no output tensor");
    if (input_count != 2)
        throw std::runtime_error(
            "expected the S-series NV12 detector contract: exactly Y and UV "
            "input tensors, got " + std::to_string(input_count));

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
        rc = hbDNNGetOutputTensorProperties(&impl_->output_tensors[i].properties,
                                            impl_->dnn_handle, i);
        if (rc != 0)
            throw_dnn(rc, "hbDNNGetOutputTensorProperties");
    }

    // Validate the NV12 input contract before indexing or copying any tensor
    // buffer (the shape arrays hold 8 entries; declared counts beyond that
    // cannot carry the NV12 layout).
    const auto& y_shape = impl_->input_tensors[0].properties.validShape;
    const auto& uv_shape = impl_->input_tensors[1].properties.validShape;
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

    // Validate the prediction-map output: the map is read linearly as
    // map_h * map_w values in the map's own domain, so the declared shape
    // must fit the allocated buffer at every step of the product.
    const auto& out_props = impl_->output_tensors[0].properties;
    if (out_props.validShape.numDimensions < 2 ||
        out_props.validShape.numDimensions > 8)
        throw std::runtime_error(
            "detector output tensor must have between 2 and 8 dimensions");
    if (out_props.tensorType == HB_DNN_TENSOR_TYPE_S16) {
        // Quantized int16 map (PP-OCRv3 export): requires scale quantization
        // with a non-null scale map before the buffer is read as int16.
        if (out_props.quantiType != SCALE || out_props.scale.scaleData == nullptr)
            throw std::runtime_error(
                "S16 detector output must be scale-quantized with scale data");
        impl_->quantized_output = true;
    } else if (out_props.tensorType == HB_DNN_TENSOR_TYPE_F32) {
        // PP-OCRv6 export: float32 map read directly, no quantization.
        if (out_props.quantiType != NONE)
            throw std::runtime_error(
                "F32 detector output must not declare quantization");
    } else {
        throw std::runtime_error(
            "detector output tensor must be F32 or S16+scale for this runtime");
    }
    const int elem_size = impl_->quantized_output
                              ? static_cast<int>(sizeof(int16_t))
                              : static_cast<int>(sizeof(float));
    const int64_t max_values =
        static_cast<int64_t>(out_props.alignedByteSize) / elem_size;
    if (max_values <= 0)
        throw std::runtime_error(
            "declared detector output values exceed the allocated output buffer");
    int64_t shape_product = 1;
    for (int i = 0; i < out_props.validShape.numDimensions; ++i) {
        const int32_t dim = out_props.validShape.dimensionSize[i];
        if (dim <= 0)
            throw std::runtime_error("detector output tensor has a non-positive extent");
        if (shape_product > max_values / dim)
            throw std::runtime_error(
                "declared detector output values exceed the allocated output buffer");
        shape_product *= dim;
    }
    impl_->map_h =
        out_props.validShape.dimensionSize[out_props.validShape.numDimensions - 2];
    impl_->map_w =
        out_props.validShape.dimensionSize[out_props.validShape.numDimensions - 1];
    if (static_cast<int64_t>(impl_->map_h) * impl_->map_w > max_values)
        throw std::runtime_error(
            "declared detector output values exceed the allocated output buffer");

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
    // row stride; a fixed stride smaller than the row, or a plane span
    // beyond the allocation, would overlap rows or overrun the buffer.
    {
        const int64_t row_bytes = impl_->input_w;
        const int64_t row_stride = impl_->input_tensors[0].properties.stride[1];
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
            span > static_cast<int64_t>(impl_->input_tensors[1].sysMem.memSize))
            throw std::runtime_error(
                "UV input row stride cannot carry the input plane");
    }
}

PaddleOCRDet::~PaddleOCRDet() = default;

OcrDetPrepared PaddleOCRDet::preprocess(const cv::Mat& image) const
{
    if (image.empty() || image.type() != CV_8UC3)
        throw std::invalid_argument(
            "detector preprocess expects a nonempty BGR uint8 image");

    // The DB detector was delivered on directly resized input: INTER_AREA to
    // the model resolution, no letterboxing.
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(impl_->input_w, impl_->input_h), 0, 0,
               cv::INTER_AREA);

    // Convert BGR -> I420 (Y + U + V planar) and keep the planes as owned
    // NV12 bytes.
    cv::Mat yuv_mat;
    cv::cvtColor(resized, yuv_mat, cv::COLOR_BGR2YUV_I420);
    const uint8_t* yuv_data = yuv_mat.ptr<uint8_t>();

    const int input_h = impl_->input_h;
    const int input_w = impl_->input_w;
    const int uv_height = input_h / 2;
    const int uv_width = input_w / 2;

    OcrDetPrepared prepared;
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

OcrDetRaw PaddleOCRDet::infer(const OcrDetPrepared& input)
{
    auto& y_tensor = impl_->input_tensors[0];
    auto& uv_tensor = impl_->input_tensors[1];
    const int input_h = impl_->input_h;
    const int input_w = impl_->input_w;
    const int uv_height =
        static_cast<int>(uv_tensor.properties.validShape.dimensionSize[1]);
    const int uv_width =
        static_cast<int>(uv_tensor.properties.validShape.dimensionSize[2]);

    if (static_cast<int>(input.y.size()) != input_h * input_w ||
        static_cast<int>(input.uv.size()) != uv_height * uv_width * 2)
        throw std::invalid_argument(
            "prepared input does not match the model geometry");

    // Upload Y row by row, honoring the resolved row byte stride.
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

    // Copy the prediction map out of the reused buffer so the raw values
    // stay owned by the caller across subsequent inferences. The map is read
    // linearly (map_h rows of map_w values), the delivered layout; the
    // constructor bounded the values against the allocated buffer.
    const int map_len = impl_->map_h * impl_->map_w;
    OcrDetRaw raw;
    raw.quantized = impl_->quantized_output;
    raw.map_h = impl_->map_h;
    raw.map_w = impl_->map_w;
    if (impl_->quantized_output) {
        const int16_t* data =
            static_cast<const int16_t*>(impl_->output_tensors[0].sysMem.virAddr);
        raw.pred_s16.assign(data, data + map_len);
        raw.scale = impl_->output_tensors[0].properties.scale.scaleData[0];
    } else {
        const float* data =
            static_cast<const float*>(impl_->output_tensors[0].sysMem.virAddr);
        raw.pred_f32.assign(data, data + map_len);
    }
    return raw;
}

TextDetResult PaddleOCRDet::postprocess(const OcrDetRaw& raw, const cv::Mat& image,
                                        const OcrOptions& options) const
{
    if (image.empty() || image.type() != CV_8UC3)
        throw std::invalid_argument(
            "detector postprocess expects a nonempty BGR uint8 image");
    const int map_len = raw.map_h * raw.map_w;
    if (raw.map_h <= 0 || raw.map_w <= 0 ||
        (raw.quantized
             ? static_cast<int>(raw.pred_s16.size()) != map_len
             : static_cast<int>(raw.pred_f32.size()) != map_len))
        throw std::invalid_argument(
            "raw detector output does not match its declared map geometry");

    // 1) Threshold the prediction map in its own domain and resize the
    //    binary mask to the original image size (bilinear).
    cv::Mat preds_bin(raw.map_h, raw.map_w, CV_8UC1);
    if (raw.quantized) {
        // int16 quantized path (legacy PP-OCRv3): the float threshold is
        // converted to the quantized domain once and compared as integers.
        const int32_t int_threshold =
            static_cast<int32_t>(options.threshold / raw.scale);
        for (int y = 0; y < raw.map_h; ++y)
            for (int x = 0; x < raw.map_w; ++x) {
                const int idx = y * raw.map_w + x;
                preds_bin.at<uint8_t>(y, x) =
                    (static_cast<int32_t>(raw.pred_s16[idx]) > int_threshold)
                        ? 255
                        : 0;
            }
    } else {
        // float32 path (PP-OCRv6): direct comparison.
        for (int y = 0; y < raw.map_h; ++y)
            for (int x = 0; x < raw.map_w; ++x) {
                const int idx = y * raw.map_w + x;
                preds_bin.at<uint8_t>(y, x) =
                    (raw.pred_f32[idx] > options.threshold) ? 255 : 0;
            }
    }
    cv::Mat preds;
    cv::resize(preds_bin, preds, cv::Size(image.cols, image.rows), 0, 0,
               cv::INTER_LINEAR);

    // 2) Find external contours.
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(preds, contours, hierarchy, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    // 3) Dilate polygons by D' = area * ratio_prime / perimeter.
    const auto dilated_polys = dilate_contours(contours, options.ratio_prime);

    // 4) Convert to minimum-area bounding boxes (100 px^2 floor).
    auto boxes_list = get_bounding_boxes(dilated_polys, 100.f);

    // 5) Crop and rectify each detected box.
    TextDetResult result;
    result.boxes = std::move(boxes_list);
    result.crops.reserve(result.boxes.size());
    for (const auto& box : result.boxes)
        result.crops.push_back(crop_and_rotate_image(image, box));

    return result;
}

int PaddleOCRDet::input_width() const { return impl_->input_w; }

int PaddleOCRDet::input_height() const { return impl_->input_h; }

// ===========================================================================
// PaddleOCRRec
// ===========================================================================

/** SDK state of the recognizer: handles plus the reusable tensor buffers. */
struct PaddleOCRRec::Impl
{
    hbDNNPackedHandle_t packed_dnn_handle{nullptr};  ///< Packed model handle.
    hbDNNHandle_t dnn_handle{nullptr};               ///< Selected model handle.
    int model_count{0};                              ///< Models in the pack.
    std::vector<hbDNNTensor> input_tensors;          ///< Reusable input buffers.
    std::vector<hbDNNTensor> output_tensors;         ///< Reusable output buffers.
    int input_h{0};                                  ///< Input height (pixels, NCHW dim 2).
    int input_w{0};                                  ///< Input width (pixels, NCHW dim 3).
    int seq_len{0};                                  ///< Output sequence length T.
    int num_classes{0};                              ///< Output vocabulary V (incl. blank).

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

PaddleOCRRec::PaddleOCRRec(const std::string& model_path) : impl_(new Impl)
{
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
        throw std::runtime_error("recognizer exposes no input or no output tensor");
    if (input_count != 1)
        throw std::runtime_error(
            "expected the recognizer contract: exactly one NCHW float input, "
            "got " + std::to_string(input_count));

    impl_->input_tensors.resize(input_count);
    impl_->output_tensors.resize(output_count);

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
        rc = hbDNNGetOutputTensorProperties(&impl_->output_tensors[i].properties,
                                            impl_->dnn_handle, i);
        if (rc != 0)
            throw_dnn(rc, "hbDNNGetOutputTensorProperties");
    }

    // Validate the NCHW float input contract (dims: N, C, H, W).
    const auto& in_props = impl_->input_tensors[0].properties;
    if (in_props.validShape.numDimensions < 4 ||
        in_props.validShape.numDimensions > 8)
        throw std::runtime_error(
            "recognizer input tensor must have between 4 and 8 dimensions");
    if (in_props.tensorType != HB_DNN_TENSOR_TYPE_F32)
        throw std::runtime_error("recognizer input tensor must be F32");
    if (in_props.validShape.dimensionSize[1] != 3)
        throw std::runtime_error(
            "recognizer input expects a 3-channel RGB NCHW layout");
    impl_->input_h = in_props.validShape.dimensionSize[2];
    impl_->input_w = in_props.validShape.dimensionSize[3];
    if (impl_->input_h <= 0 || impl_->input_w <= 0)
        throw std::runtime_error("recognizer input tensor has a non-positive extent");

    // Validate the CTC output head: [1, T, V] float logits.
    const auto& out_props = impl_->output_tensors[0].properties;
    if (out_props.tensorType != HB_DNN_TENSOR_TYPE_F32)
        throw std::runtime_error("recognizer output tensor must be F32");
    if (out_props.validShape.numDimensions < 3 ||
        out_props.validShape.numDimensions > 8)
        throw std::runtime_error(
            "recognizer output tensor must have between 3 and 8 dimensions");
    const int64_t max_values =
        static_cast<int64_t>(out_props.alignedByteSize) /
        static_cast<int64_t>(sizeof(float));
    if (max_values <= 0)
        throw std::runtime_error(
            "declared recognizer output values exceed the allocated output buffer");
    int64_t output_len = 1;
    for (int i = 0; i < out_props.validShape.numDimensions; ++i) {
        const int32_t dim = out_props.validShape.dimensionSize[i];
        if (dim <= 0)
            throw std::runtime_error("recognizer output tensor has a non-positive extent");
        if (output_len > max_values / dim)
            throw std::runtime_error(
                "declared recognizer output values exceed the allocated output buffer");
        output_len *= dim;
    }
    impl_->seq_len = out_props.validShape.dimensionSize[1];
    impl_->num_classes = out_props.validShape.dimensionSize[2];

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

    // The upload writes each CHW plane row at stride[2] within a plane
    // pitched at stride[1]; validate the resolved strides carry the planes
    // and the whole span fits the allocation before any copy can run.
    {
        const int64_t row_bytes =
            static_cast<int64_t>(impl_->input_w) * sizeof(float);
        const int64_t row_stride = in_props.stride[2];
        const int64_t plane_span =
            static_cast<int64_t>(impl_->input_h - 1) * row_stride + row_bytes;
        if (row_stride < row_bytes ||
            in_props.stride[1] < plane_span ||
            static_cast<int64_t>(in_props.stride[1]) * (3 - 1) + plane_span >
                static_cast<int64_t>(impl_->input_tensors[0].sysMem.memSize))
            throw std::runtime_error(
                "recognizer input strides cannot carry the CHW planes");
    }
    // The CTC read walks T timestep rows at stride[1].
    {
        const int64_t row_bytes =
            static_cast<int64_t>(impl_->num_classes) * sizeof(float);
        const int64_t row_stride = out_props.stride[1];
        const int64_t span =
            static_cast<int64_t>(impl_->seq_len - 1) * row_stride + row_bytes;
        if (row_stride < row_bytes ||
            span > static_cast<int64_t>(
                       impl_->output_tensors[0].sysMem.memSize))
            throw std::runtime_error(
                "recognizer output row stride cannot carry the CTC logits");
    }
}

PaddleOCRRec::~PaddleOCRRec() = default;

OcrRecPrepared PaddleOCRRec::preprocess(const cv::Mat& crop) const
{
    if (crop.empty() || crop.type() != CV_8UC3)
        throw std::invalid_argument(
            "recognizer preprocess expects a nonempty BGR uint8 crop");

    // 1. BGR -> RGB.
    cv::Mat rgb_mat;
    cv::cvtColor(crop, rgb_mat, cv::COLOR_BGR2RGB);

    // 2. Resize to the model input.
    cv::Mat resized;
    cv::resize(rgb_mat, resized, cv::Size(impl_->input_w, impl_->input_h), 0, 0,
               cv::INTER_AREA);

    // 3. To float32 in [0,1]. The recognition model is exported with
    //    NORM_TYPE="no_preprocess" and was calibrated on raw [0,1] crops, so
    //    ImageNet mean/std normalization must NOT be applied here — doing so
    //    shifts the input off the calibration range and produces garbled CTC
    //    decodes.
    resized.convertTo(resized, CV_32F, 1.0f / 255.0f);

    // 4. Split into owned CHW planes (R, G, B).
    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);
    OcrRecPrepared prepared;
    prepared.chw.resize(static_cast<size_t>(3) * impl_->input_h * impl_->input_w);
    for (int c = 0; c < 3; ++c) {
        float* dst = prepared.chw.data() +
                     static_cast<size_t>(c) * impl_->input_h * impl_->input_w;
        for (int h = 0; h < impl_->input_h; ++h)
            std::memcpy(dst + static_cast<size_t>(h) * impl_->input_w,
                        channels[c].ptr<float>(h),
                        static_cast<size_t>(impl_->input_w) * sizeof(float));
    }
    return prepared;
}

OcrRecRaw PaddleOCRRec::infer(const OcrRecPrepared& input)
{
    auto& in_tensor = impl_->input_tensors[0];
    const int input_h = impl_->input_h;
    const int input_w = impl_->input_w;

    if (static_cast<int>(input.chw.size()) != 3 * input_h * input_w)
        throw std::invalid_argument(
            "prepared input does not match the model geometry");

    // Upload the CHW planes honoring the resolved plane/row byte strides.
    uint8_t* base = reinterpret_cast<uint8_t*>(in_tensor.sysMem.virAddr);
    const int64_t plane_stride = in_tensor.properties.stride[1];
    const int64_t row_stride = in_tensor.properties.stride[2];
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < input_h; ++h) {
            const float* src = input.chw.data() +
                               (static_cast<size_t>(c) * input_h + h) * input_w;
            float* dst = reinterpret_cast<float*>(
                base + c * plane_stride + h * row_stride);
            std::memcpy(dst, src, static_cast<size_t>(input_w) * sizeof(float));
        }
    }

    // Ensure the uploaded data is visible to the BPU.
    int32_t rc = hbUCPMemFlush(&in_tensor.sysMem, HB_SYS_MEM_CACHE_CLEAN);
    if (rc != 0)
        throw_ucp(rc, "hbUCPMemFlush(input)");

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

    for (std::size_t i = 0; i < impl_->output_tensors.size(); ++i) {
        rc = hbUCPMemFlush(&impl_->output_tensors[i].sysMem,
                           HB_SYS_MEM_CACHE_INVALIDATE);
        if (rc != 0)
            throw_ucp(rc, "hbUCPMemFlush(output)");
    }

    rc = hbUCPReleaseTask(task.handle);
    task.handle = nullptr;  // released or unreleasable — never release twice
    if (rc != 0)
        throw_ucp(rc, "hbUCPReleaseTask");

    // Copy the CTC logits out of the reused buffer timestep row by timestep
    // row (stride-aware, as delivered) so the raw values stay owned by the
    // caller across subsequent inferences.
    OcrRecRaw raw;
    raw.logits.resize(static_cast<size_t>(impl_->seq_len) * impl_->num_classes);
    const uint8_t* out_base =
        static_cast<const uint8_t*>(impl_->output_tensors[0].sysMem.virAddr);
    const int64_t out_row_stride = impl_->output_tensors[0].properties.stride[1];
    float* dst = raw.logits.data();
    for (int t = 0; t < impl_->seq_len; ++t) {
        const float* row = reinterpret_cast<const float*>(
            out_base + t * out_row_stride);
        std::memcpy(dst, row,
                    static_cast<size_t>(impl_->num_classes) * sizeof(float));
        dst += impl_->num_classes;
    }
    return raw;
}

std::string PaddleOCRRec::postprocess(const OcrRecRaw& raw,
                                      const std::vector<std::string>& id2token) const
{
    if (static_cast<int>(raw.logits.size()) !=
        impl_->seq_len * impl_->num_classes)
        throw std::invalid_argument(
            "raw recognizer output does not match the model head");
    if (id2token.size() < static_cast<std::size_t>(impl_->num_classes))
        throw std::invalid_argument(
            "token dictionary is smaller than the model vocabulary");

    // Greedy CTC decode: argmax per timestep (first maximum wins, matching
    // std::max_element), collapse consecutive repeats, skip the blank id 0.
    std::string result;
    int prev_idx = -1;
    for (int t = 0; t < impl_->seq_len; ++t) {
        const float* row =
            raw.logits.data() + static_cast<size_t>(t) * impl_->num_classes;
        const int idx = static_cast<int>(
            std::max_element(row, row + impl_->num_classes) - row);
        if (idx != 0 && idx != prev_idx)
            result += id2token[idx];
        prev_idx = idx;
    }
    return result;
}

int PaddleOCRRec::input_width() const { return impl_->input_w; }

int PaddleOCRRec::input_height() const { return impl_->input_h; }

int PaddleOCRRec::seq_len() const { return impl_->seq_len; }

int PaddleOCRRec::num_classes() const { return impl_->num_classes; }

// ===========================================================================
// PaddleOCR pipeline
// ===========================================================================

PaddleOCR::PaddleOCR(const std::string& det_model_path,
                     const std::string& rec_model_path)
    : det_(det_model_path), rec_(rec_model_path)
{
}

OcrResult PaddleOCR::predict(const cv::Mat& image,
                             const std::vector<std::string>& dictionary_lines,
                             const OcrOptions& options)
{
    // id2token layout consumed by the CTC decode: the blank at id 0, then the
    // dictionary lines in file order, then the trailing space so the vector
    // covers the model's num_classes outputs (blank + dict + space).
    std::vector<std::string> id2token;
    id2token.reserve(dictionary_lines.size() + 2);
    id2token.push_back("blank");
    id2token.insert(id2token.end(), dictionary_lines.begin(),
                    dictionary_lines.end());
    id2token.push_back(" ");

    // Detection: NV12 planes -> BPU prediction map -> thresholded crops.
    OcrResult result;
    result.det =
        det_.postprocess(det_.infer(det_.preprocess(image)), image, options);

    // Recognition: one CTC pass per crop. No detected text regions means no
    // recognition calls at all. A failing crop (degenerate geometry or an SDK
    // fault) skips only its own text while the loop continues, as the
    // delivered pipeline did; the original crop index and failure cause are
    // kept in the result so the CLI can report them, and the surviving
    // texts keep their original crop indices.
    result.texts.reserve(result.det.crops.size());
    for (std::size_t i = 0; i < result.det.crops.size(); ++i) {
        OcrRecRaw raw;
        try {
            raw = rec_.infer(rec_.preprocess(result.det.crops[i]));
        } catch (const std::exception& error) {
            result.crop_errors.push_back(OcrCropError{i, error.what()});
            continue;
        }
        result.text_crop_indices.push_back(i);
        result.texts.push_back(rec_.postprocess(raw, id2token));
    }
    return result;
}

const PaddleOCRDet& PaddleOCR::detector() const { return det_; }

const PaddleOCRRec& PaddleOCR::recognizer() const { return rec_; }
