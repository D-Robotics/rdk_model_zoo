// Copyright (c) 2025 D-Robotics Corporation
// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file ocr.hpp
 * @brief Public interface of the two-stage PaddleOCR runtime.
 *
 * A named model class per stage plus the visible pipeline: PaddleOCRDet
 * (DB text detection, NV12 input), PaddleOCRRec (CRNN recognition, F32 NCHW
 * input, CTC output) and PaddleOCR, whose predict() orchestrates detection,
 * perspective cropping and per-crop recognition.
 *
 * The stage data types are owned by the caller between calls: the prepared
 * NV12/CHW planes and the raw detector/recognizer outputs are plain
 * std::vector storage that survives later inferences. DNN/UCP handles and
 * tensor buffers are implementation details of the .cpp file.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

/** Detection and pipeline tuning parameters (DB postprocess defaults). */
struct OcrOptions
{
    float threshold{0.5f};    ///< Binarization threshold for the prediction map (float domain)
    float ratio_prime{2.7f};  ///< Contour dilation ratio (D' = area * ratio_prime / perimeter)
};

/** Owned NV12 planes prepared for the detector's BPU input. */
struct OcrDetPrepared
{
    std::vector<uint8_t> y;   ///< Y plane, input_h x input_w bytes
    std::vector<uint8_t> uv;  ///< Interleaved UV plane, (input_h/2) x (input_w/2) x 2 bytes
};

/**
 * @brief Owned detector output: the DB prediction map.
 *
 * PP-OCRv6 models emit a float32 map (pred_f32); the legacy PP-OCRv3
 * variants emit int16 values with a dequantization scale (pred_s16 with
 * scale, quantized = true). The threshold comparison in postprocess runs
 * in the map's own domain.
 */
struct OcrDetRaw
{
    bool quantized{false};           ///< True when the map is int16 + scale
    std::vector<float> pred_f32;     ///< Float map, map_h * map_w values
    std::vector<int16_t> pred_s16;   ///< Quantized map, map_h * map_w values
    float scale{0.0f};               ///< Dequantization scale for pred_s16
    int map_h{0};                    ///< Map height
    int map_w{0};                    ///< Map width
};

/**
 * @brief Detection-stage result: perspective-rectified crops and boxes.
 *
 * Crops are ready for the recognition stage; boxes are 4-point polygons in
 * original-image pixel coordinates, index-aligned with crops.
 */
struct TextDetResult
{
    std::vector<cv::Mat> crops;                ///< Rectified crop images
    std::vector<std::vector<cv::Point>> boxes; ///< 4-point polygon boxes
};

/** Owned recognizer input: RGB float32 CHW planes scaled to [0, 1]. */
struct OcrRecPrepared
{
    std::vector<float> chw;  ///< 3 * input_h * input_w values, plane-major (R, G, B)
};

/** Owned recognizer output: CTC logits packed as seq_len rows of num_classes. */
struct OcrRecRaw
{
    std::vector<float> logits;  ///< seq_len * num_classes values, timestep-major
};

/** Diagnostic for one detection crop whose recognition failed. */
struct OcrCropError
{
    std::size_t crop_index{0};  ///< Original detection-crop index that failed
    std::string message;        ///< Cause of the failure (exception description)
};

/**
 * @brief Pipeline result: detection crops/boxes plus the recognition outcome.
 *
 * texts holds the recognized string of every surviving crop in detection
 * order, with text_crop_indices giving each text's original crop index (a
 * skipped crop leaves a gap, so texts and det.crops are not index-aligned).
 * crop_errors carries the original index and cause of every skipped crop so
 * the caller can report them.
 */
struct OcrResult
{
    TextDetResult det;  ///< Detection-stage crops and boxes
    std::vector<std::string> texts;  ///< Recognized text per surviving crop
    std::vector<std::size_t> text_crop_indices;  ///< Original crop index per text
    std::vector<OcrCropError> crop_errors;       ///< Skipped crops and their causes
};

/** DB-algorithm text detector (NV12 input, prediction-map output). */
class PaddleOCRDet
{
public:
    /** Load the detector model; throws std::runtime_error on failure. */
    explicit PaddleOCRDet(const std::string& model_path);
    ~PaddleOCRDet();
    PaddleOCRDet(const PaddleOCRDet&) = delete;
    PaddleOCRDet& operator=(const PaddleOCRDet&) = delete;

    /** BGR image -> resized NV12 planes at the model input resolution. */
    OcrDetPrepared preprocess(const cv::Mat& image) const;
    /** Upload the planes, run one BPU task, own the prediction map. */
    OcrDetRaw infer(const OcrDetPrepared& input);
    /** Threshold the map, find/dilate contours, crop the text regions. */
    TextDetResult postprocess(const OcrDetRaw& raw, const cv::Mat& image,
                              const OcrOptions& options) const;

    int input_width() const;
    int input_height() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/** CRNN text recognizer (F32 NCHW input, CTC logits output). */
class PaddleOCRRec
{
public:
    /** Load the recognizer model; throws std::runtime_error on failure. */
    explicit PaddleOCRRec(const std::string& model_path);
    ~PaddleOCRRec();
    PaddleOCRRec(const PaddleOCRRec&) = delete;
    PaddleOCRRec& operator=(const PaddleOCRRec&) = delete;

    /** BGR crop -> RGB, resize, [0, 1] float32 CHW planes. */
    OcrRecPrepared preprocess(const cv::Mat& crop) const;
    /** Upload the planes, run one BPU task, own the CTC logits. */
    OcrRecRaw infer(const OcrRecPrepared& input);
    /** Greedy CTC decode: per-step argmax, collapse repeats, skip blank. */
    std::string postprocess(const OcrRecRaw& raw,
                            const std::vector<std::string>& id2token) const;

    int input_width() const;
    int input_height() const;
    int seq_len() const;
    int num_classes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Two-stage OCR pipeline: detect, then recognize each crop.
 *
 * predict() composes the stage calls visibly; a detection with no text
 * regions short-circuits the recognition stage (no crops, no texts). A
 * failing crop skips only its own text while recognition continues with the
 * remaining crops; the result records the surviving crops' original indices
 * and each skipped crop's cause.
 */
class PaddleOCR
{
public:
    PaddleOCR(const std::string& det_model_path,
              const std::string& rec_model_path);
    ~PaddleOCR() = default;
    PaddleOCR(const PaddleOCR&) = delete;
    PaddleOCR& operator=(const PaddleOCR&) = delete;

    /**
     * @brief Run detection + recognition on one image.
     *
     * @p dictionary_lines are the verbatim vocabulary-file lines; the CTC
     * blank is prepended at id 0 and the trailing space appended here, so
     * id2token covers the model's num_classes outputs.
     */
    OcrResult predict(const cv::Mat& image,
                      const std::vector<std::string>& dictionary_lines,
                      const OcrOptions& options);

    const PaddleOCRDet& detector() const;
    const PaddleOCRRec& recognizer() const;

private:
    PaddleOCRDet det_;
    PaddleOCRRec rec_;
};
