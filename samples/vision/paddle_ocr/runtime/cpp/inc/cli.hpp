// Copyright (c) 2025 D-Robotics Corporation
// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file cli.hpp
 * @brief PaddleOCR command-line options, input loading and result rendering.
 *
 * The CLI owns option parsing (kebab-case flags matching the Python runtime),
 * path defaults that depend on the configured board, verbatim dictionary
 * loading and result presentation. No model or SDK work happens here.
 */

#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "ocr.hpp"  // OcrResult reported by print_results

namespace ocr {

/** Parsed command line for one program run. */
struct CliOptions
{
    std::string det_model_path;  ///< Detector HBM path (default: board model location).
    std::string rec_model_path;  ///< Recognizer HBM path (default: board model location).
    std::string test_img = "../../../test_data/gt_2322.jpg";  ///< BGR input image.
    std::string vocabulary_path;  ///< Character vocabulary, one token per line.
    float threshold = 0.5f;       ///< Detection-map binarization threshold.
    float ratio_prime = 2.7f;     ///< Contour dilation ratio.
    std::string img_save_path = "result.jpg";  ///< Result image destination.
    std::string font_path;        ///< TrueType font for rendering recognized text.
    bool help = false;            ///< Print usage and exit.
};

/**
 * @brief Default detector HBM path for the configured board.
 *
 * S600 builds use the S600 model location; every other build uses the S100
 * location, matching the historical SoC-dependent defaults.
 *
 * @return Absolute default model path.
 */
std::string default_det_model_path();

/** Default recognizer HBM path for the configured board (see above). */
std::string default_rec_model_path();

/**
 * @brief Parse argv into options.
 *
 * Accepted flags: --det-model-path, --rec-model-path, --test-img,
 * --vocabulary-path, --threshold, --ratio-prime, --img-save-path,
 * --font-path and --help (both "--flag value" and "--flag=value").
 * Unknown flags, missing values or unparsable numbers throw
 * std::invalid_argument.
 *
 * @param argc Argument count including the program name.
 * @param argv Argument values.
 * @return Parsed options with the documented defaults filled in.
 */
CliOptions parse_options(int argc, char** argv);

/** Print the usage text (one line per option). */
void print_help(const char* program);

/**
 * @brief Load the BGR test image.
 *
 * @param path Image path.
 * @return Loaded image; throws std::runtime_error when it cannot be read.
 */
cv::Mat load_image(const std::string& path);

/**
 * @brief Read the character dictionary verbatim, one token per line.
 *
 * PaddleOCR vocab files contain single-character lines that are exactly '{',
 * '}' or ','. The generic linewise label loader strips these characters as
 * if they were dict delimiters and then skips the resulting empty line,
 * which silently shifts every subsequent character id and produces garbled
 * CTC decodes — so the file is read verbatim here (only a trailing '\r' is
 * dropped). The model's predict() prepends the CTC blank and the trailing
 * space.
 *
 * @param path Dictionary file path.
 * @return Dictionary lines in file order; throws std::runtime_error when the
 *         file cannot be opened.
 */
std::vector<std::string> load_token_dictionary(const std::string& path);

/**
 * @brief Print the per-crop recognition outcomes.
 *
 * One line per surviving crop on stdout with its original crop index
 * ("[i] Prediction: text"), then one line per skipped crop on stderr with the
 * original index and failure cause.
 *
 * @param result Pipeline result carrying texts, their crop indices and the
 *               skipped-crop diagnostics.
 */
void print_results(const OcrResult& result);

/**
 * @brief Draw the detection boxes and recognized texts, save the side-by-side
 *        result image.
 *
 * Left half: polygon boxes on a copy of the input. Right half: the texts
 * rendered near their boxes with FreeType on a white canvas. The texts are
 * the surviving crops' strings in detection order; when crops were skipped,
 * they pair with boxes compactly (text j pairs with boxes[j]) — the pairing
 * is not by original crop index.
 *
 * @param image Original BGR input image.
 * @param boxes Detection polygon boxes.
 * @param texts Recognized texts of the surviving crops.
 * @param font_path TrueType font file for the text rendering.
 * @param save_path Destination path for the combined image.
 * @throws std::runtime_error when the image cannot be written.
 */
void render_result(const cv::Mat& image,
                   const std::vector<std::vector<cv::Point>>& boxes,
                   const std::vector<std::string>& texts,
                   const std::string& font_path,
                   const std::string& save_path);

}  // namespace ocr
