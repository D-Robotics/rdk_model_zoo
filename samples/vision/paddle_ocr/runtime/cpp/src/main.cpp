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
 * @file main.cpp
 * @brief Provide a command-line demo application for PaddleOCR text detection
 *        and recognition.
 *
 * This file implements the executable entry point for the PaddleOCR sample.
 * It parses gflags command-line parameters, initializes the detection and
 * recognition model wrappers, and runs the standard two-stage pipeline:
 *
 *   1) Detection:   preprocess (BGR->NV12) -> infer -> postprocess (boxes + crops)
 *   2) Recognition: for each crop: preprocess (BGR->RGB->F32) -> infer -> CTC decode
 *   3) Visualization: draw polygon boxes + render recognized text with FreeType font
 *
 * Results are saved to disk as a side-by-side image.
 *
 * @note This sample only supports RDK S100 platform.
 *
 * @see paddle_ocr.hpp
 */

#include <iostream>
#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "paddle_ocr.hpp"
#include "file_io.hpp"

// ---------------------------------------------------------------------------
// Command-line flags
// ---------------------------------------------------------------------------
DEFINE_string(det_model_path,
              "/opt/hobot/model/s100/basic/cn_PP-OCRv3_det_infer-deploy_640x640_nv12.hbm",
              "Path to BPU quantized text detection model (*.hbm). S100 only.");
DEFINE_string(rec_model_path,
              "/opt/hobot/model/s100/basic/cn_PP-OCRv3_rec_infer-deploy_48x320_rgb.hbm",
              "Path to BPU quantized text recognition model (*.hbm). S100 only.");
DEFINE_string(test_image,
              "../../../test_data/gt_2322.jpg",
              "Path to the test input image.");
DEFINE_string(label_file,
              "../../../test_data/ppocr_keys_v1.txt",
              "Path to the character vocabulary file (one token per line).");
DEFINE_double(threshold,
              0.5,
              "Binarization threshold for the text detection mask (0.0-1.0).");
DEFINE_double(ratio_prime,
              2.7,
              "Contour expansion ratio for bounding box dilation.");
DEFINE_string(img_save_path,
              "result.jpg",
              "Path to save the final result image.");
DEFINE_string(font_path,
              "../../../test_data/FangSong.ttf",
              "Path to the TrueType font file used for rendering recognized text.");

/**
 * @brief Main entry for running the PaddleOCR detection + recognition demo.
 *
 * Pipeline:
 * 1) Parse gflags command-line parameters.
 * 2) Initialize PaddleOCRDet and load the detection model.
 * 3) Load input image (BGR) from disk.
 * 4) Detection: pre_process_det -> infer -> post_process_det -> (crops, boxes).
 * 5) Initialize PaddleOCRRec and load the recognition model.
 * 6) Load character dictionary from label file; prepend blank at index 0.
 * 7) Recognition: for each crop: pre_process_rec -> infer -> post_process_rec -> text.
 * 8) Visualization: draw_polygon_boxes + draw_text on white canvas.
 * 9) hconcat box image and text image; imwrite to FLAGS_img_save_path.
 *
 * @param[in] argc Number of command-line arguments.
 * @param[in] argv Argument vector (expects gflags flags; see DEFINE_* above).
 * @return int Process exit code (0 on success).
 */
int main(int argc, char** argv)
{
    int32_t ret;
    gflags::SetUsageMessage(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    std::cout << gflags::GetArgv() << std::endl;

    // --- Detection stage --------------------------------------------------
    PaddleOCRDet det;
    ret = det.init(FLAGS_det_model_path.c_str());
    if (ret != 0) {
        fprintf(stderr, "PaddleOCRDet init failed (ret=%d)\n", ret);
        return ret;
    }

    // Load input image (BGR)
    auto image = load_bgr_image(FLAGS_test_image);

    // Preprocess, infer, postprocess for detection
    ret = pre_process_det(det.input_tensors, image, det.input_w, det.input_h);
    if (ret != 0) {
        fprintf(stderr, "pre_process_det failed (ret=%d)\n", ret);
        return ret;
    }

    ret = infer(det.output_tensors, det.input_tensors, det.dnn_handle);
    if (ret != 0) {
        fprintf(stderr, "infer (det) failed (ret=%d)\n", ret);
        return ret;
    }

    TextDetResult det_result = post_process_det(
        det.output_tensors, image,
        static_cast<float>(FLAGS_threshold),
        static_cast<float>(FLAGS_ratio_prime));
    auto& cropped_images = det_result.crops;
    auto& boxes_list     = det_result.boxes;

    // --- Recognition stage ------------------------------------------------
    PaddleOCRRec rec;
    ret = rec.init(FLAGS_rec_model_path.c_str());
    if (ret != 0) {
        fprintf(stderr, "PaddleOCRRec init failed (ret=%d)\n", ret);
        return ret;
    }

    // Load dictionary and prepend CTC blank at index 0
    std::vector<std::string> id2token = load_linewise_labels(FLAGS_label_file);
    id2token.insert(id2token.begin(), "blank");

    std::vector<std::string> recognized_texts;
    recognized_texts.reserve(cropped_images.size());

    for (size_t i = 0; i < cropped_images.size(); ++i) {
        cv::Mat& crop = cropped_images[i];

        ret = pre_process_rec(rec.input_tensors, crop, rec.input_w, rec.input_h);
        if (ret != 0) {
            fprintf(stderr, "pre_process_rec[%zu] failed (ret=%d)\n", i, ret);
            continue;
        }

        ret = infer(rec.output_tensors, rec.input_tensors, rec.dnn_handle);
        if (ret != 0) {
            fprintf(stderr, "infer (rec)[%zu] failed (ret=%d)\n", i, ret);
            continue;
        }

        std::string text = post_process_rec(rec.output_tensors, id2token,
                                            rec.seq_len, rec.num_classes);
        recognized_texts.push_back(text);
        std::cout << "[" << i << "] Prediction: " << text << std::endl;
    }

    // --- Visualization ----------------------------------------------------
    // Draw polygon boxes on a copy of the original image
    auto img_boxes = draw_polygon_boxes(image, boxes_list);

    // Create white canvas and render recognized text near boxes
    cv::Mat white_canvas(img_boxes.size(), CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat img_with_text = draw_text(
        white_canvas,
        recognized_texts,
        boxes_list,
        FLAGS_font_path,
        35,
        cv::Scalar(0, 0, 255),  // red (BGR)
        2
    );

    // Side-by-side: left = detected boxes, right = recognized text
    cv::Mat combined;
    cv::hconcat(img_boxes, img_with_text, combined);

    cv::imwrite(FLAGS_img_save_path, combined);
    std::cout << "[Saved] Result saved to: " << FLAGS_img_save_path << std::endl;

    return 0;
}
