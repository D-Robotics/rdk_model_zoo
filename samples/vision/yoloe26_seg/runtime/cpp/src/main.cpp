// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#include "yoloe26seg.hpp"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>
void render_result(const cv::Mat& source, const std::vector<std::string>& names,
                   const yoloe26::Result& result, const std::string& output_path) {
    cv::Mat image = source.clone();
    for (size_t index = 0; index < result.detections.size(); ++index) {
        const auto& detection = result.detections[index];
        const cv::Scalar color((detection.label * 47) % 200 + 40,
                               (detection.label * 83) % 200 + 40,
                               (detection.label * 131) % 200 + 40);
        cv::Mat blended;
        cv::addWeighted(image, .6, cv::Mat(image.size(), image.type(), color), .4, 0, blended);
        blended.copyTo(image, result.masks[index]);
        cv::Point a(cvRound(detection.box[0]), cvRound(detection.box[1]));
        cv::Point b(cvRound(detection.box[2]), cvRound(detection.box[3]));
        cv::rectangle(image, a, b, color, 2);
        cv::putText(image, names[detection.label] + " " + std::to_string(detection.score), a,
                    cv::FONT_HERSHEY_SIMPLEX, .45, color, 1);
        std::cout << detection.label << ' ' << detection.score;
        for (float value : detection.box) std::cout << ' ' << value;
        std::cout << '\n';
    }
    if (!cv::imwrite(output_path, image)) throw std::runtime_error("Cannot write output image");
}

int main(int argc, char** argv) {
    if (argc < 5 || argc > 8) {
        std::cerr << "Usage: yoloe26seg model.hbm labels.names image output.jpg [score=.25] [max_det=300] [multi_label=0]\n";
        return 2;
    }
    try {
        std::ifstream stream(argv[2]);
        std::vector<std::string> names;
        std::string name;
        while (std::getline(stream, name)) {
            if (!name.empty() && name.back() == '\r') name.pop_back();
            names.push_back(name);
        }
        if (names.size() != 4585) throw std::runtime_error("Expected the matching 4585 labels");
        cv::Mat image = cv::imread(argv[3]);
        if (image.empty()) throw std::runtime_error("Cannot read image");
        const float score = argc > 5 ? std::stof(argv[5]) : .25f;
        const int max_det = argc > 6 ? std::stoi(argv[6]) : 300;
        const int multi = argc > 7 ? std::stoi(argv[7]) : 0;
        if (multi != 0 && multi != 1) throw std::runtime_error("multi_label must be 0 or 1");
        yoloe26::YoloE26Seg model(argv[1]);
        render_result(image, names, model.predict(image, score, max_det, !multi), argv[4]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
