// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_cpp_contract.cpp
 * @brief Optional board regression for the public YOLOE-26 C++ contract.
 *
 * This test intentionally uses a real HBM and image supplied by the caller.
 * It checks lifecycle behavior and the result shape that can be checked
 * independently of model accuracy.
 */

#include "yoloe26seg.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

/**
 * @brief Run lifecycle, malformed-input, and bbox-local-mask checks.
 * @param[in] argc Argument count; expects model and image paths.
 * @param[in] argv Argument vector.
 * @return 0 when all checks pass, otherwise 1.
 */
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: test_cpp_contract MODEL.hbm IMAGE\n";
        return 2;
    }
    try {
        yoloe26::YoloE26SegConfig config;
        config.model_path = argv[1];
        yoloe26::YoloE26Seg model(config);

        bool guarded = false;
        try {
            model.predict(cv::Mat(8, 8, CV_8UC3));
        } catch (const std::runtime_error&) {
            guarded = true;
        }
        require(guarded, "predict must reject an uninitialized model");

        const std::string missing =
            "/tmp/yoloe26_contract_missing_nashm_640x640_nv12.hbm";
        require(model.init(missing.c_str()) != 0,
                "missing HBM must fail without throwing");
        require(!model.initialized(), "failed init must leave model uninitialized");
        require(model.init() == 0, "model must initialize after a failed init");
        require(model.init() != 0, "repeated init must be rejected");

        const cv::Mat image = cv::imread(argv[2], cv::IMREAD_COLOR);
        require(!image.empty(), "test image could not be read");
        const InstanceSegResult result = model.predict(image);
        require(result.detections.size() == result.masks.size(),
                "detections and masks must stay index-aligned");
        for (size_t i = 0; i < result.detections.size(); ++i) {
            const auto& box = result.detections[i].bbox;
            const int x1 = std::clamp(static_cast<int>(box[0]), 0, image.cols);
            const int y1 = std::clamp(static_cast<int>(box[1]), 0, image.rows);
            const int x2 = std::clamp(static_cast<int>(box[2]), 0, image.cols);
            const int y2 = std::clamp(static_cast<int>(box[3]), 0, image.rows);
            const cv::Mat& mask = result.masks[i];
            if (x2 <= x1 || y2 <= y1) {
                require(mask.empty(), "degenerate boxes must retain empty masks");
                continue;
            }
            require(mask.type() == CV_8UC1 && mask.rows == y2 - y1 &&
                        mask.cols == x2 - x1,
                    "mask must be a bbox-local CV_8UC1 ROI");
            for (int y = 0; y < mask.rows; ++y) {
                for (int x = 0; x < mask.cols; ++x) {
                    require(mask.at<unsigned char>(y, x) <= 1,
                            "mask values must be 0 or 1");
                }
            }
        }

        // Shape validation must reject before attempting to read model memory.
        std::vector<hbDNNTensor> malformed(10);
        static float tiny_buffers[10]{};
        for (int i = 0; i < 10; ++i) {
            auto& tensor = malformed[i];
            tensor.properties.validShape.numDimensions = 4;
            tensor.properties.validShape.dimensionSize[0] = 1;
            tensor.properties.validShape.dimensionSize[1] = 1;
            tensor.properties.validShape.dimensionSize[2] = 1;
            tensor.properties.validShape.dimensionSize[3] = 1;
            tensor.properties.tensorType = HB_DNN_TENSOR_TYPE_F32;
            tensor.properties.stride[0] = 4;
            tensor.properties.stride[1] = 4;
            tensor.properties.stride[2] = 4;
            tensor.properties.stride[3] = 4;
            tensor.sysMem.virAddr = &tiny_buffers[i];
        }
        bool rejected = false;
        try {
            yoloe26::post_process(malformed, config, image.cols, image.rows);
        } catch (const std::invalid_argument& error) {
            rejected = std::string(error.what()) ==
                       "Invalid raw-v1 output tensor shape or memory";
        }
        require(rejected, "malformed output shape must be rejected safely");
        std::cout << "C++ contract checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
