// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "tensor_contract.hpp"
#include <memory>
#include <functional>
#include <string>
#include <opencv2/core.hpp>

namespace unetmobilenet {
// SDK/resource owner. A failed constructor or forward releases only allocated
// buffers/handles; the task itself does not manage SDK lifetimes.
class ModelRunner {
public:
    using ExecutionGate=std::function<void(const std::string&,const std::string&)>;
    // An explicit gate is a host-test seam, never exposed as a CLI bypass.
    ModelRunner(const std::string& model_path, const std::string& target,
                int priority=0, int bpu_core=-1, ExecutionGate gate={});
    ~ModelRunner();
    ModelRunner(const ModelRunner&)=delete;
    ModelRunner& operator=(const ModelRunner&)=delete;
    RawScores run(const cv::Mat& y, const cv::Mat& uv);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace unetmobilenet
