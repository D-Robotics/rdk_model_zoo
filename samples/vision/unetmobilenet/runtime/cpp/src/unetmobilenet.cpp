// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#include "unetmobilenet.hpp"
#include <cstring>
#include <stdexcept>
#include <utility>
#include <opencv2/imgproc.hpp>

namespace unetmobilenet {
UnetMobileNetTask::UnetMobileNetTask(RawRunner runner):runner_(std::move(runner)) {
    if(!runner_)throw std::invalid_argument("A raw runner is required");
}
PreparedInput UnetMobileNetTask::pre_process(const cv::Mat& image) const {
    if(image.empty() || image.type()!=CV_8UC3)throw std::invalid_argument("Expected nonempty BGR uint8 image");
    cv::Mat resized,i420;
    cv::resize(image,resized,cv::Size(2048,1024),0,0,cv::INTER_AREA);
    cv::cvtColor(resized,i420,cv::COLOR_BGR2YUV_I420);
    PreparedInput result{cv::Mat(1024,2048,CV_8UC1),cv::Mat(512,1024,CV_8UC2),{image.rows,image.cols}};
    const auto* source=i420.ptr<unsigned char>();
    std::memcpy(result.y.data,source,1024*2048);
    const auto* u=source+1024*2048;
    const auto* v=u+512*1024;
    auto* uv=result.uv.ptr<unsigned char>();
    for(int i=0;i<512*1024;++i) { uv[2*i]=u[i];uv[2*i+1]=v[i]; }
    return result;
}
RawScores UnetMobileNetTask::forward(const PreparedInput& prepared) const {
    return runner_(prepared.y,prepared.uv);
}
cv::Mat UnetMobileNetTask::post_process(const RawScores& raw,const ImageContext& context) const {
    const auto labels=decode_scores(raw,context.original_height,context.original_width);
    cv::Mat result(context.original_height,context.original_width,CV_32S);
    std::memcpy(result.data,labels.data(),labels.size()*sizeof(std::int32_t));
    return result;
}
cv::Mat UnetMobileNetTask::predict(const cv::Mat& image) const {
    const auto prepared=pre_process(image);
    return post_process(forward(prepared),prepared.context);
}
}  // namespace unetmobilenet
