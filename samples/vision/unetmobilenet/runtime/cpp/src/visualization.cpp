// Copyright (c) 2025-2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#include "visualization.hpp"
#include <cmath>
#include <stdexcept>
#include <opencv2/imgproc.hpp>
namespace unetmobilenet {
cv::Mat render_overlay(const cv::Mat& image,const cv::Mat& labels,double alpha_f) {
    if(image.empty() || image.type()!=CV_8UC3 || labels.type()!=CV_32S || labels.size()!=image.size())
        throw std::invalid_argument("Overlay requires original-sized BGR image and int32 labels");
    if(!std::isfinite(alpha_f) || alpha_f<0 || alpha_f>1)throw std::invalid_argument("alpha-f must be 0..1");
    static const cv::Vec3b colors[19]={
        {56,56,255},{151,157,255},{31,112,255},{29,178,255},{49,210,207},
        {10,249,72},{23,204,146},{134,219,61},{52,147,26},{187,212,0},
        {168,153,44},{255,194,0},{147,69,52},{255,115,100},{236,24,0},
        {255,56,132},{133,0,82},{255,56,203},{200,149,255}
    };
    cv::Mat colored(image.size(),CV_8UC3),result;
    for(int y=0;y<labels.rows;++y)for(int x=0;x<labels.cols;++x) {
        const int id=labels.at<std::int32_t>(y,x);
        if(id<0 || id>=19)throw std::invalid_argument("Class ID is outside 0..18");
        colored.at<cv::Vec3b>(y,x)=colors[id];
    }
    cv::addWeighted(image,alpha_f,colored,1-alpha_f,0,result);
    return result;
}
}
