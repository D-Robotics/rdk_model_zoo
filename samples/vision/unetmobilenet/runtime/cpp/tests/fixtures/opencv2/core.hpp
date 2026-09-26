// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Minimal host test double, NOT an OpenCV or SDK compatibility certificate.
#include <vector>
#define CV_8UC1 1
#define CV_8UC2 2
namespace cv {
class Mat {
    int channels_;
    std::vector<unsigned char> bytes_;
public:
    int rows,cols;
    Mat(int h,int w,int channels):channels_(channels),bytes_(h*w*channels),rows(h),cols(w){}
    int type() const{return channels_;}
    const unsigned char* ptr(int row) const{return bytes_.data()+row*cols*channels_;}
};
}
