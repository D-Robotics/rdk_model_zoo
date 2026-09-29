// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#include "tensor_contract.hpp"
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace unetmobilenet {
namespace {
std::size_t multiply(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max()/b)
        throw std::invalid_argument("Tensor size overflow");
    return a*b;
}
std::size_t add(std::size_t a, std::size_t b) {
    if (a > std::numeric_limits<std::size_t>::max()-b)
        throw std::invalid_argument("Tensor offset overflow");
    return a+b;
}
std::size_t axis_index(const ScoreSpec& spec, std::size_t y, std::size_t x, int c) {
    const int axis=spec.axis<0?spec.axis+4:spec.axis;
    const std::array<std::size_t,4> coordinates{0,y,x,static_cast<std::size_t>(c)};
    return spec.scales.size()==1?0:coordinates[axis];
}
}

void validate_scores(const ScoreSpec& spec) {
    if (spec.shape[0]!=1 || spec.shape[3]!=19 || spec.shape[1]<=0 || spec.shape[2]<=0)
        throw std::invalid_argument("Expected NHWC [1,H,W,19] scores");
    for (auto stride:spec.stride)
        if (stride<=0) throw std::invalid_argument("Unresolved or invalid output stride");
    if (spec.stride[3]<4) throw std::invalid_argument("Score elements overlap");
    for (int i=0;i<3;++i)
        if (static_cast<std::size_t>(spec.stride[i]) < multiply(spec.stride[i+1],spec.shape[i+1]))
            throw std::invalid_argument("Tensor rows or channels overlap");
    std::size_t span=4;
    for (int i=0;i<4;++i) span=add(span,multiply(spec.shape[i]-1,spec.stride[i]));
    if (span>spec.storage_bytes) throw std::invalid_argument("Score tensor exceeds allocation");
    if (spec.type==ScoreType::Int32 && spec.scaled) {
        if (spec.scales.empty()) throw std::invalid_argument("Missing SCALE values");
        for (float scale:spec.scales)
            if (!std::isfinite(scale) || scale<=0) throw std::invalid_argument("Invalid SCALE value");
        if (spec.scales.size()==1) {
            if (spec.zero_points.size()>1) throw std::invalid_argument("Scalar scale needs scalar offset");
        } else {
            int axis=spec.axis<0?spec.axis+4:spec.axis;
            if (axis<0 || axis>=4 || spec.scales.size()!=static_cast<std::size_t>(spec.shape[axis]))
                throw std::invalid_argument("SCALE axis/length mismatch");
            if (!spec.zero_points.empty() && spec.zero_points.size()!=1 && spec.zero_points.size()!=spec.scales.size())
                throw std::invalid_argument("Zero-point length mismatch");
        }
    }
}

std::vector<std::int32_t> decode_scores(const RawScores& raw, int height, int width) {
    validate_scores(raw.spec);
    if (height<=0 || width<=0 || raw.bytes.size()<raw.spec.storage_bytes)
        throw std::invalid_argument("Invalid original geometry or truncated scores");
    const auto& spec=raw.spec;
    std::vector<std::int32_t> small(multiply(spec.shape[1],spec.shape[2]));
    for (std::size_t y=0;y<static_cast<std::size_t>(spec.shape[1]);++y) {
        for (std::size_t x=0;x<static_cast<std::size_t>(spec.shape[2]);++x) {
            double best=-std::numeric_limits<double>::infinity();
            int winner=0;
            for (int c=0;c<19;++c) {
                const auto offset=y*spec.stride[1]+x*spec.stride[2]+c*spec.stride[3];
                double value;
                if (spec.type==ScoreType::Int32) {
                    std::int32_t integer=0;
                    std::memcpy(&integer,raw.bytes.data()+offset,4);
                    value=integer;
                    if (spec.scaled) {
                        const auto qi=axis_index(spec,y,x,c);
                        const auto zero=spec.zero_points.empty()?0:spec.zero_points[spec.zero_points.size()==1?0:qi];
                        value=(value-static_cast<double>(zero))*spec.scales[qi];
                    }
                } else {
                    float floating=0;
                    std::memcpy(&floating,raw.bytes.data()+offset,4);
                    value=floating;
                }
                if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite output score");
                if (value>best) { best=value;winner=c; }
            }
            small[y*spec.shape[2]+x]=winner;
        }
    }
    std::vector<std::int32_t> restored(multiply(height,width));
    for (int y=0;y<height;++y) {
        const auto source_y=multiply(y,spec.shape[1])/height;
        for (int x=0;x<width;++x) {
            const auto source_x=multiply(x,spec.shape[2])/width;
            restored[static_cast<std::size_t>(y)*width+x]=small[source_y*spec.shape[2]+source_x];
        }
    }
    return restored;
}
}  // namespace unetmobilenet
