// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#include "tensor_contract.hpp"
#include "target_identity.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace unetmobilenet;

template<class F> void rejects(F action) {
    bool rejected=false;
    try { action(); } catch (const std::invalid_argument&) { rejected=true; }
    assert(rejected);
}
int main() {
    assert(match_native_target("S100\n","RDK S100P")=="s100p");
    assert(match_native_target("s100","s100p")=="s100p");
    assert(match_native_target("s600","")=="s600");
    assert(match_native_target("(null)","").empty());
    RawScores raw;
    raw.spec.shape={1,1,3,19};
    raw.spec.stride={256,256,80,4};  // 4-byte padding after each pixel
    raw.spec.storage_bytes=256;
    raw.bytes.resize(256,0);
    for (int x=0;x<3;++x) {
        std::int32_t value=7;
        std::memcpy(raw.bytes.data()+x*80+x*4,&value,4);
    }
    auto labels=decode_scores(raw,1,6);
    assert((labels==std::vector<std::int32_t>{0,0,1,1,2,2}));
    auto unchanged=raw.bytes;
    raw.spec.scaled=true;raw.spec.scales.assign(19,1.F);raw.spec.scales[1]=100.F;
    raw.spec.zero_points={0};
    std::int32_t a=10,b=2;
    std::memcpy(raw.bytes.data(),&a,4);std::memcpy(raw.bytes.data()+4,&b,4);
    assert(decode_scores(raw,1,3)[0]==1);
    unchanged=raw.bytes;decode_scores(raw,2,7);assert(raw.bytes==unchanged);
    raw.spec.scales={1.F};raw.spec.zero_points={0};
    a=33554432;b=33554433;
    std::memcpy(raw.bytes.data(),&a,4);std::memcpy(raw.bytes.data()+4,&b,4);
    assert(decode_scores(raw,1,3)[0]==1);  // no F32 rounding tie
    std::memcpy(raw.bytes.data(),&b,4);assert(decode_scores(raw,1,3)[0]==0);
    for (int which=0;which<5;++which) {
        auto bad=raw;
        if(which==0)bad.spec.storage_bytes=1;
        if(which==1)bad.spec.stride[2]=4;
        if(which==2)bad.spec.shape[3]=18;
        if(which==3)bad.spec.scales={0.F};
        if(which==4)bad.bytes.clear();
        rejects([&]{decode_scores(bad,1,3);});
    }
    rejects([&]{decode_scores(raw,0,3);});
    raw.spec.type=ScoreType::Float32;raw.spec.scaled=false;
    std::fill(raw.bytes.begin(),raw.bytes.end(),0);
    float positive=0.7F;
    std::memcpy(raw.bytes.data()+12*4,&positive,4);
    assert(decode_scores(raw,1,3)[0]==12);
    float invalid=std::numeric_limits<float>::quiet_NaN();
    std::memcpy(raw.bytes.data(),&invalid,4);
    rejects([&]{decode_scores(raw,1,3);});
    std::cout<<"native score/stride/quantization/resize cases passed\n";
}
