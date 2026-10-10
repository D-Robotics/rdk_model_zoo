// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "yolo.hpp"
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

// Explicit checks remain enabled even when the build defines NDEBUG.
#undef assert
#define assert(value) do { if (!(value)) throw std::runtime_error(#value); } while (0)

template<class F> void rejects(F fn) {
  bool rejected = false;
  try { fn(); } catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
}
int main() {
  using yolo::classification_plan;
  for (const auto& shape : std::vector<std::vector<int>>{{1000},{1,1000},{1,1000,1,1},{1,1,1,1000}}) {
    std::vector<size_t> strides(shape.size(),4);
    for (int i=static_cast<int>(shape.size())-2;i>=0;--i) strides[i]=strides[i+1]*shape[i+1];
    auto plan=classification_plan(shape,strides,4000,true,true);
    std::vector<float> raw(1000,0); raw[999]=8; raw[2]=8;
    auto ranked=yolo::classification_topk(raw.data(),4000,plan,5);
    assert(ranked.size()==5 && ranked[0].id==2 && ranked[1].id==999);
    assert(std::abs(ranked[0].probability-(std::exp(8.)/(2*std::exp(8.)+998)))<1e-6);
  }
  // Strided class axis must skip padding, even if padding is NaN.
  std::vector<float> raw(4000,std::numeric_limits<float>::quiet_NaN());
  for (int i=0;i<1000;++i) raw[i*4]=i==177 ? 1000 : -1000;
  auto padded=classification_plan({1,1000,1,1},{16000,16,8,4},16000,true,true);
  auto ranked=yolo::classification_topk(raw.data(),16000,padded,5);
  assert(ranked[0].id==177 && ranked[0].probability==1);
  rejects([&]{classification_plan({2,1000},{4000,4},8000,true,true);});
  rejects([&]{classification_plan({1,10,100},{4000,400,4},4000,true,true);});
  rejects([&]{classification_plan({1,999},{3996,4},4000,true,true);});
  rejects([&]{classification_plan({1,1000},{4000,4},3999,true,true);});
  rejects([&]{classification_plan({1,1000},{4000,3},4000,true,true);});
  rejects([&]{classification_plan({1,1000},{4000,0},4000,true,true);});
  rejects([&]{classification_plan({1,1000},{4000,4},4000,false,true);});
  rejects([&]{classification_plan({1,1000},{4000,4},4000,true,false);});
  rejects([&]{classification_plan({1,1000},{4000,std::numeric_limits<size_t>::max()-3},4000,true,true);});
  rejects([&]{yolo::classification_topk(raw.data(),15,padded,5);});
  rejects([&]{yolo::classification_topk(nullptr,16000,padded,5);});
  rejects([&]{yolo::classification_topk(raw.data(),16000,padded,1001);});
  raw[0]=std::numeric_limits<float>::infinity();
  rejects([&]{yolo::classification_topk(raw.data(),16000,padded,5);});
  raw[0]=std::numeric_limits<float>::quiet_NaN();
  rejects([&]{yolo::classification_topk(raw.data(),16000,padded,5);});
}
