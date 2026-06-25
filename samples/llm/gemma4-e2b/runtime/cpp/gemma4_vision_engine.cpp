#include "gemma4_vision_engine.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "gemma4_config.h"
#include "gemma4_vision_preprocess.h"
#include "hb_utils.h"

namespace gemma4 {

namespace {

uint16_t FloatToF16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000;
  int32_t exp = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
  uint32_t mant = bits & 0x7fffff;

  if (exp <= 0) {
    if (exp < -10) {
      return static_cast<uint16_t>(sign);
    }
    mant = (mant | 0x800000) >> (1 - exp);
    return static_cast<uint16_t>(sign | (mant >> 13));
  }
  if (exp >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00);
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) |
                                 (mant >> 13));
}

}  // namespace

VisionEngine::VisionEngine(const std::string& vision_hbm) {
  const char* path = vision_hbm.c_str();
  const char* paths[] = {path};

  auto t0 = std::chrono::steady_clock::now();
  HBDNN_CHECK(hbDNNInitializeFromFiles(&packed_, paths, 1), "load vision hbm");
  auto t1 = std::chrono::steady_clock::now();
  load_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();

  HBDNN_CHECK(hbDNNGetModelHandle(&handle_, packed_, "Gemma4VisionModel"),
              "get Gemma4VisionModel");

  int input_count = 0;
  HBDNN_CHECK(hbDNNGetInputCount(&input_count, handle_), "vision input count");
  inputs_.reserve(static_cast<size_t>(input_count));
  for (int i = 0; i < input_count; ++i) {
    inputs_.push_back(MakeTensor(handle_, true, i));
  }

  int output_count = 0;
  HBDNN_CHECK(hbDNNGetOutputCount(&output_count, handle_), "vision output count");
  outputs_.reserve(static_cast<size_t>(output_count));
  for (int i = 0; i < output_count; ++i) {
    outputs_.push_back(MakeTensor(handle_, false, i));
  }
}

VisionEngine::~VisionEngine() {
  FreeTensors(inputs_);
  FreeTensors(outputs_);
  if (packed_) {
    hbDNNRelease(packed_);
  }
}

std::vector<float> VisionEngine::Infer(const std::string& image_path) {
  const std::vector<float> patches = PreprocessImage(image_path);

  // Debug: patch statistics
  {
    double sum = 0, sq = 0;
    float pmin = patches[0], pmax = patches[0];
    for (size_t i = 0; i < patches.size(); ++i) {
      sum += patches[i];
      sq += patches[i] * patches[i];
      if (patches[i] < pmin) pmin = patches[i];
      if (patches[i] > pmax) pmax = patches[i];
    }
    double mean = sum / patches.size();
    double var = sq / patches.size() - mean * mean;
    std::cerr << "[DEBUG] patches: size=" << patches.size()
              << " min=" << pmin << " max=" << pmax
              << " mean=" << mean << " std=" << std::sqrt(var) << std::endl;
  }

  std::vector<uint16_t> f16(patches.size());
  for (size_t i = 0; i < patches.size(); ++i) {
    f16[i] = FloatToF16(patches[i]);
  }

  // Debug: print input tensor properties
  {
    const auto& props = inputs_[0].properties;
    std::cerr << "[DEBUG] input tensor: type=" << props.tensorType
              << " ndim=" << props.validShape.numDimensions
              << " shape=[";
    for (int d = 0; d < props.validShape.numDimensions; ++d) {
      if (d) std::cerr << ",";
      std::cerr << props.validShape.dimensionSize[d];
    }
    std::cerr << "] aligned_bytes=" << props.alignedByteSize << std::endl;
  }

  WriteInputTensor(inputs_[0], f16.data());
  RunInfer(handle_, inputs_, outputs_);

  const auto& props = outputs_[0].properties;
  const int64_t elems =
      ProdSize(props.validShape.dimensionSize, props.validShape.numDimensions);

  // Debug: output tensor properties
  std::cerr << "[DEBUG] output tensor: type=" << props.tensorType
            << " ndim=" << props.validShape.numDimensions
            << " shape=[";
  for (int d = 0; d < props.validShape.numDimensions; ++d) {
    if (d) std::cerr << ",";
    std::cerr << props.validShape.dimensionSize[d];
  }
  std::cerr << "] elems=" << elems << " elem_size=" << ElementSize(props.tensorType)
            << " aligned_bytes=" << props.alignedByteSize << std::endl;

  std::vector<float> out(static_cast<size_t>(elems));
  const void* raw_out = outputs_[0].sysMem.virAddr;

  // Handle different output tensor types
  if (props.tensorType == HB_DNN_TENSOR_TYPE_F32) {
    const auto* src = static_cast<const float*>(raw_out);
    std::copy(src, src + elems, out.data());
  } else if (props.tensorType == HB_DNN_TENSOR_TYPE_F16) {
    const auto* src = static_cast<const uint16_t*>(raw_out);
    for (int64_t i = 0; i < elems; ++i) {
      uint16_t h = src[static_cast<size_t>(i)];
      uint32_t sign = (h >> 15) & 1;
      uint32_t exp = (h >> 10) & 0x1f;
      uint32_t mant = h & 0x3ff;
      float val;
      if (exp == 0) {
        val = mant ? static_cast<float>(mant) / 1024.f * std::ldexp(1.f, -14) : 0.f;
      } else if (exp == 31) {
        val = mant ? NAN : (sign ? -INFINITY : INFINITY);
      } else {
        val = std::ldexp(1.f + static_cast<float>(mant) / 1024.f,
                         static_cast<int>(exp) - 15);
      }
      out[static_cast<size_t>(i)] = sign ? -val : val;
    }
  } else {
    // Fallback: try float
    std::cerr << "[DEBUG] WARNING: unexpected tensor type " << props.tensorType
              << ", trying float cast" << std::endl;
    const auto* src = static_cast<const float*>(raw_out);
    std::copy(src, src + elems, out.data());
  }

  // Debug: output statistics
  {
    double sum = 0, sq = 0;
    float omin = out[0], omax = out[0];
    for (size_t i = 0; i < out.size(); ++i) {
      sum += out[i];
      sq += out[i] * out[i];
      if (out[i] < omin) omin = out[i];
      if (out[i] > omax) omax = out[i];
    }
    double mean = sum / out.size();
    double var = sq / out.size() - mean * mean;
    std::cerr << "[DEBUG] vision output: size=" << out.size()
              << " min=" << omin << " max=" << omax
              << " mean=" << mean << " std=" << std::sqrt(var) << std::endl;
  }

  return out;
}

}  // namespace gemma4
