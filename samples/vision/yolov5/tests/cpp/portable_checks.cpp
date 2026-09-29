// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Host-compilable behaviour checks for the YOLOv5 native numeric core. The
// target SDK is absent on a host, so the checks drive the SDK-free gate, decoder
// and dump modules with explicit metadata values instead of asserting on source
// text. Usage: portable_checks <check-name> [scratch-dir]

#include "yolov5_decode.hpp"
#include "yolov5_dump.hpp"
#include "yolov5_gate.hpp"
#include "yolov5_s_native.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cout << "FAIL: " << message << "\n";
    ++failures;
  }
}

yolov5::TensorMeta nv12_input(long long size, long long aligned_width = -1,
                              long long aligned_bytes = -1, long long storage = -1) {
  yolov5::TensorMeta meta;
  meta.num_dimensions = 4;
  meta.valid[0] = 1; meta.valid[1] = 3; meta.valid[2] = size; meta.valid[3] = size;
  meta.aligned[0] = 1; meta.aligned[1] = 3;
  meta.aligned[2] = size;
  meta.aligned[3] = aligned_width < 0 ? size : aligned_width;
  const long long required = size * size * 3 / 2;
  meta.aligned_byte_size = aligned_bytes < 0 ? required : aligned_bytes;
  meta.storage_bytes = storage < 0 ? required : storage;
  return meta;
}

yolov5::TensorMeta f32_head(long long input_size, long long classes, long long height,
                            long long aligned_height = -1, long long aligned_bytes = -1,
                            long long storage = -1) {
  yolov5::TensorMeta meta;
  meta.dtype = yolov5::kDtypeF32;
  meta.quanti_type = yolov5::kQuantiNone;
  meta.num_dimensions = 4;
  const long long channels = 3 * (5 + classes);
  meta.valid[0] = 1; meta.valid[1] = height; meta.valid[2] = height; meta.valid[3] = channels;
  meta.aligned[0] = 1; meta.aligned[1] = aligned_height < 0 ? height : aligned_height;
  meta.aligned[2] = height; meta.aligned[3] = channels;
  const long long required = height * height * channels * 4;
  meta.aligned_byte_size = aligned_bytes < 0 ? required : aligned_bytes;
  meta.storage_bytes = storage < 0 ? required : storage;
  (void)input_size;
  return meta;
}

yolov5::TensorMeta s32_output(long long height, long long width, long long channels,
                              int dtype = yolov5::kDtypeS32, int quanti = yolov5::kQuantiScale,
                              long long scale_len = 1, long long zero_point_len = 0,
                              long long row_stride = -1, long long storage = -1,
                              long long channel_stride = 4, long long pixel_stride = -1,
                              long long aligned_bytes = -1, long long axis = 3) {
  yolov5::TensorMeta meta;
  meta.dtype = dtype;
  meta.quanti_type = quanti;
  meta.num_dimensions = 4;
  meta.valid[0] = 1; meta.valid[1] = height; meta.valid[2] = width; meta.valid[3] = channels;
  meta.scale_len = scale_len;
  meta.zero_point_len = zero_point_len;
  meta.quantize_axis = axis;
  meta.stride[3] = channel_stride;
  meta.stride[2] = pixel_stride < 0 ? channels * channel_stride : pixel_stride;
  meta.stride[1] = row_stride < 0 ? width * meta.stride[2] : row_stride;
  meta.stride[0] = height * meta.stride[1];
  // Allocation needed for the addressing formula's last byte: the final
  // element of the final pixel of the final row.
  const long long required =
      (height * width - 1) * meta.stride[2] + (channels - 1) * channel_stride + 4;
  meta.aligned_byte_size = aligned_bytes < 0 ? required : aligned_bytes;
  meta.storage_bytes = storage < 0 ? meta.aligned_byte_size : storage;
  return meta;
}

void check_x5_input() {
  using namespace yolov5;
  expect(static_cast<bool>(check_x5_nv12_input(kImageNv12, nv12_input(640), 640)),
         "compact NV12 input should be accepted");
  expect(!check_x5_nv12_input(kImageUnknown, nv12_input(640), 640),
         "unknown image type must be rejected");
  TensorMeta rank3 = nv12_input(640);
  rank3.num_dimensions = 3;
  expect(!check_x5_nv12_input(kImageNv12, rank3, 640), "rank-3 input must be rejected");
  TensorMeta wrong = nv12_input(640);
  wrong.valid[1] = 1;
  expect(!check_x5_nv12_input(kImageNv12, wrong, 640),
         "non-NV12 channel count must be rejected");
  expect(!check_x5_nv12_input(kImageNv12, nv12_input(640, /*aligned_width=*/672), 640),
         "padded NV12 aligned layout must be rejected");
  expect(!check_x5_nv12_input(kImageNv12, nv12_input(640, 640, /*aligned_bytes=*/100), 640),
         "undersized alignedByteSize must be rejected");
  expect(!check_x5_nv12_input(kImageNv12, nv12_input(640, 640, -1, /*storage=*/100), 640),
         "undersized allocation must be rejected");
  TensorMeta unreported = nv12_input(640);
  for (int i = 0; i < 4; ++i) unreported.aligned[i] = 0;
  expect(static_cast<bool>(check_x5_nv12_input(kImageNv12, unreported, 640)),
         "an unreported aligned layout must not be treated as padding");
}

void check_x5_head() {
  using namespace yolov5;
  int stride = 0;
  expect(static_cast<bool>(check_x5_head(f32_head(640, 80, 80), 640, 80, &stride)) && stride == 8,
         "80x80 head should be accepted as stride 8");
  expect(static_cast<bool>(check_x5_head(f32_head(640, 80, 40), 640, 80, &stride)) && stride == 16,
         "40x40 head should be accepted as stride 16");
  expect(static_cast<bool>(check_x5_head(f32_head(640, 80, 20), 640, 80, &stride)) && stride == 32,
         "20x20 head should be accepted as stride 32");

  TensorMeta s32 = f32_head(640, 80, 80);
  s32.dtype = kDtypeS32;
  expect(!check_x5_head(s32, 640, 80, &stride), "non-F32 X5 head must be rejected");
  TensorMeta scaled = f32_head(640, 80, 80);
  scaled.quanti_type = kQuantiScale;
  expect(!check_x5_head(scaled, 640, 80, &stride), "quantized X5 head must be rejected");
  TensorMeta bad_channels = f32_head(640, 80, 80);
  bad_channels.valid[3] = 84;
  expect(!check_x5_head(bad_channels, 640, 80, &stride),
         "channels other than 3*(5+classes) must be rejected");
  TensorMeta padded = f32_head(640, 80, 80, /*aligned_height=*/96);
  expect(!check_x5_head(padded, 640, 80, &stride),
         "padded head aligned layout must be rejected");
  TensorMeta unreported = f32_head(640, 80, 80);
  for (int i = 0; i < 4; ++i) unreported.aligned[i] = 0;
  expect(static_cast<bool>(check_x5_head(unreported, 640, 80, &stride)),
         "a head without a reported aligned layout should still be accepted");
  expect(!check_x5_head(f32_head(640, 80, 80, 80, /*aligned_bytes=*/100), 640, 80, &stride),
         "head alignedByteSize below its float elements must be rejected");
  expect(!check_x5_head(f32_head(640, 80, 80, 80, -1, /*storage=*/100), 640, 80, &stride),
         "head allocation below its float elements must be rejected");
  TensorMeta odd = f32_head(640, 80, 64);
  expect(!check_x5_head(odd, 640, 80, &stride),
         "head whose stride is not 8/16/32 must be rejected");
}

void check_s_plane() {
  using namespace yolov5;
  auto plane = [](long long rows, long long cols, long long channels, long long row_stride,
                  long long storage) {
    TensorMeta meta;
    meta.num_dimensions = 4;
    meta.valid[0] = 1; meta.valid[1] = rows; meta.valid[2] = cols; meta.valid[3] = channels;
    meta.stride[3] = 1;
    meta.stride[2] = channels;
    meta.stride[1] = row_stride;
    meta.stride[0] = rows * row_stride;
    meta.storage_bytes = storage;
    meta.aligned_byte_size = storage;
    return meta;
  };
  expect(static_cast<bool>(check_s_nv12_plane(plane(672, 672, 1, 672, 672 * 672), 672, 672, 1)),
         "compact Y plane should be accepted");
  expect(static_cast<bool>(check_s_nv12_plane(plane(336, 336, 2, 672, 336 * 672), 336, 336, 2)),
         "compact UV plane should be accepted");
  expect(!check_s_nv12_plane(plane(672, 672, 1, 512, 672 * 512), 672, 672, 1),
         "row stride shorter than a row must be rejected");
  expect(!check_s_nv12_plane(plane(672, 672, 1, 672, 100), 672, 672, 1),
         "allocation shorter than the plane must be rejected");
  expect(!check_s_nv12_plane(plane(672, 672, 1, 672, 672 * 672), 640, 640, 1),
         "mismatched logical shape must be rejected");
}

void check_s_dequant() {
  using namespace yolov5;
  long long count = 0;
  expect(static_cast<bool>(check_s32_dequant(s32_output(84, 84, 84), &count)) && count == 84ll * 84 * 84,
         "contiguous S32 SCALE output should be accepted");
  expect(static_cast<bool>(check_s32_dequant(
             s32_output(84, 84, 84, kDtypeS32, kQuantiScale, /*scale_len=*/84), &count)),
         "per-channel scale descriptor should be accepted");

  expect(!check_s32_dequant(s32_output(84, 84, 84, kDtypeF32, kQuantiScale), &count),
         "SCALE output that is not S32 must be rejected");
  expect(!check_s32_dequant(s32_output(84, 84, 84, kDtypeS32, kQuantiScale, /*scale_len=*/0), &count),
         "SCALE output without a scale descriptor must be rejected");
  expect(!check_s32_dequant(s32_output(84, 84, 84, kDtypeS32, kQuantiScale, /*scale_len=*/10), &count),
         "scale descriptor shorter than the channels must be rejected");
  expect(!check_s32_dequant(s32_output(84, 84, 84, kDtypeS32, kQuantiShift), &count),
         "unsupported quantization type must be rejected");
  expect(!check_s32_dequant(s32_output(84, 84, 84, kDtypeS32, kQuantiScale, 1, 0,
                                       /*row_stride=*/100), &count),
         "a stride[1] that is not width*stride[2] must be rejected");
  expect(!check_s32_dequant(s32_output(84, 84, 84, kDtypeS32, kQuantiScale, 1, 0, -1,
                                       /*storage=*/100), &count),
         "S32 allocation smaller than the stored extent must be rejected");

  // Row padding is genuinely supported by the addressing formula: it reads
  // (h, w, c) at (h*W + w) * stride[2] + c * stride[3], so a pixel stride
  // larger than the compact pixel is read correctly as long as stride[1]
  // stays width*stride[2] and the allocation covers the padded extent.
  TensorMeta row_padded = s32_output(84, 84, 84, kDtypeS32, kQuantiScale, 1, 0,
                                     /*row_stride=*/-1, /*storage=*/-1,
                                     /*channel_stride=*/4, /*pixel_stride=*/28256);
  expect(static_cast<bool>(check_s32_dequant(row_padded, &count)) && count == 84ll * 84 * 84,
         "row-padded S32 layout must be accepted (stride[2] > compact row)");
  TensorMeta padded_short = s32_output(84, 84, 84, kDtypeS32, kQuantiScale, 1, 0,
                                       /*row_stride=*/-1, /*storage=*/-1,
                                       /*channel_stride=*/4, /*pixel_stride=*/28256,
                                       /*aligned_bytes=*/84 * 84 * 84 * 4);
  expect(!check_s32_dequant(padded_short, &count),
         "row-padded layout whose allocation cannot cover the stored extent must be rejected");
  expect(!check_s32_dequant(s32_output(84, 84, 84, kDtypeS32, kQuantiScale, 1, 0,
                                       /*row_stride=*/85 * 84 * 4), &count),
         "H-level padding (stride[1] > width*stride[2]) must be rejected");
  expect(static_cast<bool>(check_s32_dequant(s32_output(84, 84, 84, kDtypeS32, kQuantiScale,
                                                        1, 0, -1, -1,
                                                        /*channel_stride=*/8), &count)),
         "an element-aligned channel stride above the element size must be accepted");
  expect(!check_s32_dequant(s32_output(84, 84, 84, kDtypeS32, kQuantiScale, 1, 0, -1, -1,
                                       /*channel_stride=*/6), &count),
         "a channel stride that is not a multiple of the element size must be rejected");
  expect(!check_s32_dequant(s32_output(84, 84, 84, kDtypeS32, kQuantiScale, 1, 0, -1, -1,
                                       /*channel_stride=*/2), &count),
         "a channel stride below the element size must be rejected");

  // The real published S100 model reports exactly this layout for its large
  // head (independent probe 2026-09-24): 255 channels, pixel stride 1024,
  // row stride width*1024. The padding is legal and must be accepted.
  TensorMeta published = s32_output(84, 84, 255, kDtypeS32, kQuantiScale, 255, 0,
                                    /*row_stride=*/86016, /*storage=*/7225344,
                                    /*channel_stride=*/4, /*pixel_stride=*/1024,
                                    /*aligned_bytes=*/7225344);
  expect(static_cast<bool>(check_s32_dequant(published, &count)) && count == 84ll * 84 * 255,
         "the published S100 head layout (pixel stride 1024) must be accepted");
  // The probe's overlapping counterexample: 400 < 255*4 makes consecutive
  // pixels share bytes and must be rejected (a width*stride[3] bound would
  // wrongly accept it because 400 >= 84*4).
  expect(!check_s32_dequant(s32_output(84, 84, 255, kDtypeS32, kQuantiScale, 255, 0,
                                       /*row_stride=*/33600, /*storage=*/-1,
                                       /*channel_stride=*/4, /*pixel_stride=*/400), &count),
         "an overlapping pixel layout must be rejected");
  TensorMeta overflow = s32_output(84, 84, 255, kDtypeS32, kQuantiScale, 255, 0,
                                   /*row_stride=*/0, /*storage=*/-1,
                                   /*channel_stride=*/4, /*pixel_stride=*/(1LL << 62),
                                   /*aligned_bytes=*/1LL << 40);
  overflow.stride[0] = 0;
  expect(!check_s32_dequant(overflow, &count),
         "a layout whose extent overflows must be rejected, not wrap around");
  expect(!check_s32_dequant(s32_output(1LL << 21, 1LL << 21, 1LL << 21, kDtypeS32,
                                       kQuantiScale, 1, 0, -1, -1, 4, -1,
                                       /*aligned_bytes=*/1LL << 40), &count),
         "an element count that overflows must be rejected, not wrap around");

  // Per-channel descriptors are only readable when the SDK declares the
  // quantization axis as the channel axis; scalar descriptors are axis-free.
  expect(!check_s32_dequant(s32_output(84, 84, 84, kDtypeS32, kQuantiScale, 84, 0,
                                       -1, -1, 4, -1, -1, /*axis=*/-1), &count),
         "a per-channel descriptor with an unreported axis must be rejected");
  expect(!check_s32_dequant(s32_output(84, 84, 84, kDtypeS32, kQuantiScale, 84, 0,
                                       -1, -1, 4, -1, -1, /*axis=*/2), &count),
         "a per-channel descriptor on a non-channel axis must be rejected");
  expect(static_cast<bool>(check_s32_dequant(s32_output(84, 84, 84, kDtypeS32,
                                                       kQuantiScale, 1, 0, -1, -1, 4, -1,
                                                       -1, /*axis=*/-1), &count)),
         "a scalar descriptor is axis-free and must be accepted");

  expect(static_cast<bool>(check_s32_dequant(
             s32_output(84, 84, 84, kDtypeF32, kQuantiNone), &count)),
         "unquantized F32 output should be accepted through the raw path");
  expect(!check_s32_dequant(s32_output(84, 84, 84, kDtypeS32, kQuantiNone), &count),
         "unquantized S32 output must be rejected");
}

void check_target_identity() {
  using namespace yolov5;
  expect(static_cast<bool>(check_target_matches_build("s600", "s600")),
         "matching build target should be accepted");
  expect(!check_target_matches_build("s100", "s600"),
         "requested target different from the build target must be rejected");
  expect(!check_target_matches_build("x5", "unknown"),
         "a binary without a build identity must be rejected");
}

// Drives the private S dequantizer with a tiny 1x2x2 layout whose every byte
// position is known, including pixel padding and a scalar (broadcasting)
// descriptor, so the addressing and arithmetic are verified against hand
// computed values rather than by trusting the formula's shape.
void check_s_dequant_values() {
  using namespace yolov5;
  // Layout: N=1, H=1, W=2, C=2, stride[3]=4, stride[2]=16 (8 bytes of pixel
  // padding), stride[1]=32. Storage holds two padded pixels of 16 bytes.
  TensorMeta meta = s32_output(1, 2, 2, kDtypeS32, kQuantiScale,
                               /*scale_len=*/1, /*zero_point_len=*/1,
                               /*row_stride=*/32, /*storage=*/-1,
                               /*channel_stride=*/4, /*pixel_stride=*/16);
  long long count = 0;
  expect(static_cast<bool>(check_s32_dequant(meta, &count)) && count == 4,
         "the tiny padded fixture must pass its gate");
  const std::int32_t q[4] = {10, 20, 30, 40};   // pixel0 c0/c1, pixel1 c0/c1
  const unsigned char storage[32] = {};
  std::memcpy(const_cast<unsigned char*>(storage) + 0, &q[0], 4);
  std::memcpy(const_cast<unsigned char*>(storage) + 4, &q[1], 4);
  std::memcpy(const_cast<unsigned char*>(storage) + 16, &q[2], 4);
  std::memcpy(const_cast<unsigned char*>(storage) + 20, &q[3], 4);
  // Scalar descriptors: scale 0.5 everywhere, zero point 2 everywhere. A
  // per-channel reading would index past the single-element arrays.
  const float scale[1] = {0.5F};
  const std::int32_t zero[1] = {2};
  const auto values = dequant_s32_nhwc(storage, meta, scale, 1, zero, 1);
  expect(values.size() == 4, "broadcasting dequant must return the full element count");
  if (values.size() == 4) {
    expect(std::fabs(values[0] - 4.0F) < 1e-6F && std::fabs(values[1] - 9.0F) < 1e-6F &&
               std::fabs(values[2] - 14.0F) < 1e-6F && std::fabs(values[3] - 19.0F) < 1e-6F,
           "broadcasting dequant must read the padded pixels at the right offsets");
  }
  // Per-channel descriptors must still index channel-wise.
  TensorMeta per_channel = s32_output(1, 2, 2, kDtypeS32, kQuantiScale,
                                      /*scale_len=*/2, /*zero_point_len=*/2,
                                      /*row_stride=*/32, /*storage=*/-1,
                                      /*channel_stride=*/4, /*pixel_stride=*/16);
  const float scales[2] = {1.0F, 10.0F};
  const std::int32_t zeros[2] = {0, 1};
  const auto channelwise = dequant_s32_nhwc(storage, per_channel, scales, 2, zeros, 2);
  expect(channelwise.size() == 4 &&
             std::fabs(channelwise[0] - 10.0F) < 1e-6F &&
             std::fabs(channelwise[1] - 190.0F) < 1e-6F &&
             std::fabs(channelwise[2] - 30.0F) < 1e-6F &&
             std::fabs(channelwise[3] - 390.0F) < 1e-6F,
         "per-channel descriptors must multiply channel by channel");
  // A scale descriptor that is neither scalar nor channel-covering is refused
  // up front instead of being read out of bounds (3 channels, 2 scales).
  bool refused = false;
  try {
    dequant_s32_nhwc(storage, s32_output(1, 2, 3, kDtypeS32, kQuantiScale,
                                         /*scale_len=*/2, /*zero_point_len=*/0,
                                         /*row_stride=*/48, /*storage=*/-1,
                                         /*channel_stride=*/4, /*pixel_stride=*/16),
                     scales, 2, nullptr, 0);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  expect(refused, "a short non-scalar descriptor must be refused, never blind-read");
  // A declared zero point without a buffer must be refused before any read.
  bool null_zero_refused = false;
  try {
    dequant_s32_nhwc(storage, s32_output(1, 2, 2, kDtypeS32, kQuantiScale,
                                         /*scale_len=*/1, /*zero_point_len=*/1,
                                         /*row_stride=*/32, /*storage=*/-1,
                                         /*channel_stride=*/4, /*pixel_stride=*/16),
                     scale, 1, nullptr, 1);
  } catch (const std::invalid_argument&) {
    null_zero_refused = true;
  }
  expect(null_zero_refused,
         "zero_point_len > 0 with a null buffer must be refused, never dereferenced");
  // NONE outputs pass through as float32 at the same offsets.
  TensorMeta raw = s32_output(1, 2, 2, kDtypeF32, kQuantiNone,
                              /*scale_len=*/0, /*zero_point_len=*/0,
                              /*row_stride=*/32, /*storage=*/-1,
                              /*channel_stride=*/4, /*pixel_stride=*/16);
  const float raw_values[4] = {1.5F, -2.5F, 3.5F, -4.5F};
  std::memcpy(const_cast<unsigned char*>(storage) + 0, &raw_values[0], 4);
  std::memcpy(const_cast<unsigned char*>(storage) + 4, &raw_values[1], 4);
  std::memcpy(const_cast<unsigned char*>(storage) + 16, &raw_values[2], 4);
  std::memcpy(const_cast<unsigned char*>(storage) + 20, &raw_values[3], 4);
  const auto floats = dequant_s32_nhwc(storage, raw, nullptr, 0, nullptr, 0);
  expect(floats.size() == 4 && std::fabs(floats[0] - 1.5F) < 1e-6F &&
             std::fabs(floats[3] + 4.5F) < 1e-6F,
         "unquantized F32 output must pass through at the same offsets");
}

// The scheduler backend is a bitmask; the CLI core index must be mapped, not
// assigned (0 would select no backend and 1 would select core 0).
void check_core_mapping() {
  using namespace yolov5;
  unsigned long long backend = 0;
  expect(bpu_core_to_backend(-1, &backend) && backend == (1ULL << 7),
         "bpu core -1 must map to HB_UCP_BPU_CORE_ANY (1ULL<<7)");
  expect(bpu_core_to_backend(0, &backend) && backend == (1ULL << 0),
         "bpu core 0 must map to HB_UCP_BPU_CORE_0, not to a raw 0 backend");
  expect(bpu_core_to_backend(1, &backend) && backend == (1ULL << 1),
         "bpu core 1 must map to HB_UCP_BPU_CORE_1, not to core 0");
  expect(bpu_core_to_backend(3, &backend) && backend == (1ULL << 3),
         "bpu core 3 must map to HB_UCP_BPU_CORE_3");
  expect(!bpu_core_to_backend(4, &backend), "bpu core index 4 must be rejected");
  expect(!bpu_core_to_backend(-2, &backend), "bpu core index -2 must be rejected");
  expect(!bpu_core_to_backend(99, &backend), "bpu core index 99 must be rejected");
  expect(!bpu_core_to_backend(0, nullptr), "a missing backend output must be rejected");
}

// One anchor whose objectness and class logits are both 0 gives an exact
// confidence of 0.5 * 0.5 = 0.25, which is the boundary under test.
std::vector<std::vector<float>> boundary_heads(long long classes) {
  const long long channels = 3 * (5 + classes);
  std::vector<std::vector<float>> heads;
  heads.emplace_back(static_cast<std::size_t>(8 * 8 * channels), -10.0F);
  heads.emplace_back(static_cast<std::size_t>(4 * 4 * channels), -10.0F);
  heads.emplace_back(static_cast<std::size_t>(2 * 2 * channels), -10.0F);
  for (long long i = 0; i < 4; ++i) heads[0][static_cast<std::size_t>(i)] = 0.0F;
  heads[0][4] = 0.0F;
  heads[0][5] = 0.0F;
  return heads;
}

void check_decode_boundary() {
  using namespace yolov5;
  const std::array<float, 18> anchors{};
  const std::vector<HeadShape> heads{{8, 8, 18}, {4, 4, 18}, {2, 2, 18}};
  auto raw = boundary_heads(1);
  DecodePolicy inclusive;
  inclusive.score_threshold = 0.25F;
  inclusive.nms_threshold = 0.45F;
  auto kept = decode_heads(raw, heads, 64, 1, inclusive, anchors);
  expect(kept.size() == 1, "S boundary must keep a score equal to the threshold");
  expect(!kept.empty() && std::fabs(kept[0].score - 0.25F) < 1.0e-6F,
         "boundary detection score should be 0.25");
  DecodePolicy strict = inclusive;
  strict.strict_score_boundary = true;
  expect(decode_heads(raw, heads, 64, 1, strict, anchors).empty(),
         "X5 strict boundary must drop a score equal to the threshold");
}

void check_decode_topk() {
  using namespace yolov5;
  const std::array<float, 18> anchors{};
  const long long classes = 1;
  const long long channels = 3 * (5 + classes);
  const std::vector<HeadShape> heads{{80, 80, channels}, {40, 40, channels}, {20, 20, channels}};
  std::vector<std::vector<float>> raw;
  long long total = 0;
  for (const auto& head : heads) {
    std::vector<float> level(static_cast<std::size_t>(head.height * head.width * channels), -10.0F);
    for (long long i = 0; i < head.height * head.width; ++i) {
      level[static_cast<std::size_t>(i * channels + 4)] = 5.0F;
      level[static_cast<std::size_t>(i * channels + 5)] = 5.0F;
      ++total;
    }
    raw.push_back(std::move(level));
  }
  DecodePolicy unlimited;
  unlimited.score_threshold = 0.01F;
  unlimited.nms_threshold = 1.0F;  // disable suppression so only top_k is measured
  auto all = decode_heads(raw, heads, 640, classes, unlimited, anchors);
  expect(static_cast<long long>(all.size()) == total,
         "unlimited policy must keep every surviving candidate");
  DecodePolicy capped = unlimited;
  capped.top_k_per_class = 300;
  auto top = decode_heads(raw, heads, 640, classes, capped, anchors);
  expect(top.size() == 300, "top_k policy must cap each class at 300 boxes");
}

void check_dump(const std::string& dir) {
  using namespace yolov5;
  expect(sha256_hex("abc", 3) ==
             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
         "sha256('abc') must match the published vector");
  expect(sha256_hex("", 0) ==
             "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
         "sha256('') must match the published vector");

  DumpRecord record;
  record.dir = dir;
  record.utc = utc_timestamp();
  record.target = "x5";
  record.build_target = "x5";
  record.asset_id = "x5:yolov5:s-v2.0";
  record.model_path = dir + "/nonexistent.bin";
  record.image_path = dir + "/nonexistent.jpg";
  record.argv = {"yolov5_cpp", "--target", "x5"};
  record.return_code = 0;
  DumpTensorInfo reported;
  reported.name = "output0";
  reported.dtype = "float32";
  reported.shape = {1, 2, 2};
  reported.quanti = "none";
  reported.aligned_byte_size = 16;
  reported.stride[0] = 16;
  reported.stride[1] = 8;
  reported.stride[2] = 4;
  reported.stride[3] = 4;
  reported.aligned[0] = 1;
  reported.aligned[1] = 2;
  reported.aligned[2] = 2;
  reported.aligned[3] = 4;
  reported.quantize_axis = 3;
  record.outputs.push_back(reported);
  DumpTensorInfo unreported;
  unreported.name = "input0";
  unreported.dtype = "uint8";
  unreported.shape = {1, 3, 2, 2};
  unreported.quanti = "none";
  // aligned_byte_size/stride/aligned stay unreported and must serialize as null.
  record.inputs.push_back(unreported);
  const float values[4] = {1.0F, 2.0F, 3.0F, 4.0F};
  DumpTensor tensor;
  tensor.name = "output0";
  tensor.dtype = "float32";
  tensor.shape = {1, 2, 2};
  tensor.bytes.assign(reinterpret_cast<const unsigned char*>(values),
                      reinterpret_cast<const unsigned char*>(values) + sizeof(values));
  DumpTensor input_payload;
  input_payload.name = "input0";
  input_payload.dtype = "uint8";
  input_payload.shape = {1, 3, 2, 2};
  input_payload.bytes = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2};
  record.input_tensors.push_back(input_payload);
  // The raw and transformed payloads of one output are deliberately different
  // byte sequences: a dump that lets one overwrite the other loses evidence.
  DumpTensor raw_tensor = tensor;
  raw_tensor.dtype = "int32";
  const std::int32_t raw_values[4] = {11, 22, 33, 44};
  raw_tensor.bytes.assign(reinterpret_cast<const unsigned char*>(raw_values),
                          reinterpret_cast<const unsigned char*>(raw_values) +
                              sizeof(raw_values));
  record.raw_tensors.push_back(raw_tensor);
  record.transformed_tensors.push_back(tensor);
  record.detections.push_back({1.0F, 2.0F, 3.0F, 4.0F, 0.75F, 7});
  std::string error;
  expect(write_dump(record, &error), "write_dump should succeed: " + error);

  const auto read_file = [](const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    return std::vector<unsigned char>((std::istreambuf_iterator<char>(input)),
                                      std::istreambuf_iterator<char>());
  };
  const std::vector<unsigned char> raw_read = read_file(dir + "/raw/0-output0.bin");
  const std::vector<unsigned char> transformed_read =
      read_file(dir + "/transformed/0-output0.bin");
  const std::vector<unsigned char> input_read = read_file(dir + "/input/0-input0.bin");
  expect(raw_read == raw_tensor.bytes,
         "the raw payload file must still hold the original int32 bytes");
  expect(transformed_read == tensor.bytes,
         "the transformed payload file must hold the float bytes, not the raw ones");
  expect(input_read == input_payload.bytes,
         "the input payload file must hold the submitted input bytes");
  expect(sha256_hex(raw_read.data(), raw_read.size()) ==
             sha256_hex(raw_tensor.bytes.data(), raw_tensor.bytes.size()),
         "the raw file digest must match its own content");
}

int run(const std::string& name, const std::string& scratch) {
  if (name == "x5_input") check_x5_input();
  else if (name == "x5_head") check_x5_head();
  else if (name == "s_plane") check_s_plane();
  else if (name == "s_dequant") check_s_dequant();
  else if (name == "s_dequant_values") check_s_dequant_values();
  else if (name == "core_mapping") check_core_mapping();
  else if (name == "target_identity") check_target_identity();
  else if (name == "decode_boundary") check_decode_boundary();
  else if (name == "decode_topk") check_decode_topk();
  else if (name == "dump") check_dump(scratch);
  else {
    std::cout << "unknown check: " << name << "\n";
    return 3;
  }
  return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "usage: portable_checks <check-name> [scratch-dir]\n";
    return 2;
  }
  return run(argv[1], argc > 2 ? argv[2] : std::string());
}
