// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Faster R-CNN detect model, one translation unit. The SDK-free anchor
// table, geometry/NMS ops, RPN selection, feat0 RoIAlign, PartB layouts and
// decoding tail are compiled unconditionally so their numeric behaviour stays
// host-testable against the Python fixtures; the X5 HB-DNN and S UCP sessions
// for both graphs live in the same file behind compile-time target guards
// (FRCNN_TARGET_X5 / FRCNN_TARGET_S), so there are no per-board task files.
// A build without a target macro still compiles and links, but the model
// refuses to construct because it has no board backend. Stage arithmetic
// mirrors runtime/python/detect.py expression for expression (float32 where
// numpy computes float32, double accumulations where it casts to float64),
// and -ffp-contract=off (set in CMakeLists.txt) keeps GCC from fusing
// multiply-adds numpy never fuses.

#include "detect.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <utility>

#if defined(FRCNN_TARGET_X5)
#include <dnn/hb_dnn.h>
#include <dnn/hb_dnn_ext.h>
#include <opencv2/opencv.hpp>
#define FRCNN_BOARD 1
#elif defined(FRCNN_TARGET_S)
#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include <opencv2/opencv.hpp>
#define FRCNN_BOARD 1
#endif

#ifndef FRCNN_TARGET_NAME
#define FRCNN_TARGET_NAME "host"
#endif

#ifndef FRCNN_SAMPLE_DIR
#define FRCNN_SAMPLE_DIR "."
#endif

namespace fasterrcnn {
namespace {

// Half-to-even rounding (np.round / np.rint semantics).
double round_half_even(double v) { return std::nearbyint(v); }

constexpr double kBboxXformClip = 4.1351665567423563;  // log(1000.0 / 16.0)

constexpr std::size_t kPooledCount = std::size_t(kNumProposals) * 256 * 7 * 7;  // 1,881,600
constexpr std::size_t kFeat0Count = 256 * 10 * 10;

// ------------------------------------------------------------------ sha256 ----

struct Sha256 {
  std::uint32_t h[8];
  std::uint64_t total;
  std::uint8_t block[64];
  std::size_t used;

  void init() {
    static const std::uint32_t kInit[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372,
                                           0xa54ff53a, 0x510e527f, 0x9b05688c,
                                           0x1f83d9ab, 0x5be0cd19};
    std::memcpy(h, kInit, sizeof h);
    total = 0;
    used = 0;
  }

  static std::uint32_t rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

  void compress(const std::uint8_t* p) {
    static const std::uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = (std::uint32_t(p[4 * i]) << 24) | (std::uint32_t(p[4 * i + 1]) << 16) |
             (std::uint32_t(p[4 * i + 2]) << 8) | std::uint32_t(p[4 * i + 3]);
    for (int i = 16; i < 64; ++i) {
      const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    std::uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const std::uint32_t ch = (e & f) ^ (~e & g);
      const std::uint32_t t1 = hh + s1 + ch + k[i] + w[i];
      const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t t2 = s0 + maj;
      hh = g; g = f; f = e; e = d + t1;
      d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }

  void update(const std::uint8_t* p, std::size_t n) {
    total += n;
    while (n > 0) {
      const std::size_t take = std::min(n, std::size_t(64) - used);
      std::memcpy(block + used, p, take);
      used += take; p += take; n -= take;
      if (used == 64) { compress(block); used = 0; }
    }
  }

  void final_hex(char out[65]) {
    const std::uint64_t bits = total * 8;
    const std::uint8_t pad = 0x80;
    update(&pad, 1);
    const std::uint8_t zero = 0;
    while (used != 56) update(&zero, 1);
    std::uint8_t len[8];
    for (int i = 0; i < 8; ++i) len[i] = std::uint8_t(bits >> (56 - 8 * i));
    update(len, 8);  // exactly 56 + 8: compresses once, leaves used == 0
    static const char* hex = "0123456789abcdef";
    std::size_t at = 0;
    for (int i = 0; i < 8; ++i)
      for (int shift = 28; shift >= 0; shift -= 4) out[at++] = hex[(h[i] >> shift) & 0xF];
    out[at] = 0;
  }
};

bool all_finite(const float* v, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    if (!std::isfinite(v[i])) return false;
  return true;
}

}  // namespace

// -------------------------------------------------------------- public: sha ----

std::string file_sha256(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open " + path + " for hashing");
  Sha256 sha;
  sha.init();
  std::vector<std::uint8_t> buf(1 << 20);
  while (true) {
    const std::size_t got = std::fread(buf.data(), 1, buf.size(), f);
    if (got > 0) sha.update(buf.data(), got);
    if (got < buf.size()) {
      if (std::ferror(f)) {
        std::fclose(f);
        throw std::runtime_error("read error while hashing " + path);
      }
      break;
    }
  }
  std::fclose(f);
  char hex[65];
  sha.final_hex(hex);
  return std::string(hex);
}

// ---------------------------------------------------------- public: anchors ----

const std::vector<float>& anchor_table() {
  // TorchVision AnchorGenerator facts for this model: identical anchor-size
  // tuple on every level, ratios (0.5, 1.0, 2.0), grids 10x10/10x10/5x5 with
  // strides 32/32/64 (levels concatenate in feature order, ratio index outer
  // and size index inner).
  static const std::vector<float> table = [] {
    const double sizes[5] = {32, 64, 128, 256, 512};
    const double ratios[3] = {0.5, 1.0, 2.0};
    const int grids[3][2] = {{10, 10}, {10, 10}, {5, 5}};
    const int strides[3] = {32, 32, 64};
    std::vector<float> rows;
    rows.reserve(std::size_t(kAnchorCount) * 4);
    for (int level = 0; level < 3; ++level) {
      for (int gy = 0; gy < grids[level][0]; ++gy) {
        for (int gx = 0; gx < grids[level][1]; ++gx) {
          const double shift_x = double(gx * strides[level]);
          const double shift_y = double(gy * strides[level]);
          for (int r = 0; r < 3; ++r) {
            const double ratio = std::sqrt(ratios[r]);
            for (int s = 0; s < 5; ++s) {
              const double w = sizes[s] / ratio;
              const double h = sizes[s] * ratio;
              // TorchVision rounds the zero-centered cell anchors.
              rows.push_back(float(shift_x + round_half_even(-w / 2.0)));
              rows.push_back(float(shift_y + round_half_even(-h / 2.0)));
              rows.push_back(float(shift_x + round_half_even(w / 2.0)));
              rows.push_back(float(shift_y + round_half_even(h / 2.0)));
            }
          }
        }
      }
    }
    return rows;
  }();
  return table;
}

// -------------------------------------------------------- public: geometry ----

LetterboxContext letterbox_geometry(int rows, int cols, int canvas) {
  if (rows < 1 || cols < 1)
    throw std::invalid_argument("image size must be positive");
  const double scale = std::min(double(canvas) / cols, double(canvas) / rows);
  // np.round (half-to-even) on the resize targets.
  const int net_rows = int(round_half_even(rows * scale));
  const int net_cols = int(round_half_even(cols * scale));
  if (net_rows < 1 || net_cols < 1)
    throw std::invalid_argument("image aspect ratio produces an empty letterbox dimension");
  LetterboxContext ctx;
  ctx.original_rows = rows;
  ctx.original_cols = cols;
  ctx.net_rows = net_rows;
  ctx.net_cols = net_cols;
  ctx.canvas = canvas;
  ctx.scale = scale;
  return ctx;
}

void LetterboxContext::to_original(float box[4]) const {
  const float sx = float(scale_x());
  const float sy = float(scale_y());
  box[0] = box[0] / sx;
  box[2] = box[2] / sx;
  box[1] = box[1] / sy;
  box[3] = box[3] / sy;
  const float w = float(original_cols);
  const float h = float(original_rows);
  box[0] = std::min(std::max(box[0], 0.0F), w);
  box[2] = std::min(std::max(box[2], 0.0F), w);
  box[1] = std::min(std::max(box[1], 0.0F), h);
  box[3] = std::min(std::max(box[3], 0.0F), h);
}

void decode_deltas(const float* anchors, const float* codes, int rows, int code_cols,
                   const float weights[4], float* out) {
  for (int i = 0; i < rows; ++i) {
    const float* a = anchors + std::size_t(i) * 4;
    const float width = a[2] - a[0];
    const float height = a[3] - a[1];
    const float ctr_x = a[0] + 0.5F * width;
    const float ctr_y = a[1] + 0.5F * height;
    for (int j = 0; j < code_cols / 4; ++j) {
      const float* c = codes + std::size_t(i) * code_cols + std::size_t(j) * 4;
      const float dx = c[0] / weights[0];
      const float dy = c[1] / weights[1];
      const float dw = std::min(c[2] / weights[2], float(kBboxXformClip));
      const float dh = std::min(c[3] / weights[3], float(kBboxXformClip));
      const float pred_ctr_x = dx * width + ctr_x;
      const float pred_ctr_y = dy * height + ctr_y;
      const float pred_w = std::exp(dw) * width;
      const float pred_h = std::exp(dh) * height;
      float* o = out + std::size_t(i) * code_cols + std::size_t(j) * 4;
      o[0] = pred_ctr_x - 0.5F * pred_w;
      o[1] = pred_ctr_y - 0.5F * pred_h;
      o[2] = pred_ctr_x + 0.5F * pred_w;
      o[3] = pred_ctr_y + 0.5F * pred_h;
    }
  }
}

void clip_boxes_to_image(float* boxes, int n, float height, float width) {
  for (int i = 0; i < n; ++i) {
    float* b = boxes + std::size_t(i) * 4;
    b[0] = std::min(std::max(b[0], 0.0F), width);
    b[2] = std::min(std::max(b[2], 0.0F), width);
    b[1] = std::min(std::max(b[1], 0.0F), height);
    b[3] = std::min(std::max(b[3], 0.0F), height);
  }
}

namespace {

// Greedy NMS over a descending-score index order; shared by nms() (identity
// offsets) and batched_nms() (which pre-shifts boxes per group instead).
// ``areas``/``finite`` are indexed by box id so every lookup is O(1).
// Areas, intersection, union and the IoU division all run in float32 with a
// strict ``>`` threshold — the TorchVision CPU kernel's arithmetic, which
// the Python port reproduces (NumPy weak scalars demote the threshold to
// float32 too). Double arithmetic here is NOT equivalent: boxes [0,0,17,1]
// vs [3,0,20,1] at threshold 0.7f gives float32 IoU 14/20 =
// 0.69999998807907104 (kept) but double IoU 0.69999999999999996 (dropped).
std::vector<int> nms_ordered(const float* boxes, int n, const std::vector<int>& order,
                             float iou_threshold) {
  std::vector<float> areas(std::size_t(n), 0.0F);
  std::vector<char> finite(std::size_t(n), 1);
  for (int i = 0; i < n; ++i) {
    const float* b = boxes + std::size_t(i) * 4;
    if (!std::isfinite(b[0]) || !std::isfinite(b[1]) || !std::isfinite(b[2]) ||
        !std::isfinite(b[3]))
      finite[std::size_t(i)] = 0;
    areas[std::size_t(i)] = (b[2] - b[0]) * (b[3] - b[1]);
  }
  std::vector<int> keep;
  for (int i : order) {
    if (!finite[std::size_t(i)] || areas[std::size_t(i)] <= 0.0F) continue;
    const float* bi = boxes + std::size_t(i) * 4;
    bool suppressed = false;
    for (int j : keep) {
      const float* bj = boxes + std::size_t(j) * 4;
      const float xx1 = std::max(bi[0], bj[0]);
      const float yy1 = std::max(bi[1], bj[1]);
      const float xx2 = std::min(bi[2], bj[2]);
      const float yy2 = std::min(bi[3], bj[3]);
      const float iw = std::max(0.0F, xx2 - xx1);
      const float ih = std::max(0.0F, yy2 - yy1);
      const float inter = iw * ih;
      const float uni = areas[std::size_t(i)] + areas[std::size_t(j)] - inter;
      if (uni > 0.0F && inter / uni > iou_threshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) keep.push_back(i);
  }
  return keep;
}

}  // namespace

std::vector<int> nms(const float* boxes, const float* scores, int n, float iou_threshold) {
  std::vector<int> order(std::size_t(n), 0);
  for (int i = 0; i < n; ++i) order[std::size_t(i)] = i;
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return scores[a] > scores[b];
  });
  return nms_ordered(boxes, n, order, iou_threshold);
}

std::vector<int> batched_nms(const float* boxes, const float* scores,
                             const std::int64_t* idxs, int n, float iou_threshold) {
  if (n == 0) return {};
  float max_coordinate = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < 4; ++k)
      max_coordinate = std::max(max_coordinate, boxes[std::size_t(i) * 4 + k]);
  // TorchVision composition: shift each group far away by (max + 1) * idx.
  // The offset product is float64, cast to float32 before the addition —
  // exactly what the Python port does.
  std::vector<float> shifted(std::size_t(n) * 4);
  for (int i = 0; i < n; ++i) {
    const double offset = double(idxs[i]) * (double(max_coordinate) + 1.0);
    const float off = float(offset);
    for (int k = 0; k < 4; ++k) shifted[std::size_t(i) * 4 + k] = boxes[std::size_t(i) * 4 + k] + off;
  }
  return nms(shifted.data(), scores, n, iou_threshold);
}

// --------------------------------------------------------- public: RPN stage ----

MiddleResult rpn_select(const float* rpn_obj, const float* rpn_deltas,
                        const float* anchors) {
  if (!all_finite(rpn_obj, kAnchorCount) || !all_finite(rpn_deltas, std::size_t(kAnchorCount) * 4))
    throw std::invalid_argument("PartA RPN outputs contain non-finite values.");
  const int level_counts[3] = {1500, 1500, 375};
  const float kRpnWeights[4] = {1.0F, 1.0F, 1.0F, 1.0F};

  std::vector<float> decoded(std::size_t(kAnchorCount) * 4);
  decode_deltas(anchors, rpn_deltas, kAnchorCount, 4, kRpnWeights, decoded.data());

  std::vector<float> boxes;
  std::vector<float> scores;
  std::vector<std::int64_t> levels;
  boxes.reserve(450 * 4);
  scores.reserve(450);
  levels.reserve(450);
  int offset = 0;
  for (int level = 0; level < 3; ++level) {
    const int count = level_counts[level];
    const int k = std::min(kRpnPreNmsTopN, count);
    // Top-k on the raw logits (monotone with the sigmoid), stable for ties.
    std::vector<int> idx(std::size_t(count), 0);
    for (int i = 0; i < count; ++i) idx[std::size_t(i)] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
      return rpn_obj[offset + a] > rpn_obj[offset + b];
    });
    idx.resize(std::size_t(k));
    for (int i : idx) {
      const float* d = decoded.data() + std::size_t(offset + i) * 4;
      boxes.insert(boxes.end(), d, d + 4);
      const float logit = rpn_obj[offset + i];
      // Numerically stable sigmoid, like the Python port.
      scores.push_back(logit >= 0.0F ? 1.0F / (1.0F + std::exp(-logit))
                                     : std::exp(logit) / (1.0F + std::exp(logit)));
      levels.push_back(level);
    }
    offset += count;
  }
  const int m = int(scores.size());
  clip_boxes_to_image(boxes.data(), m, float(kInputSize), float(kInputSize));

  // Small-box removal (>= min_size on both extents), then the score gate.
  std::vector<int> keep;
  for (int i = 0; i < m; ++i) {
    const float* b = boxes.data() + std::size_t(i) * 4;
    if (b[2] - b[0] >= kRpnMinSize && b[3] - b[1] >= kRpnMinSize && scores[std::size_t(i)] >= kRpnScoreThresh)
      keep.push_back(i);
  }
  std::vector<float> fboxes;
  std::vector<float> fscores;
  std::vector<std::int64_t> flevels;
  fboxes.reserve(keep.size() * 4);
  for (int i : keep) {
    const float* b = boxes.data() + std::size_t(i) * 4;
    fboxes.insert(fboxes.end(), b, b + 4);
    fscores.push_back(scores[std::size_t(i)]);
    flevels.push_back(levels[std::size_t(i)]);
  }
  std::vector<int> kept = batched_nms(fboxes.data(), fscores.data(), flevels.data(),
                                      int(fscores.size()), kRpnNmsThresh);
  if (int(kept.size()) > kRpnPostNmsTopN) kept.resize(std::size_t(kRpnPostNmsTopN));

  MiddleResult result;
  result.proposals.assign(std::size_t(kNumProposals) * 4, 0.0F);
  result.n_valid = int(kept.size());
  for (int i = 0; i < result.n_valid; ++i) {
    const float* b = fboxes.data() + std::size_t(kept[std::size_t(i)]) * 4;
    std::memcpy(result.proposals.data() + std::size_t(i) * 4, b, 4 * sizeof(float));
  }
  return result;
}

// ----------------------------------------------------- public: RoIAlign stage ----

void roi_align_feat0(const float* feat, int channels, int height, int width,
                     const float* rois, int roi_count, float* out,
                     float spatial_scale, int output_size, int sampling_ratio) {
  const std::size_t plane = std::size_t(channels) * output_size * output_size;
  std::fill(out, out + std::size_t(roi_count) * plane, 0.0F);
  if (roi_count == 0) return;
  // Python normalizes the float64 accumulator once per bin: (acc * inv) cast.
  const double inv = 1.0 / double(sampling_ratio * sampling_ratio);
  std::vector<double> acc(std::size_t(channels), 0.0);

  for (int r = 0; r < roi_count; ++r) {
    const float* roi = rois + std::size_t(r) * 4;
    const float x1 = roi[0] * spatial_scale;
    const float y1 = roi[1] * spatial_scale;
    const float x2 = roi[2] * spatial_scale;
    const float y2 = roi[3] * spatial_scale;
    // TorchVision forces malformed ROIs to 1x1 in feature pixels.
    const float roi_w = std::max(x2 - x1, 1.0F);
    const float roi_h = std::max(y2 - y1, 1.0F);
    const float bin_w = roi_w / float(output_size);
    const float bin_h = roi_h / float(output_size);
    float* dst_roi = out + std::size_t(r) * plane;
    for (int iy = 0; iy < output_size; ++iy) {
      for (int ix = 0; ix < output_size; ++ix) {
        const float yb = y1 + iy * bin_h;
        const float xb = x1 + ix * bin_w;
        std::fill(acc.begin(), acc.end(), 0.0);
        for (int fy = 0; fy < sampling_ratio; ++fy) {
          const float fyo = (float(fy) + 0.5F) / float(sampling_ratio);
          for (int fx = 0; fx < sampling_ratio; ++fx) {
            const float fxo = (float(fx) + 0.5F) / float(sampling_ratio);
            const float ys = yb + fyo * bin_h;
            const float xs = xb + fxo * bin_w;
            // Outside [-1, H]/[-1, W]: the sample contributes zero.
            const bool valid = ys >= -1.0F && ys <= float(height) && xs >= -1.0F &&
                               xs <= float(width);
            if (!valid) continue;
            const float y = ys < 0.0F ? 0.0F : ys;
            const float x = xs < 0.0F ? 0.0F : xs;
            const int y_low_unc = int(y);
            const int x_low_unc = int(x);
            const bool clamp_y = y_low_unc >= height - 1;
            const bool clamp_x = x_low_unc >= width - 1;
            const int y_low = clamp_y ? height - 1 : y_low_unc;
            const int x_low = clamp_x ? width - 1 : x_low_unc;
            const int y_high = clamp_y ? height - 1 : y_low_unc + 1;
            const int x_high = clamp_x ? width - 1 : x_low_unc + 1;
            const double y_used = clamp_y ? double(y_low) : double(y);
            const double x_used = clamp_x ? double(x_low) : double(x);
            const double ly = y_used - y_low;
            const double lx = x_used - x_low;
            const double hy = 1.0 - ly;
            const double hx = 1.0 - lx;
            const double w00 = hy * hx, w01 = hy * lx, w10 = ly * hx, w11 = ly * lx;
            for (int c = 0; c < channels; ++c) {
              const double v00 = double(feat[(std::size_t(c) * height + y_low) * width + x_low]);
              const double v01 = double(feat[(std::size_t(c) * height + y_low) * width + x_high]);
              const double v10 = double(feat[(std::size_t(c) * height + y_high) * width + x_low]);
              const double v11 = double(feat[(std::size_t(c) * height + y_high) * width + x_high]);
              // Same grouping as the Python row: the four weighted corners
              // sum first, then the sample joins the bin accumulator.
              acc[std::size_t(c)] += v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11;
            }
          }
        }
        // Store [r, c, iy, ix] (channel-major, matching the [150,256,7,7]
        // PartB input) after the single float64 normalization.
        for (int c = 0; c < channels; ++c)
          dst_roi[std::size_t(c) * output_size * output_size + std::size_t(iy) * output_size +
                  std::size_t(ix)] = float(acc[std::size_t(c)] * inv);
      }
    }
  }
}

// ------------------------------------------------------ public: PartB layouts ----

void pack_bank(const float* pooled, float* bank) {
  // [150,256,7,7] -> [1,256,1050,7]: bank[c][i*7+py][px] = pooled[i][c][py][px].
  for (int i = 0; i < kNumProposals; ++i)
    for (int c = 0; c < 256; ++c)
      for (int py = 0; py < 7; ++py)
        std::memcpy(bank + (((std::size_t(c) * kNumProposals + i) * 7 + py) * 7),
                    pooled + (((std::size_t(i) * 256 + c) * 7 + py) * 7), 7 * sizeof(float));
}

void unpack_bank_outputs(const float* cls_packed, const float* reg_packed, float* cls,
                         float* reg) {
  // [1,C,150,1] -> [150,C] rows.
  for (int c = 0; c < kNumClasses; ++c)
    for (int i = 0; i < kNumProposals; ++i)
      cls[std::size_t(i) * kNumClasses + c] =
          cls_packed[(std::size_t(c) * kNumProposals + i)];
  for (int c = 0; c < kNumClasses * 4; ++c)
    for (int i = 0; i < kNumProposals; ++i)
      reg[std::size_t(i) * kNumClasses * 4 + c] =
          reg_packed[(std::size_t(c) * kNumProposals + i)];
}

// ---------------------------------------------------------- public: tail stage ----

DetectionResult tail(const float* cls, const float* reg, const float* proposals,
                     int n_valid) {
  if (n_valid < 0 || n_valid > kNumProposals)
    throw std::invalid_argument("n_valid must be within [0,150]");
  DetectionResult result;
  if (n_valid == 0) return result;
  const std::size_t rows = std::size_t(n_valid);
  if (!all_finite(cls, rows * kNumClasses) || !all_finite(reg, rows * kNumClasses * 4))
    throw std::invalid_argument("PartB outputs contain non-finite values.");

  const float kTailWeights[4] = {10.0F, 10.0F, 5.0F, 5.0F};
  std::vector<float> boxes(rows * kNumClasses * 4);
  decode_deltas(proposals, reg, n_valid, kNumClasses * 4, kTailWeights, boxes.data());

  // Softmax over all 91 columns of every row (background included in the
  // denominator, exactly like the TorchVision head), background column
  // dropped after the division.
  std::vector<float> scores(rows * (kNumClasses - 1));
  for (std::size_t i = 0; i < rows; ++i) {
    const float* logits = cls + i * kNumClasses;
    float mx = logits[0];
    for (int c = 1; c < kNumClasses; ++c) mx = std::max(mx, logits[c]);
    float sum = 0.0F;
    for (int c = 0; c < kNumClasses; ++c) sum += std::exp(logits[c] - mx);
    for (int c = 1; c < kNumClasses; ++c)
      scores[i * (kNumClasses - 1) + (c - 1)] = std::exp(logits[c] - mx) / sum;
  }
  // Clip every per-class box (rows * 91) before the background column and
  // the score filter drop most of them.
  clip_boxes_to_image(boxes.data(), int(rows) * kNumClasses, float(kInputSize),
                      float(kInputSize));

  // Flatten roi-major/class-minor (label c for column c-1) and filter.
  const int cols = kNumClasses - 1;
  std::vector<int> keep;
  for (std::size_t i = 0; i < rows; ++i)
    for (int c = 0; c < cols; ++c) {
      const float* b = boxes.data() + (i * kNumClasses + c + 1) * 4;
      const float s = scores[i * cols + c];
      if (!(s > kBoxScoreThresh)) continue;
      if (b[2] - b[0] >= kBoxMinSize && b[3] - b[1] >= kBoxMinSize) keep.push_back(int(i) * cols + c);
    }
  std::vector<float> fboxes;
  std::vector<float> fscores;
  std::vector<std::int64_t> flabels;
  fboxes.reserve(keep.size() * 4);
  for (int k : keep) {
    const std::size_t i = std::size_t(k) / cols;
    const int c = int(std::size_t(k) % cols);
    const float* b = boxes.data() + (i * kNumClasses + c + 1) * 4;
    fboxes.insert(fboxes.end(), b, b + 4);
    fscores.push_back(scores[i * cols + c]);
    flabels.push_back(c + 1);
  }
  std::vector<int> kept = batched_nms(fboxes.data(), fscores.data(), flabels.data(),
                                      int(fscores.size()), kBoxNmsThresh);
  if (int(kept.size()) > kBoxDetectionsPerImg) kept.resize(std::size_t(kBoxDetectionsPerImg));
  for (int k : kept) {
    result.boxes.insert(result.boxes.end(), fboxes.begin() + std::size_t(k) * 4,
                        fboxes.begin() + (std::size_t(k) + 1) * 4);
    result.scores.push_back(fscores[std::size_t(k)]);
    result.labels.push_back(flabels[std::size_t(k)]);
  }
  return result;
}

// ------------------------------------------------------ public: descriptor table ----

const std::vector<DescriptorRow>& descriptor_rows() {
  // Exact transcription of model/frcnn_mbv3_320.json (targets table). The
  // JSON descriptor is the source of truth; the two must be updated together.
  static const std::vector<DescriptorRow> rows = {
      {"x5", "bayes-e", "frcnn_mbv3_320_parta.bin",
       "a6f6509e5ab3b42f709d10062cfdb0a930b9eebb3ceb1d1a4bff3baeda42af86",
       "frcnn_mbv3_320_partb.bin",
       "2ade35e91d66debf616594a28e2cae051b5e1ede617dfa914b25d06c3255724e",
       PartbLayout::kPooled, 1515.1, 38389.8},
      {"s100", "nash-e", "frcnn_mbv3_320_parta.hbm",
       "ca4c565497b3b027cad2241110e689c5eb54e24939a328f6e4fb8bd013e51337",
       "frcnn_mbv3_320_partb.hbm",
       "d4d21cebaece91c6195794b7062f607d665ef10c3055ce22d512ba14a5307ff4",
       PartbLayout::kPacked, 419.3, 2643.3},
      {"s100p", "nash-m", "frcnn_mbv3_320_parta.hbm",
       "e7f4611d5283ff71673f8b0d67953111e70d2878405b8054bf56536d12924710",
       "frcnn_mbv3_320_partb.hbm",
       "ed91c61f81f2ac21a2d9dd8d20170f6a85243fab4157247e0943c8af535809f7",
       PartbLayout::kPacked, 332.0, 1773.4},
      {"s600", "nash-p", "frcnn_mbv3_320_parta.hbm",
       "a0c857181e95b275190aca68c4149814c7381ceb7e3c9aac1cdd8a22cd2e0e84",
       "frcnn_mbv3_320_partb.hbm",
       "2ae5360e0b1bbc1d463cfd12f4764936e2c4e9dd5741cc35d0df705d848603ef",
       PartbLayout::kPacked, 275.3, 1648.2},
  };
  return rows;
}

const DescriptorRow* find_descriptor_row(const std::string& target) {
  for (const DescriptorRow& row : descriptor_rows())
    if (target == row.target) return &row;
  return nullptr;
}

std::string descriptor_model_dir(const std::string& target) {
  return std::string(FRCNN_SAMPLE_DIR) + "/model/" + target;
}

// ============================================================ board backends ====
#if defined(FRCNN_BOARD)

namespace {

void check(int rc, const char* what) {
  if (rc != 0) throw std::runtime_error(what);
}

constexpr std::size_t kNv12Bytes = std::size_t(kInputSize) * kInputSize * 3 / 2;

// Letterbox into the zero-padded 320x320 BGR canvas, shared by both stacks.
cv::Mat letterbox_canvas(const cv::Mat& source, const LetterboxContext& ctx) {
  cv::Mat resized;
  cv::resize(source, resized, cv::Size(ctx.net_cols, ctx.net_rows), 0, 0, cv::INTER_LINEAR);
  cv::Mat canvas(kInputSize, kInputSize, CV_8UC3, cv::Scalar(0, 0, 0));
  resized.copyTo(canvas(cv::Rect(0, 0, ctx.net_cols, ctx.net_rows)));
  return canvas;
}

// BGR canvas -> I420 -> Y plane + interleaved UV plane (the exact arithmetic
// of the shared utils/py_utils/image.py converter, one OpenCV cvtColor call
// plus plane copies).
void nv12_planes(const cv::Mat& canvas, std::vector<std::uint8_t>* y,
                 std::vector<std::uint8_t>* uv) {
  cv::Mat yuv;
  cv::cvtColor(canvas, yuv, cv::COLOR_BGR2YUV_I420);
  const std::size_t area = std::size_t(kInputSize) * kInputSize;
  const std::size_t quarter = area / 4;
  const std::uint8_t* data = yuv.ptr<std::uint8_t>();
  y->assign(data, data + area);
  uv->resize(2 * quarter);
  const std::uint8_t* u = data + area;
  const std::uint8_t* v = u + quarter;
  for (std::size_t i = 0; i < quarter; ++i) {
    (*uv)[2 * i] = u[i];
    (*uv)[2 * i + 1] = v[i];
  }
}

// Trim trailing singleton dimensions from one SDK-declared shape. The X5
// packed-model spelling pads graph-declared shapes to 4D with trailing 1s
// ((3375,1) spelled (3375,1,1,1)); a trailing size-1 dimension contributes
// exactly one element, so the flat row-major buffers are identical. At least
// one dimension always remains.
int trim_trailing_singletons(const int32_t* dims, int num_dimensions) {
  int n = num_dimensions;
  while (n > 1 && dims[n - 1] == 1) --n;
  return n;
}

// Does one declared shape equal ``expected`` after trimming trailing
// singleton dimensions from both sides?
bool shape_equals(const int32_t* dims, int num_dimensions, std::initializer_list<int> expected) {
  int want_size = int(expected.size());
  const int* want = expected.begin();
  while (want_size > 1 && want[want_size - 1] == 1) --want_size;
  const int got_size = trim_trailing_singletons(dims, num_dimensions);
  if (got_size != want_size) return false;
  for (int d = 0; d < got_size; ++d)
    if (dims[d] != want[d]) return false;
  return true;
}

}  // namespace

#if defined(FRCNN_TARGET_X5)

namespace {

// Owns one submitted task: released exactly once on every path. A failed
// explicit release clears the handle first, so the destructor can never
// double-release.
struct X5TaskGuard {
  hbDNNTaskHandle_t task = nullptr;
  X5TaskGuard() = default;
  X5TaskGuard(const X5TaskGuard&) = delete;
  X5TaskGuard& operator=(const X5TaskGuard&) = delete;
  ~X5TaskGuard() {
    if (task) hbDNNReleaseTask(task);
  }
  int release() {
    const hbDNNTaskHandle_t done = task;
    task = nullptr;
    return hbDNNReleaseTask(done);
  }
};

// One loaded artifact: packed handle + model handle + queried metadata.
struct X5Graph {
  hbPackedDNNHandle_t packed = nullptr;
  hbDNNHandle_t model = nullptr;
  std::vector<hbDNNTensorProperties> input_props;
  std::vector<hbDNNTensorProperties> output_props;
  std::vector<hbDNNTensor> inputs;
  std::vector<hbDNNTensor> outputs;
  std::vector<bool> input_allocated, output_allocated;

  ~X5Graph() {
    for (std::size_t i = 0; i < inputs.size(); ++i)
      if (i < input_allocated.size() && input_allocated[i]) hbSysFreeMem(&inputs[i].sysMem[0]);
    for (std::size_t i = 0; i < outputs.size(); ++i)
      if (i < output_allocated.size() && output_allocated[i]) hbSysFreeMem(&outputs[i].sysMem[0]);
    if (packed) hbDNNRelease(packed);
  }

  void load(const std::string& path) {
    const char* file = path.c_str();
    check(hbDNNInitializeFromFiles(&packed, &file, 1), "hbDNNInitializeFromFiles failed");
    const char** names = nullptr;
    int model_count = 0;
    check(hbDNNGetModelNameList(&names, &model_count, packed), "hbDNNGetModelNameList failed");
    if (model_count != 1)
      throw ContractError("each Faster R-CNN artifact must contain exactly one model: " + path);
    check(hbDNNGetModelHandle(&model, packed, names[0]), "hbDNNGetModelHandle failed");
    int32_t in = 0, out = 0;
    check(hbDNNGetInputCount(&in, model), "hbDNNGetInputCount failed");
    check(hbDNNGetOutputCount(&out, model), "hbDNNGetOutputCount failed");
    input_props.assign(std::size_t(in), hbDNNTensorProperties{});
    output_props.assign(std::size_t(out), hbDNNTensorProperties{});
    inputs.assign(std::size_t(in), hbDNNTensor{});
    outputs.assign(std::size_t(out), hbDNNTensor{});
    input_allocated.assign(std::size_t(in), false);
    output_allocated.assign(std::size_t(out), false);
    for (int32_t i = 0; i < in; ++i) {
      check(hbDNNGetInputTensorProperties(&input_props[std::size_t(i)], model, i),
            "hbDNNGetInputTensorProperties failed");
      inputs[std::size_t(i)].properties = input_props[std::size_t(i)];
    }
    for (int32_t i = 0; i < out; ++i) {
      check(hbDNNGetOutputTensorProperties(&output_props[std::size_t(i)], model, i),
            "hbDNNGetOutputTensorProperties failed");
      outputs[std::size_t(i)].properties = output_props[std::size_t(i)];
    }
  }

  void alloc_input(std::size_t i, int bytes, const char* what) {
    if (input_allocated[i]) return;
    check(hbSysAllocCachedMem(&inputs[i].sysMem[0], bytes), what);
    input_allocated[i] = true;
  }

  void alloc_outputs() {
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      if (output_allocated[i]) continue;
      check(hbSysAllocCachedMem(&outputs[i].sysMem[0],
                                int(output_props[i].alignedByteSize)),
            "hbSysAllocCachedMem output failed");
      output_allocated[i] = true;
    }
  }

  void run() {
    X5TaskGuard guard;
    hbDNNInferCtrlParam ctrl;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&ctrl);
    hbDNNTensor* out_ptr = outputs.data();
    check(hbDNNInfer(&guard.task, &out_ptr, inputs.data(), model, &ctrl), "hbDNNInfer failed");
    check(hbDNNWaitTaskDone(guard.task, 0), "hbDNNWaitTaskDone failed");
    check(guard.release(), "hbDNNReleaseTask failed");
    for (auto& t : outputs) check(hbSysFlushMem(&t.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE),
                                  "hbSysFlushMem output failed");
  }

  // Read one FLOAT32 output with an exact declared shape into compact order.
  // The X5 SDK reports the physical layout through alignedShape; a padded
  // dimension is walked per element, an unpadded one memcpy'd whole.
  std::vector<float> read_float(std::size_t i, std::initializer_list<int> expected) {
    const hbDNNTensorProperties& p = output_props[i];
    if (p.tensorType != HB_DNN_TENSOR_TYPE_F32)
      throw ContractError("output " + std::to_string(i) + " is not float32");
    const auto& valid = p.validShape;
    if (!shape_equals(valid.dimensionSize, valid.numDimensions, expected))
      throw ContractError("output " + std::to_string(i) + " has a shape this sample does "
                          "not publish");
    std::size_t count = 1;
    for (int d = 0; d < valid.numDimensions; ++d) count *= std::size_t(valid.dimensionSize[d]);
    // Physical strides in floats from the aligned extents.
    const auto& aligned = p.alignedShape;
    std::vector<std::size_t> astride(std::size_t(std::max(aligned.numDimensions, 1)), 1);
    for (int d = aligned.numDimensions - 2; d >= 0; --d)
      astride[std::size_t(d)] =
          astride[std::size_t(d + 1)] * std::size_t(std::max(aligned.dimensionSize[d + 1], 1));
    const float* base = static_cast<const float*>(outputs[i].sysMem[0].virAddr);
    std::vector<float> out(count);
    bool compact = true;
    for (int d = 0; d < valid.numDimensions; ++d)
      if (aligned.dimensionSize[d] != valid.dimensionSize[d]) compact = false;
    if (compact) {
      std::memcpy(out.data(), base, count * sizeof(float));
      return out;
    }
    std::vector<int> idx(std::size_t(std::max(valid.numDimensions, 1)), 0);
    for (std::size_t e = 0; e < count; ++e) {
      std::size_t offset = 0;
      for (int d = 0; d < valid.numDimensions; ++d)
        offset += std::size_t(idx[std::size_t(d)]) * astride[std::size_t(d)];
      if ((offset + 1) * sizeof(float) > std::size_t(p.alignedByteSize))
        throw ContractError("output " + std::to_string(i) + " reads past its allocation");
      out[e] = base[offset];
      for (int d = valid.numDimensions - 1; d >= 0; --d) {
        if (++idx[std::size_t(d)] < valid.dimensionSize[d]) break;
        idx[std::size_t(d)] = 0;
      }
    }
    return out;
  }
};

// Locate the output index carrying one exact declared shape; the three PartA
// and two PartB output shapes all differ, so the shape identifies them.
int output_by_shape(const std::vector<hbDNNTensorProperties>& props,
                    std::initializer_list<int> expected) {
  int found = -1;
  for (std::size_t i = 0; i < props.size(); ++i) {
    const auto& v = props[i].validShape;
    if (shape_equals(v.dimensionSize, v.numDimensions, expected)) {
      if (found >= 0) throw ContractError("two outputs share a declared shape");
      found = int(i);
    }
  }
  if (found < 0) throw ContractError("no output carries the expected declared shape");
  return found;
}

// The packed NV12 input is the one input whose physical packing the SDK may
// report either as a flat byte buffer or as the (1,3,H,W) image layout whose
// element count is 3*H*W — both describe the same submitted bytes.
bool nv12_shape_ok(const hbDNNTensorProperties& in) {
  const auto& v = in.validShape;
  if (v.numDimensions <= 0) return false;
  std::int64_t count = 1;
  for (int d = 0; d < v.numDimensions; ++d) {
    if (v.dimensionSize[d] <= 0) return false;
    count *= v.dimensionSize[d];
  }
  if (count == std::int64_t(kNv12Bytes)) return true;
  return shape_equals(v.dimensionSize, v.numDimensions, {1, 3, kInputSize, kInputSize});
}

// The pooled PartB input must be exactly the declared NCHW [150,256,7,7]
// batch: an NHWC-spelled artifact of the same element count would be fed
// wrong bytes, so only the declared layout is accepted.
bool pooled_shape_ok(const hbDNNTensorProperties& in) {
  const auto& v = in.validShape;
  return shape_equals(v.dimensionSize, v.numDimensions,
                      {kNumProposals, 256, kPoolOutputSize, kPoolOutputSize});
}

}  // namespace

struct FasterRCNN::Impl {
  explicit Impl(const Config& cfg) : config(cfg) {
    if (config.layout != PartbLayout::kPooled)
      throw ContractError("the x5 build requires the pooled PartB layout");

    parta.load(config.parta_path);
    if (parta.input_props.size() != 1 || parta.output_props.size() != 3)
      throw ContractError("PartA must expose one NV12 input and three outputs");
    const hbDNNTensorProperties& in = parta.input_props[0];
    if (in.tensorType != HB_DNN_IMG_TYPE_NV12)
      throw ContractError("PartA input is not an NV12 image input");
    if (!nv12_shape_ok(in))
      throw ContractError("PartA input is not a 320x320 packed NV12 tensor");
    if (std::size_t(in.alignedByteSize) < kNv12Bytes)
      throw ContractError("PartA input storage is smaller than the packed NV12 payload");
    idx_obj = output_by_shape(parta.output_props, {kAnchorCount, 1});
    idx_deltas = output_by_shape(parta.output_props, {kAnchorCount, 4});
    idx_feat0 = output_by_shape(parta.output_props, {1, 256, 10, 10});
    parta.alloc_input(0, int(kNv12Bytes), "hbSysAllocCachedMem parta input failed");
    parta.alloc_outputs();

    partb.load(config.partb_path);
    if (partb.input_props.size() != 1 || partb.output_props.size() != 2)
      throw ContractError("PartB must expose one pooled input and two outputs");
    const hbDNNTensorProperties& pin = partb.input_props[0];
    // The product PartB takes the pooled batch as a float featuremap
    // (makertbin input_type_rt=featuremap): the graph quantizes on chip with
    // its recorded calibration threshold, so the input tensor is float32.
    if (pin.tensorType != HB_DNN_TENSOR_TYPE_F32)
      throw ContractError("PartB input is not float32 (featuremap contract)");
    if (!pooled_shape_ok(pin))
      throw ContractError("PartB input must be the declared [150,256,7,7] float32 batch");
    if (std::size_t(pin.alignedByteSize) < kPooledCount * sizeof(float))
      throw ContractError("PartB input storage cannot hold the [150,256,7,7] float32 batch");
    idx_cls = output_by_shape(partb.output_props, {kNumProposals, kNumClasses});
    idx_reg = output_by_shape(partb.output_props, {kNumProposals, kNumClasses * 4});
    partb.alloc_input(0, int(kPooledCount * sizeof(float)),
                      "hbSysAllocCachedMem partb input failed");
    partb.alloc_outputs();
  }

  Prepared preprocess(const Input& input) const {
    validate_input(input);
    const LetterboxContext ctx = letterbox_geometry(input.rows, input.cols);
    const cv::Mat source(input.rows, input.cols, CV_8UC3,
                         const_cast<std::uint8_t*>(input.bgr.data()));
    const cv::Mat canvas = letterbox_canvas(source, ctx);
    std::vector<std::uint8_t> y, uv;
    nv12_planes(canvas, &y, &uv);
    Prepared prepared;
    prepared.context = ctx;
    prepared.nv12.resize(kNv12Bytes);
    std::memcpy(prepared.nv12.data(), y.data(), y.size());
    std::memcpy(prepared.nv12.data() + y.size(), uv.data(), uv.size());
    return prepared;
  }

  PartaOutputs run_parta(const Prepared& prepared) const {
    if (prepared.nv12.size() != kNv12Bytes)
      throw std::invalid_argument("prepared input does not match the packed NV12 contract");
    std::memcpy(parta.inputs[0].sysMem[0].virAddr, prepared.nv12.data(), kNv12Bytes);
    check(hbSysFlushMem(&parta.inputs[0].sysMem[0], HB_SYS_MEM_CACHE_CLEAN),
          "hbSysFlushMem parta input failed");
    parta.run();
    PartaOutputs out;
    out.rpn_obj = parta.read_float(std::size_t(idx_obj), {kAnchorCount, 1});
    out.rpn_obj.resize(kAnchorCount);
    out.rpn_deltas = parta.read_float(std::size_t(idx_deltas), {kAnchorCount, 4});
    out.feat0 = parta.read_float(std::size_t(idx_feat0), {1, 256, 10, 10});
    return out;
  }

  std::pair<std::vector<float>, std::vector<float>> run_partb(
      const std::vector<float>& pooled) const {
    if (pooled.size() != kPooledCount)
      throw std::invalid_argument("pooled batch has the wrong element count");
    std::memcpy(partb.inputs[0].sysMem[0].virAddr, pooled.data(),
                kPooledCount * sizeof(float));
    check(hbSysFlushMem(&partb.inputs[0].sysMem[0], HB_SYS_MEM_CACHE_CLEAN),
          "hbSysFlushMem partb input failed");
    partb.run();
    return {partb.read_float(std::size_t(idx_cls), {kNumProposals, kNumClasses}),
            partb.read_float(std::size_t(idx_reg), {kNumProposals, kNumClasses * 4})};
  }

  void validate_input(const Input& input) const {
    if (input.rows <= 0 || input.cols <= 0 ||
        input.bgr.size() != std::size_t(input.rows) * std::size_t(input.cols) * 3)
      throw std::invalid_argument("input pixels do not match the declared source geometry");
  }

  Config config;
  // The public detector API is const (predict/preprocess/infer), but the
  // board sessions own mutable runtime state — input/output sysMem buffers
  // and submitted tasks — so the graphs are implementation state, like
  // last_proposals below.
  mutable X5Graph parta, partb;
  int idx_obj = -1, idx_deltas = -1, idx_feat0 = -1, idx_cls = -1, idx_reg = -1;
  mutable int last_proposals = 0;
};

#elif defined(FRCNN_TARGET_S)

namespace {

// Owns one submitted UCP task: released exactly once on every path.
struct STaskGuard {
  hbUCPTaskHandle_t task = nullptr;
  STaskGuard() = default;
  STaskGuard(const STaskGuard&) = delete;
  STaskGuard& operator=(const STaskGuard&) = delete;
  ~STaskGuard() {
    if (task) hbUCPReleaseTask(task);
  }
  int release() {
    const hbUCPTaskHandle_t done = task;
    task = nullptr;
    return hbUCPReleaseTask(done);
  }
};

// Byte-stride alignment the S toolchain applies when resolving a dynamic
// (-1) stride: S600 aligns to 64, the other S targets to 32 (same rule as
// the shared c_utils allocator).
std::int64_t s_stride_align(std::int64_t value) {
#ifdef SOC_S600
  const std::int64_t alignment = 64;
#else
  const std::int64_t alignment = 32;
#endif
  return (value + alignment - 1) & ~(alignment - 1);
}

struct SGraph {
  hbDNNPackedHandle_t packed = nullptr;
  hbDNNHandle_t model = nullptr;
  std::vector<hbDNNTensor> inputs, outputs;
  std::vector<bool> input_allocated, output_allocated;

  ~SGraph() {
    for (std::size_t i = 0; i < inputs.size(); ++i)
      if (i < input_allocated.size() && input_allocated[i]) hbUCPFree(&inputs[i].sysMem);
    for (std::size_t i = 0; i < outputs.size(); ++i)
      if (i < output_allocated.size() && output_allocated[i]) hbUCPFree(&outputs[i].sysMem);
    if (packed) hbDNNRelease(packed);
  }

  const hbDNNTensorProperties& input_props(std::size_t i) const { return inputs[i].properties; }
  const hbDNNTensorProperties& output_props(std::size_t i) const { return outputs[i].properties; }

  void load(const std::string& path) {
    const char* file = path.c_str();
    check(hbDNNInitializeFromFiles(&packed, &file, 1), "hbDNNInitializeFromFiles failed");
    const char** names = nullptr;
    int model_count = 0;
    check(hbDNNGetModelNameList(&names, &model_count, packed), "hbDNNGetModelNameList failed");
    if (model_count != 1)
      throw ContractError("each Faster R-CNN artifact must contain exactly one model: " + path);
    check(hbDNNGetModelHandle(&model, packed, names[0]), "hbDNNGetModelHandle failed");
    int32_t in = 0, out = 0;
    check(hbDNNGetInputCount(&in, model), "hbDNNGetInputCount failed");
    check(hbDNNGetOutputCount(&out, model), "hbDNNGetOutputCount failed");
    inputs.assign(std::size_t(in), hbDNNTensor{});
    outputs.assign(std::size_t(out), hbDNNTensor{});
    input_allocated.assign(std::size_t(in), false);
    output_allocated.assign(std::size_t(out), false);
    for (auto& t : inputs) std::memset(&t, 0, sizeof(t));
    for (auto& t : outputs) std::memset(&t, 0, sizeof(t));
    for (int32_t i = 0; i < in; ++i)
      check(hbDNNGetInputTensorProperties(&inputs[std::size_t(i)].properties, model, i),
            "hbDNNGetInputTensorProperties failed");
    for (int32_t i = 0; i < out; ++i)
      check(hbDNNGetOutputTensorProperties(&outputs[std::size_t(i)].properties, model, i),
            "hbDNNGetOutputTensorProperties failed");
  }

  // Resolve dynamic (-1) byte strides and allocate cached sysMem of
  // stride[0] * dim[0] bytes — the rule the shared S allocator applies.
  void alloc_inputs() {
    for (std::size_t t = 0; t < inputs.size(); ++t) {
      auto& p = inputs[t].properties;
      for (int d = p.validShape.numDimensions - 1; d >= 0; --d) {
        if (p.stride[d] == -1 && d + 1 < p.validShape.numDimensions) {
          std::int64_t cur = p.stride[d + 1] * p.validShape.dimensionSize[d + 1];
          p.stride[d] = s_stride_align(cur);
        }
      }
      const std::int64_t bytes = p.stride[0] * p.validShape.dimensionSize[0];
      check(hbUCPMallocCached(&inputs[t].sysMem, int(bytes), 0), "hbUCPMallocCached input failed");
      input_allocated[t] = true;
    }
  }

  void alloc_outputs() {
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      if (output_allocated[i]) continue;
      check(hbUCPMallocCached(&outputs[i].sysMem, int(outputs[i].properties.alignedByteSize), 0),
            "hbUCPMallocCached output failed");
      output_allocated[i] = true;
    }
  }

  void run() {
    STaskGuard guard;
    check(hbDNNInferV2(&guard.task, outputs.data(), inputs.data(), model), "hbDNNInferV2 failed");
    hbUCPSchedParam schedule{};
    HB_UCP_INITIALIZE_SCHED_PARAM(&schedule);
    schedule.backend = 1ULL << 7;  // HB_UCP_BPU_CORE_ANY (any BPU core)
    schedule.priority = 0;
    check(hbUCPSubmitTask(guard.task, &schedule), "hbUCPSubmitTask failed");
    check(hbUCPWaitTaskDone(guard.task, 0), "hbUCPWaitTaskDone failed");
    check(guard.release(), "hbUCPReleaseTask failed");
    for (auto& t : outputs) check(hbUCPMemFlush(&t.sysMem, HB_SYS_MEM_CACHE_INVALIDATE),
                                  "hbUCPMemFlush output failed");
  }

  // Upload a compact row-major buffer into a strided input tensor, walking
  // the per-dimension byte strides (memcpy per innermost row when that
  // stride equals the element size). Every access is bounds-checked against
  // the allocation first.
  void upload(std::size_t i, const void* src, std::size_t bytes, std::size_t element_count,
              int element_size, const char* what) {
    const auto& p = inputs[i].properties;
    const int rank = p.validShape.numDimensions;
    if (rank <= 0 || rank > 8)
      throw ContractError(std::string(what) + ": unexpected input rank");
    std::int64_t dims[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    std::int64_t strides[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    std::size_t count = 1;
    for (int d = 0; d < rank; ++d) {
      dims[d] = p.validShape.dimensionSize[d];
      strides[d] = p.stride[d];
      count *= std::size_t(dims[d]);
    }
    if (count != element_count || bytes != count * std::size_t(element_size))
      throw ContractError(std::string(what) + ": tensor geometry does not match the payload");
    for (int d = 0; d < rank; ++d)
      if (strides[d] < element_size)
        throw ContractError(std::string(what) + ": byte stride is below the element size");
    const std::int64_t alloc = strides[0] * dims[0];
    const std::uint8_t* from = static_cast<const std::uint8_t*>(src);
    std::uint8_t* to = static_cast<std::uint8_t*>(inputs[i].sysMem.virAddr);
    const std::int64_t inner_bytes = dims[rank - 1] * element_size;
    const bool inner_contiguous = strides[rank - 1] == element_size;
    std::int64_t idx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    std::size_t done = 0;
    while (done < count) {
      std::int64_t offset = 0;
      for (int d = 0; d < rank; ++d) offset += idx[d] * strides[d];
      const std::int64_t chunk = inner_contiguous ? inner_bytes : element_size;
      if (offset + chunk > alloc)
        throw ContractError(std::string(what) + ": upload exceeds the input allocation");
      std::memcpy(to + offset, from + std::int64_t(done) * element_size, std::size_t(chunk));
      done += std::size_t(inner_contiguous ? dims[rank - 1] : 1);
      for (int d = rank - (inner_contiguous ? 2 : 1); d >= 0; --d) {
        if (++idx[std::size_t(d)] < dims[d]) break;
        idx[std::size_t(d)] = 0;
      }
    }
  }

  // Read one FLOAT32 output with an exact declared shape into compact
  // row-major order, walking the per-dimension byte strides.
  std::vector<float> read_float(std::size_t i, std::initializer_list<int> expected) {
    const auto& p = outputs[i].properties;
    if (p.tensorType != HB_DNN_TENSOR_TYPE_F32)
      throw ContractError("output " + std::to_string(i) + " is not float32");
    if (!shape_equals(p.validShape.dimensionSize, p.validShape.numDimensions, expected))
      throw ContractError("output " + std::to_string(i) + " has a shape this sample does "
                          "not publish");
    const int rank = p.validShape.numDimensions;
    std::int64_t dims[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    std::int64_t strides[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    std::size_t count = 1;
    for (int d = 0; d < rank; ++d) {
      dims[d] = p.validShape.dimensionSize[d];
      strides[d] = p.stride[d];
      count *= std::size_t(dims[d]);
    }
    const std::int64_t alloc = p.alignedByteSize;
    const std::uint8_t* base = static_cast<const std::uint8_t*>(outputs[i].sysMem.virAddr);
    std::vector<float> out(count);
    bool compact = true;
    std::int64_t compact_stride = 4;
    for (int d = rank - 1; d >= 0; --d) {
      if (strides[d] != compact_stride) compact = false;
      compact_stride *= dims[d];
    }
    if (compact && alloc >= std::int64_t(count * 4)) {
      std::memcpy(out.data(), base, count * sizeof(float));
      return out;
    }
    std::int64_t idx[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (std::size_t e = 0; e < count; ++e) {
      std::int64_t offset = 0;
      for (int d = 0; d < rank; ++d) offset += idx[d] * strides[d];
      if (offset + 4 > alloc)
        throw ContractError("output " + std::to_string(i) + " reads past its allocation");
      std::memcpy(&out[e], base + offset, 4);
      for (int d = rank - 1; d >= 0; --d) {
        if (++idx[std::size_t(d)] < dims[d]) break;
        idx[std::size_t(d)] = 0;
      }
    }
    return out;
  }

  // Validate the NV12 plane geometry of one split input.  The live S SDK
  // reports the inner strides concretely — Y: stride 1,1; UV: 2,1 (one byte
  // per channel element, `channels` bytes per pixel) — and the outer strides
  // dynamic (-1), which alloc_inputs resolves against this same row geometry
  // (as recorded from the live metadata on s100p).
  void check_nv12_plane(std::size_t i, int rows, int cols, int channels,
                        const char* what) const {
    const auto& p = inputs[i].properties;
    if (p.tensorType != HB_DNN_TENSOR_TYPE_U8)
      throw ContractError(std::string(what) + " is not uint8");
    const auto& v = p.validShape;
    if (v.numDimensions != 4 || v.dimensionSize[0] != 1 || v.dimensionSize[1] != rows ||
        v.dimensionSize[2] != cols || v.dimensionSize[3] != channels)
      throw ContractError(std::string(what) + " geometry does not match the NV12 contract");
    if (p.stride[3] != 1 || p.stride[2] != channels)
      throw ContractError(std::string(what) + " inner strides do not match the plane layout");
    const std::int64_t row_bytes = std::int64_t(cols) * channels;
    if (p.stride[1] != -1 && p.stride[1] < row_bytes)
      throw ContractError(std::string(what) + " row stride does not cover one plane row");
  }
};

int s_output_by_shape(const SGraph& g, std::initializer_list<int> expected) {
  int found = -1;
  for (std::size_t i = 0; i < g.outputs.size(); ++i) {
    const auto& v = g.outputs[i].properties.validShape;
    if (shape_equals(v.dimensionSize, v.numDimensions, expected)) {
      if (found >= 0) throw ContractError("two outputs share a declared shape");
      found = int(i);
    }
  }
  if (found < 0) throw ContractError("no output carries the expected declared shape");
  return found;
}

}  // namespace

struct FasterRCNN::Impl {
  explicit Impl(const Config& cfg) : config(cfg) {
    if (config.layout != PartbLayout::kPacked)
      throw ContractError("the S builds require the packed float32 PartB layout");

    parta.load(config.parta_path);
    if (parta.inputs.size() != 2 || parta.outputs.size() != 3)
      throw ContractError("PartA must expose two NV12 plane inputs and three outputs");
    parta.check_nv12_plane(0, kInputSize, kInputSize, 1, "PartA Y plane");
    parta.check_nv12_plane(1, kInputSize / 2, kInputSize / 2, 2, "PartA UV plane");
    idx_obj = s_output_by_shape(parta, {kAnchorCount, 1});
    idx_deltas = s_output_by_shape(parta, {kAnchorCount, 4});
    idx_feat0 = s_output_by_shape(parta, {1, 256, 10, 10});
    parta.alloc_inputs();
    parta.alloc_outputs();

    partb.load(config.partb_path);
    if (partb.inputs.size() != 1 || partb.outputs.size() != 2)
      throw ContractError("PartB must expose one bank input and two outputs");
    const auto& pin = partb.input_props(0);
    if (pin.tensorType != HB_DNN_TENSOR_TYPE_F32)
      throw ContractError("PartB bank input is not float32");
    if (!shape_equals(pin.validShape.dimensionSize, pin.validShape.numDimensions,
                      {1, 256, kNumProposals * kPoolOutputSize, kPoolOutputSize}))
      throw ContractError("PartB bank input must be the declared [1,256,1050,7] float32 "
                          "ROI bank");
    idx_cls = s_output_by_shape(partb, {1, kNumClasses, kNumProposals, 1});
    idx_reg = s_output_by_shape(partb, {1, kNumClasses * 4, kNumProposals, 1});
    partb.alloc_inputs();
    partb.alloc_outputs();
  }

  Prepared preprocess(const Input& input) const {
    validate_input(input);
    const LetterboxContext ctx = letterbox_geometry(input.rows, input.cols);
    const cv::Mat source(input.rows, input.cols, CV_8UC3,
                         const_cast<std::uint8_t*>(input.bgr.data()));
    const cv::Mat canvas = letterbox_canvas(source, ctx);
    Prepared prepared;
    prepared.context = ctx;
    nv12_planes(canvas, &prepared.y, &prepared.uv);
    return prepared;
  }

  PartaOutputs run_parta(const Prepared& prepared) const {
    if (prepared.y.size() != std::size_t(kInputSize) * kInputSize ||
        prepared.uv.size() != std::size_t(kInputSize) * kInputSize / 2)
      throw std::invalid_argument("prepared input does not match the split NV12 contract");
    parta.upload(0, prepared.y.data(), prepared.y.size(), std::size_t(kInputSize) * kInputSize,
                 1, "PartA Y plane");
    parta.upload(1, prepared.uv.data(), prepared.uv.size(),
                 std::size_t(kInputSize) * kInputSize / 2, 1, "PartA UV plane");
    check(hbUCPMemFlush(&parta.inputs[0].sysMem, HB_SYS_MEM_CACHE_CLEAN),
          "hbUCPMemFlush parta Y failed");
    check(hbUCPMemFlush(&parta.inputs[1].sysMem, HB_SYS_MEM_CACHE_CLEAN),
          "hbUCPMemFlush parta UV failed");
    parta.run();
    PartaOutputs out;
    out.rpn_obj = parta.read_float(std::size_t(idx_obj), {kAnchorCount, 1});
    out.rpn_obj.resize(kAnchorCount);
    out.rpn_deltas = parta.read_float(std::size_t(idx_deltas), {kAnchorCount, 4});
    out.feat0 = parta.read_float(std::size_t(idx_feat0), {1, 256, 10, 10});
    return out;
  }

  std::pair<std::vector<float>, std::vector<float>> run_partb(
      const std::vector<float>& pooled) const {
    if (pooled.size() != kPooledCount)
      throw std::invalid_argument("pooled batch has the wrong element count");
    std::vector<float> bank(kPooledCount);
    pack_bank(pooled.data(), bank.data());
    partb.upload(0, bank.data(), bank.size() * sizeof(float), kPooledCount, 4, "PartB bank");
    check(hbUCPMemFlush(&partb.inputs[0].sysMem, HB_SYS_MEM_CACHE_CLEAN),
          "hbUCPMemFlush partb bank failed");
    partb.run();
    std::vector<float> cls(std::size_t(kNumProposals) * kNumClasses);
    std::vector<float> reg(std::size_t(kNumProposals) * kNumClasses * 4);
    const std::vector<float> cls_packed =
        partb.read_float(std::size_t(idx_cls), {1, kNumClasses, kNumProposals, 1});
    const std::vector<float> reg_packed =
        partb.read_float(std::size_t(idx_reg), {1, kNumClasses * 4, kNumProposals, 1});
    unpack_bank_outputs(cls_packed.data(), reg_packed.data(), cls.data(), reg.data());
    return {std::move(cls), std::move(reg)};
  }

  void validate_input(const Input& input) const {
    if (input.rows <= 0 || input.cols <= 0 ||
        input.bgr.size() != std::size_t(input.rows) * std::size_t(input.cols) * 3)
      throw std::invalid_argument("input pixels do not match the declared source geometry");
  }

  Config config;
  // The public detector API is const (predict/preprocess/infer), but the
  // board sessions own mutable runtime state — input/output sysMem buffers
  // and submitted tasks — so the graphs are implementation state, like
  // last_proposals below.
  mutable SGraph parta, partb;
  int idx_obj = -1, idx_deltas = -1, idx_feat0 = -1, idx_cls = -1, idx_reg = -1;
  mutable int last_proposals = 0;
};

#else
#error "FRCNN_BOARD requires FRCNN_TARGET_X5 or FRCNN_TARGET_S"
#endif  // FRCNN_TARGET_X5 / FRCNN_TARGET_S target selection

#else  // host build: no board backend

// No board backend: construction fails with the precise reason, but the
// SDK-free stages above stay compilable and host-testable.
struct FasterRCNN::Impl {
  explicit Impl(const Config&) {
    throw std::runtime_error("Faster R-CNN inference requires a board target build "
                             "(FRCNN_TARGET x5, s100, s100p or s600)");
  }
  Prepared preprocess(const Input&) const {
    throw std::runtime_error("Faster R-CNN preprocess requires a board target build");
  }
  PartaOutputs run_parta(const Prepared&) const {
    throw std::runtime_error("Faster R-CNN infer requires a board target build");
  }
  std::pair<std::vector<float>, std::vector<float>> run_partb(
      const std::vector<float>&) const {
    throw std::runtime_error("Faster R-CNN PartB requires a board target build");
  }
  void validate_input(const Input&) const {
    throw std::runtime_error("Faster R-CNN requires a board target build");
  }
  mutable int last_proposals = 0;
};

#endif  // FRCNN_BOARD

// ------------------------------------------------------------ public: model ----

FasterRCNN::FasterRCNN(const Config& config) : impl_(std::make_unique<Impl>(config)) {}
FasterRCNN::~FasterRCNN() = default;

FasterRCNN::Prepared FasterRCNN::preprocess(const Input& input) const {
  return impl_->preprocess(input);
}

FasterRCNN::PartaOutputs FasterRCNN::infer(const Prepared& prepared) const {
  return impl_->run_parta(prepared);
}

MiddleResult FasterRCNN::middle(const PartaOutputs& outputs) const {
  if (outputs.rpn_obj.size() != std::size_t(kAnchorCount) ||
      outputs.rpn_deltas.size() != std::size_t(kAnchorCount) * 4)
    throw std::invalid_argument("PartA outputs do not match the 3375-anchor contract");
  return rpn_select(outputs.rpn_obj.data(), outputs.rpn_deltas.data(),
                    anchor_table().data());
}

std::vector<float> FasterRCNN::pool(const std::vector<float>& proposals,
                                    const std::vector<float>& feat0, int n_valid) const {
  if (feat0.size() != kFeat0Count)
    throw std::invalid_argument("feat0 must hold 256x10x10 values");
  if (n_valid < 0 || n_valid > kNumProposals)
    throw std::invalid_argument("n_valid must be within [0,150]");
  if (proposals.size() != std::size_t(kNumProposals) * 4)
    throw std::invalid_argument("proposals must be the padded [150,4] batch");
  std::vector<float> pooled(kPooledCount, 0.0F);
  if (n_valid > 0)
    roi_align_feat0(feat0.data(), 256, 10, 10, proposals.data(), n_valid, pooled.data());
  return pooled;
}

DetectionResult FasterRCNN::postprocess(const PartaOutputs& outputs,
                                        const LetterboxContext& context) const {
  const MiddleResult selected = middle(outputs);
  impl_->last_proposals = selected.n_valid;
  if (selected.n_valid == 0) return DetectionResult{};
  const std::vector<float> pooled = pool(selected.proposals, outputs.feat0, selected.n_valid);
  std::vector<float> cls, reg;
  std::tie(cls, reg) = impl_->run_partb(pooled);
  DetectionResult result = tail(cls.data(), reg.data(), selected.proposals.data(),
                                selected.n_valid);
  for (std::size_t i = 0; i < result.size(); ++i)
    context.to_original(result.boxes.data() + i * 4);
  return result;
}

DetectionResult FasterRCNN::predict(const Input& input) {
  const Prepared prepared = preprocess(input);
  const PartaOutputs outputs = infer(prepared);
  return postprocess(outputs, prepared.context);
}

int FasterRCNN::last_proposals() const { return impl_->last_proposals; }

// ---------------------------------------------------- public: person tracking ----

PersonTracker::PersonTracker(FasterRCNN& detector, const Config& config, unsigned threads)
    : detector_(&detector),
      config_(config),
      tracker_(std::make_unique<bytetrack::ByteTracker>(config.tracking, threads)) {}

FasterRCNN::Prepared PersonTracker::preprocess(const FasterRCNN::Input& input) const {
  return detector_->preprocess(input);
}

FasterRCNN::PartaOutputs PersonTracker::infer(const FasterRCNN::Prepared& prepared) const {
  return detector_->infer(prepared);
}

std::vector<bytetrack::Track> PersonTracker::postprocess(
    const FasterRCNN::PartaOutputs& outputs, const LetterboxContext& context) {
  const DetectionResult result = detector_->postprocess(outputs, context);
  std::vector<bytetrack::Detection> persons;
  for (std::size_t i = 0; i < result.size(); ++i) {
    const float* b = result.boxes.data() + i * 4;
    if (result.labels[i] != kPersonLabel) continue;
    if (!(b[2] > b[0] && b[3] > b[1])) continue;
    bytetrack::Detection d;
    d.x1 = b[0];
    d.y1 = b[1];
    d.x2 = b[2];
    d.y2 = b[3];
    d.score = result.scores[i];
    persons.push_back(d);
  }
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<bytetrack::Track> tracks =
      tracker_->update(persons, context.original_rows, context.original_cols);
  last_tracker_us_ = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
  return tracks;
}

std::vector<bytetrack::Track> PersonTracker::predict(const FasterRCNN::Input& input) {
  const FasterRCNN::Prepared prepared = preprocess(input);
  const FasterRCNN::PartaOutputs outputs = infer(prepared);
  return postprocess(outputs, prepared.context);
}

std::vector<Detection> PersonTracker::detect_persons(const FasterRCNN::Input& input) const {
  const DetectionResult result = detector_->predict(input);
  std::vector<Detection> persons;
  for (std::size_t i = 0; i < result.size(); ++i) {
    const float* b = result.boxes.data() + i * 4;
    if (result.labels[i] != kPersonLabel) continue;
    if (!(b[2] > b[0] && b[3] > b[1])) continue;
    Detection d;
    d.label = kPersonLabel;
    d.score = result.scores[i];
    d.box[0] = b[0];
    d.box[1] = b[1];
    d.box[2] = b[2];
    d.box[3] = b[3];
    persons.push_back(d);
  }
  return persons;
}

int PersonTracker::frame_id() const { return tracker_->frame_id(); }

void PersonTracker::reset() { tracker_->reset(); }

double PersonTracker::last_tracker_us() const { return last_tracker_us_; }

}  // namespace fasterrcnn
