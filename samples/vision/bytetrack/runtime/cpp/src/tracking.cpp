// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
/// BYTETracker implementation, ported statement by statement from the
/// sample's Python tracker_backend (byte_tracker.py / kalman_filter.py /
/// matching.py). The numeric contract is deliberate:
///
///  * Detection coordinates stay float until tlbr->tlwh (the width/height
///    subtraction happens in float32 exactly like the numpy float32 path),
///    then the Kalman filter, IoU, fusion and assignment run in double.
///  * Score grouping comparisons happen in float32 against float32-rounded
///    thresholds (numpy compares a float32 array against a python float in
///    float32); det_thresh = track_thresh + 0.1 is formed in double like
///    the Python constructor, then rounded to float for its comparison.
///  * IoU follows the cython_bbox convention: inclusive +1 extents,
///    union = areaA + areaB - iw*ih with the same expression order.
///  * Assignment calls the vendored lap 0.5.12 lapjv_internal on the
///    (rows+cols) matrix padded with cost_limit/2 and a zero bottom-right
///    block, then maps assignments >= the other side to -1, exactly like
///    lap/_lapjv.pyx with extend_cost=True, cost_limit=thresh.
///  * Memory: per-frame detection STracks that never enter the persistent
///    tracked/lost/removed lists are released at the end of the update
///    (Python drops them when the frame's locals go away); tracks in the
///    removed list are kept, matching the Python observable lifecycle.
#include "tracking.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numeric>
#include <unordered_set>
#include <utility>
#include <vector>

#include "lapjv.h"
#include "parallel.hpp"

#if defined(BYTETRACK_TARGET_X5) || defined(BYTETRACK_TARGET_S)
#include "detect.hpp"
#ifndef BYTETRACK_TARGET_NAME
#define BYTETRACK_TARGET_NAME "unknown"
#endif
#endif

namespace bytetrack {
namespace {

constexpr double kStdWeightPosition = 1.0 / 20.0;
constexpr double kStdWeightVelocity = 1.0 / 160.0;

using Vec8 = std::array<double, 8>;
using Mat8 = std::array<std::array<double, 8>, 8>;
using Mat4 = std::array<std::array<double, 4>, 4>;

/// Fixed-shape Kalman filter for [cx, cy, a, h, vx, vy, va, vh]; the motion
/// matrix is [[I, I], [0, I]] and the update matrix selects the first four
/// states, so every product below is a two-term sum with the same value the
/// numpy dense multiply produces.
struct KalmanFilter {
  static void initiate(const double meas[4], Vec8& mean, Mat8& cov) {
    for (int i = 0; i < 4; ++i) {
      mean[i] = meas[i];
      mean[i + 4] = 0.0;
    }
    const double h = meas[3];
    const double std1[8] = {2 * kStdWeightPosition * h, 2 * kStdWeightPosition * h, 1e-2,
                            2 * kStdWeightPosition * h, 10 * kStdWeightVelocity * h,
                            10 * kStdWeightVelocity * h, 1e-5, 10 * kStdWeightVelocity * h};
    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 8; ++j) cov[i][j] = (i == j) ? std1[i] * std1[i] : 0.0;
  }

  static void predict(const Vec8& m, const Mat8& p, Vec8& om, Mat8& op) {
    const double h = m[3];
    const double q[8] = {kStdWeightPosition * h, kStdWeightPosition * h, 1e-2,
                         kStdWeightPosition * h, kStdWeightVelocity * h,
                         kStdWeightVelocity * h, 1e-5, kStdWeightVelocity * h};
    for (int i = 0; i < 4; ++i) om[i] = m[i] + m[i + 4];
    for (int i = 4; i < 8; ++i) om[i] = m[i];
    double fp[8][8];
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 8; ++j) fp[i][j] = p[i][j] + p[i + 4][j];
    for (int i = 4; i < 8; ++i)
      for (int j = 0; j < 8; ++j) fp[i][j] = p[i][j];
    for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 4; ++j) op[i][j] = fp[i][j] + fp[i][j + 4];
      for (int j = 4; j < 8; ++j) op[i][j] = fp[i][j];
    }
    for (int i = 0; i < 8; ++i) op[i][i] += q[i] * q[i];
  }

  static void project(const Vec8& m, const Mat8& p, double pm[4], Mat4& s) {
    const double h = m[3];
    const double r[4] = {kStdWeightPosition * h, kStdWeightPosition * h, 1e-1,
                         kStdWeightPosition * h};
    for (int i = 0; i < 4; ++i) {
      pm[i] = m[i];
      for (int j = 0; j < 4; ++j) s[i][j] = p[i][j];
    }
    for (int i = 0; i < 4; ++i) s[i][i] += r[i] * r[i];
  }

  static void update(const Vec8& m, const Mat8& p, const double meas[4], Vec8& nm, Mat8& np) {
    double pm[4];
    Mat4 s;
    project(m, p, pm, s);
    // Lower Cholesky of the 4x4 projected covariance (scipy cho_factor,
    // lower=True). If S is not numerically positive definite the Python run
    // would produce NaNs from LAPACK; fail loudly instead.
    double l[4][4] = {};
    for (int j = 0; j < 4; ++j) {
      double d = s[j][j];
      for (int k = 0; k < j; ++k) d -= l[j][k] * l[j][k];
      if (!(d > 0.0) || !std::isfinite(d))
        throw std::runtime_error("tracker update failed: covariance lost positive definiteness");
      l[j][j] = std::sqrt(d);
      for (int i = j + 1; i < 4; ++i) {
        double e = s[i][j];
        for (int k = 0; k < j; ++k) e -= l[i][k] * l[j][k];
        l[i][j] = e / l[j][j];
      }
    }
    // Solve L L^T X = B with B = (P H^T)^T, i.e. B[r][c] = P[c][r]; the gain
    // is K = X^T (cho_solve(...).T in kalman_filter.py).
    double x[4][8];
    for (int c = 0; c < 8; ++c) {
      double y[4];
      for (int i = 0; i < 4; ++i) {
        double a = p[c][i];
        for (int k = 0; k < i; ++k) a -= l[i][k] * y[k];
        y[i] = a / l[i][i];
      }
      for (int i = 3; i >= 0; --i) {
        double a = y[i];
        for (int k = i + 1; k < 4; ++k) a -= l[k][i] * x[k][c];
        x[i][c] = a / l[i][i];
      }
    }
    double inn[4];
    for (int i = 0; i < 4; ++i) inn[i] = meas[i] - pm[i];
    for (int c = 0; c < 8; ++c) {
      double a = 0.0;
      for (int k = 0; k < 4; ++k) a += inn[k] * x[k][c];
      nm[c] = m[c] + a;
    }
    // new_cov = P - K S K^T with K[i][k] = X[k][i] (multi_dot order K@S@K.T).
    double ks[8][4];
    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 4; ++j) {
        double a = 0.0;
        for (int k = 0; k < 4; ++k) a += x[k][i] * s[k][j];
        ks[i][j] = a;
      }
    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 8; ++j) {
        double a = 0.0;
        for (int k = 0; k < 4; ++k) a += ks[i][k] * x[k][j];
        np[i][j] = p[i][j] - a;
      }
  }
};

enum class TState : int8_t { New = 0, Tracked = 1, Lost = 2, Removed = 3 };

struct STrack {
  double tlwh[4] = {0., 0., 0., 0.};  // stored float64 (np.asarray(..., float64))
  double score = 0.;
  bool is_activated = false;
  TState state = TState::New;
  int track_id = 0;
  int tracklet_len = 0;
  int frame_id = 0;
  int start_frame = 0;
  bool has_state = false;
  Vec8 mean{};
  Mat8 covariance{};

  /// Detection constructor: the width/height subtraction stays in float32
  /// (the numpy float32 tlbr->tlwh step) and everything after is double.
  static STrack from_tlbr(const float tlbr[4], double det_score) {
    STrack t;
    const float w = tlbr[2] - tlbr[0];
    const float h = tlbr[3] - tlbr[1];
    t.tlwh[0] = static_cast<double>(tlbr[0]);
    t.tlwh[1] = static_cast<double>(tlbr[1]);
    t.tlwh[2] = static_cast<double>(w);
    t.tlwh[3] = static_cast<double>(h);
    t.score = det_score;  // float32 value promoted exactly
    return t;
  }

  /// tlwh property (from the Kalman mean once activated).
  void get_tlwh(double out[4]) const {
    if (!has_state) {
      out[0] = tlwh[0];
      out[1] = tlwh[1];
      out[2] = tlwh[2];
      out[3] = tlwh[3];
      return;
    }
    const double w = mean[2] * mean[3];  // ret[2] *= ret[3] before ret[:2] -= ret[2:]/2
    out[0] = mean[0] - w / 2.0;
    out[1] = mean[1] - mean[3] / 2.0;
    out[2] = w;
    out[3] = mean[3];
  }

  void get_tlbr(double out[4]) const {
    get_tlwh(out);
    out[2] += out[0];
    out[3] += out[1];
  }

  /// tlwh_to_xyah measurement for the Kalman filter (all double).
  void xyah(double out[4]) const {
    out[0] = tlwh[0] + tlwh[2] / 2.0;
    out[1] = tlwh[1] + tlwh[3] / 2.0;
    out[2] = tlwh[2] / tlwh[3];
    out[3] = tlwh[3];
  }

  /// python STrack.update: matched track absorbs the detection's box/score.
  void do_update(STrack* det, int fid) {
    frame_id = fid;
    tracklet_len += 1;
    double meas[4];
    det->xyah(meas);
    KalmanFilter::update(mean, covariance, meas, mean, covariance);
    state = TState::Tracked;
    is_activated = true;
    score = det->score;
  }

  /// python STrack.re_activate(new_id=false): lost track matched again.
  void do_re_activate(STrack* det, int fid) {
    double meas[4];
    det->xyah(meas);
    KalmanFilter::update(mean, covariance, meas, mean, covariance);
    tracklet_len = 0;
    state = TState::Tracked;
    is_activated = true;
    frame_id = fid;
    score = det->score;
  }

  /// python STrack.activate: new tracklet from an unassociated detection.
  void do_activate(long long& id_counter, int fid) {
    track_id = static_cast<int>(++id_counter);
    double meas[4];
    xyah(meas);
    KalmanFilter::initiate(meas, mean, covariance);
    has_state = true;
    tracklet_len = 0;
    state = TState::Tracked;
    if (fid == 1) is_activated = true;
    frame_id = fid;
    start_frame = fid;
  }
};

void track_predict(STrack& t) {
  Vec8 m = t.mean;
  if (t.state != TState::Tracked) m[7] = 0;
  KalmanFilter::predict(m, t.covariance, t.mean, t.covariance);
}

/// cython_bbox.bbox_overlaps convention, double precision, per cell.
double bbox_iou(const double a[4], const double b[4]) {
  const double iw = std::min(a[2], b[2]) - std::max(a[0], b[0]) + 1.0;
  if (iw > 0.0) {
    const double ih = std::min(a[3], b[3]) - std::max(a[1], b[1]) + 1.0;
    if (ih > 0.0) {
      const double ua = (a[2] - a[0] + 1.0) * (a[3] - a[1] + 1.0) +
                        (b[2] - b[0] + 1.0) * (b[3] - b[1] + 1.0) - iw * ih;
      return iw * ih / ua;
    }
  }
  return 0.0;
}

struct Matrix {
  int rows = 0, cols = 0;
  std::vector<double> v;
  void reset(int r, int c) {
    rows = r;
    cols = c;
    v.assign(static_cast<size_t>(r) * c, 0.0);
  }
  double& at(int r, int c) { return v[static_cast<size_t>(r) * cols + c]; }
  double at(int r, int c) const { return v[static_cast<size_t>(r) * cols + c]; }
};

struct Assignment {
  std::vector<std::pair<int, int>> matches;
  std::vector<int> unmatched_a, unmatched_b;
};

/// lap.lapjv(cost, extend_cost=True, cost_limit=thresh) followed by the
/// linear_assignment unpacking in matching.py. thresh is always finite in
/// this tracker (validated [0,1] config), so the cost_limit branch applies:
/// the matrix grows to (rows+cols)^2 padded with thresh/2 and a zero
/// bottom-right block; assignments into the padding become -1.
Assignment linear_assignment(const Matrix& cost, double thresh) {
  Assignment out;
  const int rows = cost.rows, cols = cost.cols;
  if (rows == 0 || cols == 0) {
    out.unmatched_a.resize(rows);
    std::iota(out.unmatched_a.begin(), out.unmatched_a.end(), 0);
    out.unmatched_b.resize(cols);
    std::iota(out.unmatched_b.begin(), out.unmatched_b.end(), 0);
    return out;
  }
  const size_t n = static_cast<size_t>(rows) + static_cast<size_t>(cols);
  std::vector<double> ext(n * n, thresh / 2.0);
  for (size_t i = rows; i < n; ++i)
    for (size_t j = cols; j < n; ++j) ext[i * n + j] = 0.0;
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j) ext[static_cast<size_t>(i) * n + j] = cost.at(i, j);
  std::vector<double*> ptrs(n);
  for (size_t i = 0; i < n; ++i) ptrs[i] = ext.data() + i * n;
  std::vector<int> x(n), y(n);
  const int ret = lapjv_internal(static_cast<uint_t>(n), ptrs.data(), x.data(), y.data());
  if (ret != 0) throw std::runtime_error("lapjv_internal failed");
  for (int i = 0; i < rows; ++i)
    if (x[i] >= cols) x[i] = -1;
  for (int j = 0; j < cols; ++j)
    if (y[j] >= rows) y[j] = -1;
  for (int i = 0; i < rows; ++i)
    if (x[i] >= 0) out.matches.emplace_back(i, x[i]);
  for (int i = 0; i < rows; ++i)
    if (x[i] < 0) out.unmatched_a.push_back(i);
  for (int j = 0; j < cols; ++j)
    if (y[j] < 0) out.unmatched_b.push_back(j);
  return out;
}

/// joint_stracks: a's order first, then b entries whose id is new.
void joint_stracks(std::vector<STrack*>& a, const std::vector<STrack*>& b,
                   std::vector<STrack*>& out) {
  out = a;
  std::unordered_set<int> ids;
  ids.reserve(a.size() + b.size());
  for (const STrack* t : a) ids.insert(t->track_id);
  for (STrack* t : b)
    if (ids.insert(t->track_id).second) out.push_back(t);
}

/// sub_stracks: a minus b, by id, preserving a's first-occurrence order (the
/// input lists hold unique ids, matching the Python dict behaviour).
void sub_stracks(const std::vector<STrack*>& a, const std::vector<STrack*>& b,
                 std::vector<STrack*>& out) {
  std::unordered_set<int> remove;
  remove.reserve(b.size());
  for (const STrack* t : b) remove.insert(t->track_id);
  out.clear();
  out.reserve(a.size());
  for (STrack* t : a)
    if (remove.find(t->track_id) == remove.end()) out.push_back(t);
}

/// remove_duplicate_stracks: pairwise cost 1-IoU < 0.15 drops the younger
/// side; a lifetime tie removes from the tracked list (python `else` branch).
void remove_duplicate_stracks(std::vector<STrack*>& a, std::vector<STrack*>& b) {
  Matrix pdist;
  pdist.reset(static_cast<int>(a.size()), static_cast<int>(b.size()));
  std::vector<double> at(a.size() * 4), bt(b.size() * 4);
  for (size_t i = 0; i < a.size(); ++i) a[i]->get_tlbr(&at[i * 4]);
  for (size_t j = 0; j < b.size(); ++j) b[j]->get_tlbr(&bt[j * 4]);
  for (size_t i = 0; i < a.size(); ++i)
    for (size_t j = 0; j < b.size(); ++j)
      pdist.at(static_cast<int>(i), static_cast<int>(j)) = 1.0 - bbox_iou(&at[i * 4], &bt[j * 4]);
  std::unordered_set<int> dupa, dupb;
  for (size_t i = 0; i < a.size(); ++i)
    for (size_t j = 0; j < b.size(); ++j)
      if (pdist.at(static_cast<int>(i), static_cast<int>(j)) < 0.15) {
        const int timep = a[i]->frame_id - a[i]->start_frame;
        const int timeq = b[j]->frame_id - b[j]->start_frame;
        if (timep > timeq)
          dupb.insert(static_cast<int>(j));
        else
          dupa.insert(static_cast<int>(i));
      }
  std::vector<STrack*> ra, rb;
  ra.reserve(a.size());
  rb.reserve(b.size());
  for (size_t i = 0; i < a.size(); ++i)
    if (dupa.find(static_cast<int>(i)) == dupa.end()) ra.push_back(a[i]);
  for (size_t j = 0; j < b.size(); ++j)
    if (dupb.find(static_cast<int>(j)) == dupb.end()) rb.push_back(b[j]);
  a.swap(ra);
  b.swap(rb);
}

void validate_config(const TrackingConfig& c) {
  const double th[2] = {c.track_thresh, c.match_thresh};
  for (double v : th)
    if (!std::isfinite(v) || v < 0.0 || v > 1.0)
      throw TrackingConfigError("Tracking thresholds must be finite [0,1].");
  if (c.track_buffer < 0) throw TrackingConfigError("track_buffer must be a nonnegative integer.");
  if (c.frame_rate <= 0) throw TrackingConfigError("frame_rate must be a positive integer.");
}

void validate_detections(const std::vector<Detection>& dets) {
  for (const Detection& d : dets) {
    const float c[5] = {d.x1, d.y1, d.x2, d.y2, d.score};
    for (float v : c)
      if (!std::isfinite(v))
        throw TrackingInputError("detections must have finite coordinates and scores");
    if (!(d.x2 > d.x1 && d.y2 > d.y1))
      throw TrackingInputError("detections must have positive area (x2 > x1, y2 > y1)");
  }
}

}  // namespace

struct ByteTracker::Impl {
  TrackingConfig cfg;
  // numpy compares float32 scores in float32; build the comparison thresholds
  // once (track_thresh and det_thresh are double in the Python constructor).
  float track_thresh_f32 = 0.f;
  float det_thresh_f32 = 0.f;
  int buffer_size = 0;
  int max_time_lost = 0;
  int frame_id = 0;
  long long next_id = 0;  // instance-local, survives reset (no ID reuse)
  std::vector<std::unique_ptr<STrack>> owned;
  std::vector<STrack*> tracked_stracks, lost_stracks, removed_stracks;
  std::unique_ptr<ParallelRunner> pool;
  std::size_t parallel_engaged = 0, parallel_inline = 0;  // monitoring only
  // Scratch reused across frames to keep the steady state allocation-light.
  Matrix dists;
  std::vector<double> atlbrs, btlbrs;

  Impl(const TrackingConfig& c, unsigned threads) : cfg(c) {
    validate_config(c);
    track_thresh_f32 = static_cast<float>(c.track_thresh);
    det_thresh_f32 = static_cast<float>(c.track_thresh + 0.1);
    buffer_size = static_cast<int>(c.frame_rate / 30.0 * c.track_buffer);
    max_time_lost = buffer_size;
    if (threads > 1) pool = std::make_unique<ParallelRunner>(threads);
  }

  void iou_distance(const std::vector<STrack*>& a, const std::vector<STrack*>& b, Matrix& cost) {
    cost.reset(static_cast<int>(a.size()), static_cast<int>(b.size()));
    if (cost.v.empty()) return;
    atlbrs.resize(a.size() * 4);
    btlbrs.resize(b.size() * 4);
    for (size_t i = 0; i < a.size(); ++i) a[i]->get_tlbr(&atlbrs[i * 4]);
    for (size_t j = 0; j < b.size(); ++j) b[j]->get_tlbr(&btlbrs[j * 4]);
    const double* A = atlbrs.data();
    const double* B = btlbrs.data();
    const int cols = cost.cols;
    double* C = cost.v.data();
    const size_t rows = a.size();
    auto row_block = [A, B, C, cols](size_t begin, size_t end) {
      for (size_t i = begin; i < end; ++i) {
        double* crow = C + i * static_cast<size_t>(cols);
        const double* bx = A + i * 4;
        for (int j = 0; j < cols; ++j) crow[j] = 1.0 - bbox_iou(bx, B + j * 4);
      }
    };
    if (pool) {
      if (pool->run(rows, 24, row_block))
        ++parallel_engaged;
      else
        ++parallel_inline;
    } else {
      row_block(0, rows);
    }
  }

  void fuse_score(Matrix& cost, const std::vector<STrack*>& dets) {
    if (cost.v.empty()) return;
    for (int i = 0; i < cost.rows; ++i)
      for (int j = 0; j < cost.cols; ++j)
        cost.at(i, j) = 1.0 - (1.0 - cost.at(i, j)) * dets[j]->score;
  }

  /// Release per-frame detection STracks that never entered a persistent
  /// list (Python frees them with the frame); tracked/lost/removed stay.
  void sweep_frame_temporaries() {
    std::unordered_set<const STrack*> live;
    live.reserve(tracked_stracks.size() + lost_stracks.size() + removed_stracks.size());
    for (const STrack* t : tracked_stracks) live.insert(t);
    for (const STrack* t : lost_stracks) live.insert(t);
    for (const STrack* t : removed_stracks) live.insert(t);
    owned.erase(std::remove_if(owned.begin(), owned.end(),
                               [&live](const std::unique_ptr<STrack>& p) {
                                 return live.find(p.get()) == live.end();
                               }),
                owned.end());
  }
};

ByteTracker::ByteTracker(const TrackingConfig& config, unsigned threads)
    : impl_(std::make_unique<Impl>(config, threads)) {}

ByteTracker::~ByteTracker() = default;

int ByteTracker::frame_id() const { return impl_->frame_id; }

std::size_t ByteTracker::persistent_track_count() const { return impl_->owned.size(); }

ByteTracker::ParallelUsage ByteTracker::parallel_usage() const {
  ParallelUsage u;
  u.engaged = impl_->parallel_engaged;
  u.inline_fallback = impl_->parallel_inline;
  u.slots = impl_->pool ? impl_->pool->slots() : 0u;
  return u;
}

void ByteTracker::reset() {
  impl_->owned.clear();
  impl_->tracked_stracks.clear();
  impl_->lost_stracks.clear();
  impl_->removed_stracks.clear();
  impl_->frame_id = 0;
  // next_id is deliberately NOT reset: IDs stay monotonic per instance.
}

std::vector<Track> ByteTracker::update(const std::vector<Detection>& detections, int img_h,
                                       int img_w) {
  if (img_h <= 0 || img_w <= 0)
    throw TrackingInputError("image size must be positive");
  validate_detections(detections);
  Impl& im = *impl_;
  const int fid = ++im.frame_id;

  std::vector<STrack*> activated_starcks, refind_stracks, lost_now, removed_now;
  std::vector<STrack*> detections_list, detections_second;

  // remain: score > track_thresh (float32); second: 0.1 < score < track_thresh.
  detections_list.reserve(detections.size());
  detections_second.reserve(detections.size());
  for (const Detection& d : detections) {
    // Explicit array: `&d.x1` indexed across struct members is not legal
    // array pointer arithmetic in C++; copying the four floats is exact
    // (float copies are bitwise) and keeps the float32 tlbr semantics.
    const float tlbr[4] = {d.x1, d.y1, d.x2, d.y2};
    STrack* s = im.owned.emplace_back(
                    std::make_unique<STrack>(STrack::from_tlbr(tlbr, d.score)))
                    .get();
    if (d.score > im.track_thresh_f32)
      detections_list.push_back(s);
    else if (d.score > 0.1f && d.score < im.track_thresh_f32)
      detections_second.push_back(s);
  }

  std::vector<STrack*> unconfirmed, tracked_stracks;
  unconfirmed.reserve(im.tracked_stracks.size());
  tracked_stracks.reserve(im.tracked_stracks.size());
  for (STrack* t : im.tracked_stracks) {
    if (!t->is_activated)
      unconfirmed.push_back(t);
    else
      tracked_stracks.push_back(t);
  }

  std::vector<STrack*> strack_pool;
  joint_stracks(tracked_stracks, im.lost_stracks, strack_pool);

  // multi_predict over the pool (optional per-track parallelism; disjoint).
  {
    auto block = [&strack_pool](size_t begin, size_t end) {
      for (size_t i = begin; i < end; ++i) track_predict(*strack_pool[i]);
    };
    if (im.pool) {
      if (im.pool->run(strack_pool.size(), 64, block))
        ++im.parallel_engaged;
      else
        ++im.parallel_inline;
    } else {
      block(0, strack_pool.size());
    }
  }

  // ---- First association: high-score detections with fused costs.
  im.iou_distance(strack_pool, detections_list, im.dists);
  if (!im.cfg.mot20) im.fuse_score(im.dists, detections_list);
  Assignment a1 = linear_assignment(im.dists, im.cfg.match_thresh);
  for (const auto& [itracked, idet] : a1.matches) {
    STrack* track = strack_pool[itracked];
    STrack* det = detections_list[idet];
    if (track->state == TState::Tracked) {
      track->do_update(det, fid);
      activated_starcks.push_back(track);
    } else {
      track->do_re_activate(det, fid);
      refind_stracks.push_back(track);
    }
  }

  // ---- Second association: low-score detections, IoU cost only, thresh 0.5.
  std::vector<STrack*> r_tracked_stracks;
  r_tracked_stracks.reserve(a1.unmatched_a.size());
  for (int i : a1.unmatched_a)
    if (strack_pool[i]->state == TState::Tracked) r_tracked_stracks.push_back(strack_pool[i]);
  im.iou_distance(r_tracked_stracks, detections_second, im.dists);
  Assignment a2 = linear_assignment(im.dists, 0.5);
  for (const auto& [itracked, idet] : a2.matches) {
    STrack* track = r_tracked_stracks[itracked];
    STrack* det = detections_second[idet];
    if (track->state == TState::Tracked) {
      track->do_update(det, fid);
      activated_starcks.push_back(track);
    } else {
      track->do_re_activate(det, fid);
      refind_stracks.push_back(track);
    }
  }
  for (int it : a2.unmatched_a) {
    STrack* track = r_tracked_stracks[it];
    if (track->state != TState::Lost) {
      track->state = TState::Lost;
      lost_now.push_back(track);
    }
  }

  // ---- Unconfirmed tracks vs detections left over from the first stage.
  std::vector<STrack*> detections_left;
  detections_left.reserve(a1.unmatched_b.size());
  for (int i : a1.unmatched_b) detections_left.push_back(detections_list[i]);
  im.iou_distance(unconfirmed, detections_left, im.dists);
  if (!im.cfg.mot20) im.fuse_score(im.dists, detections_left);
  Assignment a3 = linear_assignment(im.dists, 0.7);
  for (const auto& [itracked, idet] : a3.matches) {
    unconfirmed[itracked]->do_update(detections_left[idet], fid);
    activated_starcks.push_back(unconfirmed[itracked]);
  }
  for (int it : a3.unmatched_a) {
    STrack* track = unconfirmed[it];
    track->state = TState::Removed;
    removed_now.push_back(track);
  }

  // ---- New tracks from detections still unmatched after the unconfirmed
  // stage (u_detection is reassigned there; score >= det_thresh activates).
  for (int inew : a3.unmatched_b) {
    STrack* track = detections_left[inew];
    if (static_cast<float>(track->score) < im.det_thresh_f32) continue;
    track->do_activate(im.next_id, fid);
    activated_starcks.push_back(track);
  }

  // ---- Lost tracks past the buffer are removed (end_frame == frame_id).
  for (STrack* track : im.lost_stracks) {
    if (fid - track->frame_id > im.max_time_lost) {
      track->state = TState::Removed;
      removed_now.push_back(track);
    }
  }

  // ---- Rebuild the lists in the exact Python order: sub(lost, tracked)
  // before extend(lost_now), sub(lost, removed) before extend(removed_now)
  // — a track marked removed this frame stays in lost_stracks one more
  // update, exactly like the baseline.
  std::vector<STrack*> tracked;
  tracked.reserve(im.tracked_stracks.size() + activated_starcks.size() + refind_stracks.size());
  for (STrack* t : im.tracked_stracks)
    if (t->state == TState::Tracked) tracked.push_back(t);
  {
    std::vector<STrack*> j1, j2;
    joint_stracks(tracked, activated_starcks, j1);
    joint_stracks(j1, refind_stracks, j2);
    tracked.swap(j2);
  }
  {
    std::vector<STrack*> s1, s2;
    sub_stracks(im.lost_stracks, tracked, s1);
    s1.insert(s1.end(), lost_now.begin(), lost_now.end());
    sub_stracks(s1, im.removed_stracks, s2);
    im.lost_stracks.swap(s2);
    im.removed_stracks.insert(im.removed_stracks.end(), removed_now.begin(), removed_now.end());
  }
  im.tracked_stracks.swap(tracked);
  remove_duplicate_stracks(im.tracked_stracks, im.lost_stracks);

  std::vector<Track> out;
  out.reserve(im.tracked_stracks.size());
  for (STrack* t : im.tracked_stracks) {
    if (!t->is_activated) continue;
    Track tk;
    tk.track_id = t->track_id;
    t->get_tlbr(tk.tlbr);
    tk.score = t->score;
    tk.frame_id = fid;
    out.push_back(tk);
  }
  im.sweep_frame_temporaries();
  return out;
}

#if defined(BYTETRACK_TARGET_X5) || defined(BYTETRACK_TARGET_S)

struct ByteTrackModel::ModelImpl {
  std::unique_ptr<yolov5::Yolov5> detector;
  std::unique_ptr<ByteTracker> tracker;
  int model_size = 0;

  /// Decode + original-image mapping + person filter: the single source of
  /// the detection list both postprocess (tracker feed) and detect
  /// (evidence dump) return — structurally the same values by construction.
  std::vector<Detection> decode_persons(const ByteTrackModel::Raw& raw,
                                        const ByteTrackModel::Prepared& prepared) const {
    const yolov5::Yolov5::Result result = detector->postprocess(raw.data);
    const std::vector<yolov5::Detection> orig =
        yolov5::map_to_original(result.detections, prepared.data.source_cols,
                                prepared.data.source_rows, model_size);
    // Python postprocess person filter: class 0 and strictly positive area.
    std::vector<Detection> persons;
    persons.reserve(orig.size());
    for (const yolov5::Detection& d : orig) {
      if (d.class_id != 0) continue;
      if (!(d.x2 > d.x1 && d.y2 > d.y1)) continue;
      persons.push_back(Detection{d.x1, d.y1, d.x2, d.y2, d.score});
    }
    return persons;
  }
};

ByteTrackModel::ByteTrackModel(const Options& options)
    : model_impl_(std::make_unique<ModelImpl>()) {
  // The detector gate requires cfg.target to equal the compiled build
  // identity, which CMake derives from BYTETRACK_TARGET (x5/s100/s100p/s600).
  yolov5::Yolov5::Config cfg;
  cfg.target = BYTETRACK_TARGET_NAME;
  cfg.model_path = options.model_path;
  cfg.score_threshold = options.score_thres;
  cfg.nms_threshold = options.nms_thres;
  cfg.priority = options.bpu_priority;
  cfg.bpu_core = options.bpu_core;
  // Partial construction cleans itself up: ModelImpl's unique_ptrs release
  // whatever was already built when a throwing stage unwinds.
  model_impl_->detector = std::make_unique<yolov5::Yolov5>(cfg);
  model_impl_->tracker = std::make_unique<ByteTracker>(options.tracking, options.threads);
  model_impl_->model_size = model_impl_->detector->input_size();
}

ByteTrackModel::~ByteTrackModel() = default;

ByteTrackModel::Prepared ByteTrackModel::preprocess(const Frame& frame) const {
  Prepared p;
  p.data = model_impl_->detector->preprocess(
      yolov5::Yolov5::Input{frame.bgr, frame.cols, frame.rows});
  return p;
}

ByteTrackModel::Raw ByteTrackModel::infer(const Prepared& prepared) const {
  Raw r;
  r.data = model_impl_->detector->infer(prepared.data);
  return r;
}

std::vector<Track> ByteTrackModel::postprocess(const Raw& raw, const Prepared& prepared) {
  return model_impl_->tracker->update(model_impl_->decode_persons(raw, prepared),
                                      prepared.data.source_rows,
                                      prepared.data.source_cols);
}

std::vector<Detection> ByteTrackModel::detect(const Frame& frame) {
  Prepared prepared = preprocess(frame);
  Raw raw = infer(prepared);
  return model_impl_->decode_persons(raw, prepared);
}

std::vector<Track> ByteTrackModel::predict(const Frame& frame) {
  Prepared prepared = preprocess(frame);
  Raw raw = infer(prepared);
  return postprocess(raw, prepared);
}

int ByteTrackModel::frame_id() const { return model_impl_->tracker->frame_id(); }

void ByteTrackModel::reset() { model_impl_->tracker->reset(); }

#endif  // board targets

}  // namespace bytetrack
