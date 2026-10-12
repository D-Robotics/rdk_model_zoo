// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
/// Component-level parity goldens for the two numeric primitives whose exact
/// convention the full-track parity exercises only indirectly:
///   * bbox_iou — golden values produced by the baseline's cython_bbox
///     0.1.5 (bbox_overlaps, float64). The expression order matches cell for
///     cell, so equality must be bit-exact, not within a tolerance.
///   * linear_assignment — golden solutions produced by the baseline's
///     lap.lapjv(cost, extend_cost=True, cost_limit=0.8) on lap 0.5.12 (the
///     vendored solver is byte-identical in 0.5.13, the version installed on
///     the boards), mapped through the matching.py unpacking.
///
/// This TU compiles the production tracking.cpp directly to reach the
/// internal (anonymous-namespace) primitives; it must not also link
/// tracking.cpp, or the symbols would be duplicated.
#include "../../../runtime/cpp/src/tracking.cpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  if (!ok) {
    ++failures;
    std::printf("FAIL: %s\n", what.c_str());
  }
}

struct IouCase {
  const char* name;
  double a[4], b[4];
  double want;  // cython_bbox.bbox_overlaps output, float64
};

// Generated with the repo venv (cython_bbox 0.1.5, numpy 2.5.3) on 2026-10-12;
// the C helper's arithmetic is the same in every version (pure float64 C).
const IouCase kIouCases[] = {
    {"overlap", {100.0, 100.0, 160.0, 260.0}, {120.0, 110.0, 180.0, 270.0}, 0.460263177458925},
    {"identical", {37.5, 21.25, 240.75, 512.5}, {37.5, 21.25, 240.75, 512.5}, 1.0},
    {"disjoint", {10.0, 10.0, 60.0, 160.0}, {500.0, 500.0, 600.0, 700.0}, 0.0},
    // A one-pixel gap has iw = 0 under the inclusive +1 convention: no overlap.
    {"touch-edge", {10.0, 10.0, 60.0, 160.0}, {61.0, 10.0, 111.0, 160.0}, 0.0},
    {"contain", {0.0, 0.0, 1000.0, 800.0}, {100.0, 100.0, 200.0, 200.0}, 0.012722608228226205},
    {"fractional", {11.300000190734863, 5.5999999046325684, 77.199996948242188, 190.89999389648438},
     {13.100000381469727, 7.2999999523163099, 70.099998474121094, 185.69999694824219}, 0.8348558084921966},
    {"tiny", {0.0, 0.0, 1.0, 1.0}, {0.0, 0.0, 1.0, 2.0}, 0.6666666666666666},
    {"wide-thin", {0.0, 1079.0, 1919.0, 1080.0}, {100.0, 1075.0, 300.0, 1080.0}, 0.08656330749354005},
};

struct AssignCase {
  const char* name;
  int rows, cols;
  std::vector<double> cost;      // row-major
  double limit;
  std::vector<std::pair<int, int>> matches;
  std::vector<int> unmatched_a, unmatched_b;
};

// lap.lapjv(cost, extend_cost=True, cost_limit=0.8) on lap 0.5.12, then the
// matching.py unpacking (x[i] >= cols -> -1, y[j] >= rows -> -1, matches
// ordered by row ascending, unmatched ascending).
const AssignCase kAssignCases[] = {
    {"a-3x4", 3, 4,
     {0.75011455992560039, 1.0766565611634906, 0.93082282829423213, 0.27024862798871024,
      0.36019954189347053, 1.0482641344755141, 0.0063183654786896689, 0.98547410205931951,
      0.95648331450245538, 0.56152194341246486, 0.36363891218317623, 0.33411073452092799},
     0.8,
     {{0, 3}, {1, 2}, {2, 1}},
     {},
     {0}},
    {"b-4x3", 4, 3,
     {0.30584350518494952, 0.53409156705917582, 0.60545791074954391,
      0.66419682248939094, 1.1946003401212713, 0.95119430305650365,
      0.74661507532939519, 1.1867521772182619, 0.25837043788271874,
      0.19225444062941346, 0.73504752512763694, 0.05273040955366004},
     0.8,
     {{0, 1}, {2, 2}, {3, 0}},
     {1},
     {}},
    {"c-limit-avoidance", 2, 2,
     {0.10000000000000001, 0.90000000000000002, 0.84999999999999998, 0.050000000000000003},
     0.8,
     {{0, 0}, {1, 1}},
     {},
     {}},
};

std::string vec_str(const std::vector<int>& v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
  return s + "]";
}

void check_iou() {
  for (const IouCase& c : kIouCases) {
    const double got = bytetrack::bbox_iou(c.a, c.b);
    check(std::memcmp(&got, &c.want, sizeof got) == 0,
          std::string("iou ") + c.name + " bit-exact (got " + std::to_string(got) + ")");
    // Symmetry of the convention (same union expression both ways).
    const double rev = bytetrack::bbox_iou(c.b, c.a);
    check(std::memcmp(&got, &rev, sizeof got) == 0, std::string("iou ") + c.name + " symmetric");
  }
}

void check_assignment() {
  for (const AssignCase& c : kAssignCases) {
    bytetrack::Matrix cost;
    cost.reset(c.rows, c.cols);
    for (int i = 0; i < c.rows; ++i)
      for (int j = 0; j < c.cols; ++j) cost.at(i, j) = c.cost[i * c.cols + j];
    const bytetrack::Assignment got = bytetrack::linear_assignment(cost, c.limit);
    bool ok = got.matches == c.matches && got.unmatched_a == c.unmatched_a &&
              got.unmatched_b == c.unmatched_b;
    std::string detail;
    if (!ok) {
      detail = " (matches";
      for (const auto& m : got.matches) detail += " (" + std::to_string(m.first) + "," + std::to_string(m.second) + ")";
      detail += ") ua" + vec_str(got.unmatched_a) + " ub" + vec_str(got.unmatched_b);
    }
    check(ok, std::string("assignment ") + c.name + detail);
  }
  // Degenerate shapes take the explicit early return: everything unmatched.
  {
    bytetrack::Matrix empty;
    empty.reset(0, 3);
    const auto got = bytetrack::linear_assignment(empty, 0.8);
    check(got.matches.empty() && got.unmatched_a.empty() && got.unmatched_b.size() == 3,
          "assignment empty-rows early return");
    empty.reset(2, 0);
    const auto got2 = bytetrack::linear_assignment(empty, 0.5);
    check(got2.matches.empty() && got2.unmatched_a.size() == 2 && got2.unmatched_b.empty(),
          "assignment empty-cols early return");
  }
}

}  // namespace

int main() {
  check_iou();
  check_assignment();
  if (failures == 0) {
    std::printf("ALL COMPONENT CHECKS PASSED\n");
    return 0;
  }
  std::printf("%d COMPONENT CHECK(S) FAILED\n", failures);
  return 1;
}
