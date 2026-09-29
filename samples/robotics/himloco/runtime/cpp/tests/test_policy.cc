// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "policy.hpp"
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
template <class F> void Reject(F operation) {
  bool rejected = false;
  try { operation(); } catch (const std::invalid_argument &) { rejected = true; }
  assert(rejected);
}
}

int main(int argc, char **argv) {
  assert(argc == 2);
  int calls = 0;
  himloco::RawOutputs next{std::vector<float>(12, -0.0f), 1.25};
  himloco::HimLoco task([&](const std::vector<float> &values) {
    assert(values.size() == 270);
    ++calls;
    return next;
  });
  Reject([] { himloco::HimLoco invalid({}); });
  std::vector<float> observation(270);
  for (std::size_t i = 0; i < observation.size(); ++i)
    observation[i] = static_cast<float>(i) / 8;
  auto prepared = task.pre_process(observation);
  assert(calls == 0 && prepared.values == observation);
  observation[0] = 999;
  assert(prepared.values[0] == 0);
  auto raw = task.forward(prepared);
  assert(calls == 1 && raw.latency_ms == 1.25);
  next.actions[0] = 7;
  next.latency_ms = 5;
  auto later = task.forward(prepared);
  auto result = task.post_process(raw);
  assert(calls == 2 && result.latency_ms == 1.25);
  assert(later.actions[0] == 7 && later.latency_ms == 5);
  assert(std::signbit(result.actions[0]));
  raw.actions[0] = 42;
  assert(std::signbit(result.actions[0]));
  assert(task.predict(observation).actions == next.actions && calls == 3);
  std::cout << "owned stages, unchanged actions, interleaved timing: passed\n";

  for (std::size_t size : {0U, 269U, 271U}) {
    Reject([&] { task.pre_process(std::vector<float>(size)); });
    Reject([&] { task.forward({std::vector<float>(size)}); });
  }
  for (float value : {std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity()}) {
    auto bad = observation;
    bad[100] = value;
    Reject([&] { task.pre_process(bad); });
    Reject([&] { task.forward({bad}); });
  }
  assert(calls == 3);
  for (std::size_t size : {0U, 11U, 13U}) {
    next.actions.assign(size, 1);
    Reject([&] { task.forward(prepared); });
    Reject([&] { task.post_process(next); });
  }
  next.actions.assign(12, 1);
  for (double latency : {-1.0, std::numeric_limits<double>::infinity(),
                          std::numeric_limits<double>::quiet_NaN()}) {
    next.latency_ms = latency;
    Reject([&] { task.forward(prepared); });
    Reject([&] { task.post_process(next); });
  }
  next.latency_ms = 0;
  next.actions[0] = std::numeric_limits<float>::quiet_NaN();
  Reject([&] { task.forward(prepared); });
  Reject([&] { task.post_process(next); });
  std::cout << "invalid input, output and timing: passed\n";
  himloco::HimLoco failing([](const std::vector<float> &) -> himloco::RawOutputs {
    throw std::runtime_error("injected transport failure");
  });
  bool propagated = false;
  try { failing.predict(observation); }
  catch (const std::runtime_error &e) {
    propagated = std::string(e.what()) == "injected transport failure";
  }
  assert(propagated);
  std::cout << "runner failure propagation: passed\n";

  // Real source inputs; only identity preprocessing is checked, no model executes.
  for (int i = 0; i < 21; ++i) {
    const std::string stem = std::string(6 - std::to_string(i).size(), '0') + std::to_string(i);
    std::ifstream stream(std::string(argv[1]) + "/" + stem + ".bin", std::ios::binary);
    std::vector<float> values(270);
    stream.read(reinterpret_cast<char *>(values.data()), 1080);
    assert(stream.gcount() == 1080);
    auto input = task.pre_process(values);
    assert(std::memcmp(input.values.data(), values.data(), 1080) == 0);
  }
  std::cout << "21 source observations preserve all float bytes: passed\n";
}
