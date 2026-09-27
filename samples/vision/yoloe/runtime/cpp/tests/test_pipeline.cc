// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "yoloe.h"
#include <stdexcept>
#define EXPECT(v)                                                              \
  do {                                                                         \
    if (!(v))                                                                  \
      throw std::runtime_error(#v);                                            \
  } while (0)
template <class F> void rejects(F fn) {
  bool bad = false;
  try {
    fn();
  } catch (const std::invalid_argument &) {
    bad = true;
  }
  EXPECT(bad);
}
int destroyed = 0;
class FixtureRunner : public yoloe::Runner {
public:
  explicit FixtureRunner(yoloe::Protocol p) : kind(p) {}
  ~FixtureRunner() override { ++destroyed; }
  yoloe::Protocol protocol() const override { return kind; }
  yoloe::Heads infer(const yoloe::Nv12Input &input) override {
    ++calls;
    EXPECT(input.y.size() == 640 * 640);
    EXPECT(input.uv.size() == 320 * 640);
    if (fail)
      throw std::runtime_error("injected inference failure");
    yoloe::Heads heads;
    for (int scale = 0; scale < 3; ++scale) {
      int grid = 80 >> scale;
      heads[scale * 3].assign(grid * grid * 4585, -100);
      heads[scale * 3 + 1].assign(
          grid * grid * (kind == yoloe::Protocol::E11 ? 64 : 4), 1);
      heads[scale * 3 + 2].assign(grid * grid * 32, 0);
    }
    heads[9].assign(160 * 160 * 32, 0);
    heads[0][(30 * 80 + 30) * 4585 + 7] = 2;
    if (malformed)
      heads[9].pop_back();
    return heads;
  }
  yoloe::Protocol kind;
  int calls = 0;
  bool fail = false, malformed = false;
};
int main() {
  for (auto protocol : {yoloe::Protocol::E11, yoloe::Protocol::E26}) {
    auto owned = std::make_unique<FixtureRunner>(protocol);
    auto *fixture = owned.get();
    yoloe::Config config;
    config.protocol = protocol;
    yoloe::YOLOE task(config, std::move(owned));
    cv::Mat image(333, 1000, CV_8UC3, cv::Scalar(0, 0, 255));
    auto prepared = task.pre_process(image);
    EXPECT(fixture->calls == 0);
    auto raw = task.infer(prepared);
    EXPECT(fixture->calls == 1);
    auto result = task.post_process(raw);
    EXPECT(fixture->calls == 1);
    EXPECT(result.size() == 1);
    EXPECT(result[0].label == 7);
    EXPECT(result[0].mask.type() == CV_8UC1);
    auto alternate =
        task.pre_process(cv::Mat(517, 311, CV_8UC3, cv::Scalar(0)));
    EXPECT(alternate.geometry().width == 311);
    EXPECT(raw.geometry().width == 1000 && raw.geometry().height == 333);
    auto whole = task.predict(image);
    EXPECT(fixture->calls == 2);
    EXPECT(whole[0].box == result[0].box);
    EXPECT(cv::countNonZero(whole[0].mask != result[0].mask) == 0);
    auto saved_y = prepared.input().y;
    image.setTo(0);
    EXPECT(prepared.input().y == saved_y);
    yoloe::YOLOE other(config, std::make_unique<FixtureRunner>(protocol));
    rejects([&] { other.infer(prepared); });
    rejects([&] { other.post_process(raw); });
    fixture->fail = true;
    bool failed = false;
    try {
      task.infer(prepared);
    } catch (const std::runtime_error &) {
      failed = true;
    }
    EXPECT(failed);
    fixture->fail = false;
    fixture->malformed = true;
    auto bad = task.infer(prepared);
    rejects([&] { task.post_process(bad); });
    rejects([&] { task.pre_process(cv::Mat()); });
  }
  int before_invalid = destroyed;
  yoloe::Config invalid;
  invalid.protocol = yoloe::Protocol::E26;
  invalid.nms_threshold = 0.7f;
  rejects([&] {
    yoloe::YOLOE task(invalid,
                      std::make_unique<FixtureRunner>(invalid.protocol));
  });
  EXPECT(destroyed == before_invalid + 1);
  rejects([&] { yoloe::YOLOE task(yoloe::Config{}, nullptr); });
  rejects([&] {
    yoloe::YOLOE task(yoloe::Config{},
                      std::make_unique<FixtureRunner>(yoloe::Protocol::E26));
  });
  auto pixels = yoloe::prepare_bgr(
      cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 255)), yoloe::Protocol::E26);
  auto nv12 = yoloe::to_nv12(pixels.pixels);
  EXPECT(nv12.y[0] == 82);
  EXPECT(nv12.uv[0] == 90 && nv12.uv[1] == 240);
}
