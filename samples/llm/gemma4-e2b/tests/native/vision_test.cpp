#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "gemma4_vision_engine.hpp"
#include "vision_fixture.hpp"

namespace gemma4 {
std::vector<float> SourcePreprocessImage(const std::string &path);
}

template <typename F> void rejects(F f) {
  bool rejected = false;
  try {
    f();
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);
}

int main(int argc, char **argv) {
  using namespace gemma4;
  if (argc == 2) {
    for (int i = 1; i <= 4; ++i) {
      const auto path =
          std::string(argv[1]) + "/image" + std::to_string(i) + ".jpg";
      const auto source = SourcePreprocessImage(path);
      const auto actual = PreprocessImage(LoadImage(path));
      assert(source.size() == 2520u * 768);
      assert(source.size() == actual.size());
      assert(std::memcmp(source.data(), actual.data(),
                         source.size() * sizeof(float)) == 0);
    }
    std::cout << "four source-image preprocess comparisons: byte identical\n";
    return 0;
  }
  const cv::Mat image(1, 1, CV_8UC3, cv::Scalar(0, 0, 255));
  const auto patches = PreprocessImage(image);
  assert(patches.size() == 2520u * 768);
  for (size_t i = 0; i < patches.size(); i += 3) {
    assert(patches[i] == 1.f && patches[i + 1] == 0.f && patches[i + 2] == 0.f);
  }
  // Pixel address probes distinguish patch order and channel interleaving.
  cv::Mat grid(672, 960, CV_8UC3, cv::Scalar(0, 0, 0));
  grid.at<cv::Vec3b>(16, 16) = cv::Vec3b(255, 0, 0);
  auto tiled = PreprocessImage(grid);
  assert(tiled[(60 + 1) * 768 + 2] == 1.f);
  assert(tiled[(60 + 1) * 768] == 0.f);
  // A noncontiguous ROI must behave like an owned clone and not mutate its
  // input.
  cv::Mat canvas(5, 9, CV_8UC3, cv::Scalar(10, 20, 30));
  cv::Mat roi = canvas(cv::Rect(1, 1, 4, 3));
  auto before = canvas.clone();
  assert(PreprocessImage(roi) == PreprocessImage(roi.clone()));
  assert(cv::norm(canvas, before, cv::NORM_INF) == 0);
  rejects([] { PreprocessImage(cv::Mat()); });
  rejects([] { PreprocessImage(cv::Mat(1, 1, CV_8UC1)); });
  rejects([] { PreprocessImage(cv::Mat(1, 1, CV_32FC3)); });

  int calls = 0;
  const VisionRunner runner = [&calls](const std::vector<float> &p) {
    ++calls;
    assert(p.size() == 2520u * 768);
    return std::vector<float>(280 * 1536, 0.25f);
  };
  auto output = PredictVision(image, runner);
  assert(calls == 1 && output.size() == 280u * 1536 && output[0] == 0.25f);
  auto raw = ForwardVision(patches, runner);
  auto owned = PostprocessVision(raw);
  raw[0] = 9.f;
  assert(owned[0] == 0.25f);
  rejects([&] { ForwardVision({}, runner); });
  rejects([&] { ForwardVision(patches, VisionRunner{}); });
  auto bad = patches;
  bad[0] = std::numeric_limits<float>::quiet_NaN();
  rejects([&] { ForwardVision(bad, runner); });
  bad[0] = 1.1f;
  rejects([&] { ForwardVision(bad, runner); });
  rejects([] { PostprocessVision(std::vector<float>(1, 1.f)); });
  raw[0] = std::numeric_limits<float>::infinity();
  rejects([&] { PostprocessVision(raw); });
  assert(calls == 2);         // rejected inputs never reach the injected runner
  assert(output[0] == 0.25f); // later calls do not change retained results
  bool unreadable = false;
  try {
    LoadImage("/nonexistent/gemma-image.jpg");
  } catch (const std::runtime_error &) {
    unreadable = true;
  }
  assert(unreadable);
  std::cout << "vision geometry, stages, ownership and invalid inputs pass\n";
}
