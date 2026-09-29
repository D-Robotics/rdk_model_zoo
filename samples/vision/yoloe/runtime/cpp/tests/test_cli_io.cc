// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "cli_io.h"
#include "sha256.h"
#include <filesystem>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#define expect(v)                                                              \
  do {                                                                         \
    if (!(v))                                                                  \
      throw std::runtime_error(#v);                                            \
  } while (0)
template <class F> void rejects(F fn) {
  bool bad = false;
  try {
    fn();
  } catch (const std::exception &) {
    bad = true;
  }
  expect(bad);
}
yoloe::CliOptions options(std::vector<std::string> values) {
  std::vector<char *> argv;
  for (auto &s : values)
    argv.push_back(&s[0]);
  return yoloe::parse_cli(argv.size(), argv.data());
}
int main(int argc, char **argv) {
  expect(argc == 3);
  std::filesystem::path output = argv[1];
  std::string labels = argv[2];
  expect(!std::filesystem::exists(output));
  std::filesystem::create_directories(output);
  auto image = (output / "source.png").string();
  expect(cv::imwrite(image, cv::Mat(16, 20, CV_8UC3, cv::Scalar(20, 40, 60))));
  std::vector<std::string> args = {"demo",
                                   "--target",
                                   "s100p",
                                   "--variant",
                                   "26n",
                                   "--model-path",
                                   "model.hbm",
                                   "--model-sha256",
                                   std::string(64, 'a'),
                                   "--test-img",
                                   image,
                                   "--label-file",
                                   labels,
                                   "--output",
                                   (output / "result").string()};
  auto opts = options(args);
  expect(opts.config.protocol == yoloe::Protocol::E26 &&
         !opts.config.do_morph && opts.config.max_det == 300);
  auto bad = args;
  bad.insert(bad.end(), {"--nms-thres", "0.7"});
  rejects([&] { options(bad); });
  bad = args;
  bad.insert(bad.end(), {"--score-thres", "nan"});
  rejects([&] { options(bad); });
  bad = args;
  bad.insert(bad.end(), {"--max-det", "3x"});
  rejects([&] { options(bad); });
  bad = args;
  bad.insert(bad.end(), {"--target", "s100p"});
  rejects([&] { options(bad); });
  bad = args;
  bad.insert(bad.end(), {"--unknown", "x"});
  rejects([&] { options(bad); });
  bad = args;
  bad[2] = "s600";
  rejects([&] { options(bad); });
  bad = args;
  bad[2] = "s100";
  bad[4] = "11s";
  expect(options(bad).config.do_morph);
  bad.insert(bad.end(), {"--no-morph"});
  expect(!options(bad).config.do_morph);
  bad = args;
  bad[2] = "x5";
  bad[4] = "11s";
  expect(!options(bad).config.do_morph);
  auto inputs = yoloe::load_cli_inputs(opts);
  expect(inputs.image.rows == 16 && inputs.image.cols == 20 &&
         inputs.labels.size() == 4585);
  expect(inputs.image_sha256 == rdk::sha256_file(image));
  auto bad_options = opts;
  bad_options.label_path = image;
  rejects([&] { yoloe::load_cli_inputs(bad_options); });
  yoloe::create_output_directory(opts.output);
  rejects([&] { yoloe::create_output_directory(opts.output); });
  yoloe::Result result;
  result.push_back({{2.2f, 3.9f, 6.8f, 8.4f},
                    0.8f,
                    0,
                    cv::Mat(5, 4, CV_8UC1, cv::Scalar(1))});
  result.push_back({{9.f, 2.f, 9.f, 7.f}, 0.4f, 1, cv::Mat(5, 0, CV_8UC1)});
  yoloe::save_cli_outputs(opts, inputs, result);
  auto mask = cv::imread((output / "result/masks/000000.png").string(),
                         cv::IMREAD_GRAYSCALE);
  expect(mask.rows == 5 && mask.cols == 4 &&
         cv::countNonZero(mask != 255) == 0);
  expect(std::filesystem::exists(output / "result/report.json"));
  expect(std::filesystem::exists(output / "result/annotated.png"));
  expect(!std::filesystem::exists(output / "result/masks/000001.png"));
  rejects([&] { yoloe::save_cli_outputs(opts, inputs, result); });
}
