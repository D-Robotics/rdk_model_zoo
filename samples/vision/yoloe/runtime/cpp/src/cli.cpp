// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0

// CLI of the YOLOE native runtime: strict kebab-case option parsing with
// required model/identity arguments, byte-exact input loading (image and the
// fixed 4585-class vocabulary), the fresh-output-directory policy and the
// annotated report/mask writer.

#include "cli.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "sha256.h"

namespace yoloe {
namespace fs = std::filesystem;
namespace {
float real(const std::string &text) {
  size_t end = 0;
  float value = std::stof(text, &end);
  if (end != text.size())
    throw std::invalid_argument("Invalid real value: " + text);
  return value;
}
int integer(const std::string &text) {
  size_t end = 0;
  int value = std::stoi(text, &end);
  if (end != text.size())
    throw std::invalid_argument("Invalid integer value: " + text);
  return value;
}
} // namespace
CliOptions parse_cli(int argc, char **argv) {
  CliOptions options;
  for (int i = 0; i < argc; ++i)
    options.argv.emplace_back(argv[i]);
  if (argc == 2 && options.argv[1] == "--help") {
    options.help = true;
    return options;
  }
  const std::set<std::string> flags{"--no-morph", "--no-contour",
                                    "--multi-label"};
  const std::set<std::string> values{
      "--target",    "--variant",     "--model-path", "--model-sha256",
      "--test-img",  "--label-file",  "--output",     "--score-thres",
      "--nms-thres", "--resize-type", "--max-det"};
  std::map<std::string, std::string> args;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (args.count(key))
      throw std::invalid_argument("Duplicate argument: " + key);
    if (flags.count(key)) {
      args[key] = "true";
      continue;
    }
    if (!values.count(key) || i + 1 >= argc)
      throw std::invalid_argument("Unknown or incomplete argument: " + key);
    args[key] = argv[++i];
  }
  for (const auto &key :
       {"--target", "--variant", "--model-path", "--model-sha256", "--test-img",
        "--label-file", "--output"})
    if (!args.count(key) || args[key].empty())
      throw std::invalid_argument(std::string("Required argument: ") + key);
  options.model = {args["--model-path"], args["--target"], args["--variant"]};
  if (!supported_native_model(options.model))
    throw std::invalid_argument("Unsupported YOLOE target/variant pair");
  options.model_sha256 = args["--model-sha256"];
  options.image_path = args["--test-img"];
  options.label_path = args["--label-file"];
  options.output = args["--output"];
  (void)make_preflight(options.model_sha256,
                       options.label_path); // validates digest format only
  std::transform(options.model_sha256.begin(), options.model_sha256.end(),
                 options.model_sha256.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  options.config.protocol =
      options.model.variant.rfind("26", 0) == 0 ? Protocol::E26 : Protocol::E11;
  options.config.do_morph = options.config.protocol == Protocol::E11 &&
                            options.model.target != "x5" &&
                            !args.count("--no-morph");
  options.contours = !args.count("--no-contour");
  options.config.single_label = !args.count("--multi-label");
  if (args.count("--score-thres"))
    options.config.score_threshold = real(args["--score-thres"]);
  if (args.count("--nms-thres"))
    options.config.nms_threshold = real(args["--nms-thres"]);
  if (args.count("--resize-type"))
    options.config.resize_type = integer(args["--resize-type"]);
  if (args.count("--max-det"))
    options.config.max_det = integer(args["--max-det"]);
  validate_config(options.config);
  return options;
}
std::string cli_help() {
  return R"(YOLOE native FLOAT32 PF segmentation
Required: --target x5|s100|s100p --variant 11s|11m|11l|26n|26s|26m|26l|26x
  --model-path FILE --model-sha256 HEX64 --test-img FILE --label-file FILE --output NEW_DIR
Not every target/variant pair is supported. S models need separately converted float outputs.
Options: --score-thres 0.25 --nms-thres 0.7 (E11 only) --resize-type 1
  --max-det 300 --multi-label (E26 only) --no-morph --no-contour --help
S E11 CLI enables 5x5 opening by default; the library default is off.
Output: report.json, annotated.png, masks/*.png (0/255 ROI masks; empty ROI has no PNG).
No downloads or board identity override. Use launcher.py for publication selection and run logs.
)";
}
namespace {
std::string bytes(const std::string &path) {
  if (!fs::is_regular_file(path))
    throw std::invalid_argument("Input is not a regular file: " + path);
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::invalid_argument("Cannot open input: " + path);
  std::string data(std::istreambuf_iterator<char>(file), {});
  if (file.bad() || data.empty())
    throw std::invalid_argument("Empty or unreadable input: " + path);
  return data;
}
std::string quote(const std::string &text) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : text) {
    switch (c) {
    case '"':
      out << "\\\"";
      break;
    case '\\':
      out << "\\\\";
      break;
    case '\n':
      out << "\\n";
      break;
    case '\r':
      out << "\\r";
      break;
    case '\t':
      out << "\\t";
      break;
    default:
      if (c < 32)
        out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
            << static_cast<int>(c) << std::dec;
      else
        out << static_cast<char>(c);
    }
  }
  out << '"';
  return out.str();
}
void write_image(const fs::path &path, const cv::Mat &image) {
  if (!cv::imwrite(path.string(), image))
    throw std::runtime_error("Cannot save image: " + path.string());
}
void validate_result(const Instance &item, const cv::Mat &image,
                     size_t classes) {
  if (item.label < 0 || static_cast<size_t>(item.label) >= classes ||
      !std::isfinite(item.score) || item.score < 0 || item.score > 1)
    throw std::invalid_argument("Invalid instance class/score");
  for (int i = 0; i < 4; ++i)
    if (!std::isfinite(item.box[i]) || item.box[i] < 0 ||
        double(item.box[i]) > (i % 2 ? image.rows : image.cols))
      throw std::invalid_argument("Invalid restored box");
  if (item.box[2] < item.box[0] || item.box[3] < item.box[1] ||
      item.mask.type() != CV_8UC1 ||
      item.mask.cols != int(item.box[2]) - int(item.box[0]) ||
      item.mask.rows != int(item.box[3]) - int(item.box[1]))
    throw std::invalid_argument("ROI mask geometry differs from box");
  for (int y = 0; y < item.mask.rows; ++y)
    for (int x = 0; x < item.mask.cols; ++x)
      if (item.mask.at<uint8_t>(y, x) > 1)
        throw std::invalid_argument("ROI masks must contain only 0/1");
}
} // namespace
CliInputs load_cli_inputs(const CliOptions &options) {
  auto encoded = bytes(options.image_path), labels = bytes(options.label_path);
  CliInputs result;
  result.image_sha256 = rdk::sha256_hex(encoded.data(), encoded.size());
  result.vocabulary_sha256 = rdk::sha256_hex(labels.data(), labels.size());
  if (result.vocabulary_sha256 != kVocabularySha256)
    throw std::invalid_argument("Vocabulary checksum mismatch");
  std::istringstream lines(labels);
  std::string line;
  while (std::getline(lines, line))
    result.labels.push_back(line);
  if (result.labels.size() != 4585)
    throw std::invalid_argument("Expected 4585 fixed vocabulary entries");
  result.image = cv::imdecode(
      std::vector<uint8_t>(encoded.begin(), encoded.end()), cv::IMREAD_COLOR);
  if (result.image.empty())
    throw std::invalid_argument("Cannot decode image: " + options.image_path);
  return result;
}
void create_output_directory(const std::string &name) {
  fs::path path(name);
  if (fs::exists(path) || fs::is_symlink(path))
    throw std::invalid_argument("Output directory must be new: " + name);
  if (!path.parent_path().empty())
    fs::create_directories(path.parent_path());
  if (!fs::create_directory(path))
    throw std::runtime_error("Cannot create output directory: " + name);
}
void save_cli_outputs(const CliOptions &options, const CliInputs &inputs,
                      const Result &result) {
  fs::path output(options.output);
  if (inputs.image.empty() || inputs.labels.size() != 4585 ||
      !fs::is_directory(output) || !fs::is_empty(output))
    throw std::invalid_argument(
        "Expected valid inputs and an empty new output directory");
  for (const auto &item : result)
    validate_result(item, inputs.image, inputs.labels.size());
  fs::create_directory(output / "masks");
  auto canvas = inputs.image.clone();
#if defined(YOLOE_HOST_FIXTURE)
  const char *backend = "host-fixture";
#else
  const char *backend = "native-sdk";
#endif
  std::ostringstream report;
  report << std::setprecision(std::numeric_limits<float>::max_digits10);
  report << "{\n  \"schema\": \"rdk-model-zoo/yoloe-native-run/v1\",\n  "
            "\"execution_backend\": "
         << quote(backend) << ",\n  \"target\": " << quote(options.model.target)
         << ",\n  \"variant\": " << quote(options.model.variant)
         << ",\n  \"model_path\": "
         << quote(fs::absolute(options.model.path).string())
         << ",\n  \"model_sha256\": " << quote(options.model_sha256)
         << ",\n  \"image_sha256\": " << quote(inputs.image_sha256)
         << ",\n  \"vocabulary_sha256\": " << quote(inputs.vocabulary_sha256)
         << ",\n  \"image_shape\": [" << inputs.image.rows << ", "
         << inputs.image.cols
         << ", 3],\n  \"mask_layout\": \"roi\",\n  \"mask_encoding\": "
            "\"png-uint8-0-or-255\",\n  \"image_saved\": \"annotated.png\",\n  "
            "\"config\": {\"score_threshold\": "
         << options.config.score_threshold << ", \"nms_threshold\": ";
  if (options.config.protocol == Protocol::E11)
    report << options.config.nms_threshold.value_or(0.7f);
  else
    report << "null";
  report << ", \"resize_type\": " << options.config.resize_type
         << ", \"do_morph\": " << (options.config.do_morph ? "true" : "false")
         << ", \"max_det\": " << options.config.max_det
         << ", \"single_label\": "
         << (options.config.single_label ? "true" : "false")
         << ", \"contours\": " << (options.contours ? "true" : "false")
         << "},\n  \"argv\": [";
  for (size_t i = 0; i < options.argv.size(); ++i) {
    if (i)
      report << ", ";
    report << quote(options.argv[i]);
  }
  report << "],\n  \"count\": " << result.size() << ",\n  \"instances\": [\n";
  for (size_t i = 0; i < result.size(); ++i) {
    const auto &item = result[i];
    int x = int(item.box[0]), y = int(item.box[1]);
    cv::Scalar color((37 * item.label + 50) % 256, (67 * item.label + 80) % 256,
                     (97 * item.label + 110) % 256);
    std::string mask_name;
    if (!item.mask.empty()) {
      std::ostringstream name;
      name << "masks/" << std::setw(6) << std::setfill('0') << i << ".png";
      mask_name = name.str();
      write_image(output / mask_name, item.mask * 255);
      auto view = canvas(cv::Rect(x, y, item.mask.cols, item.mask.rows));
      cv::Mat mixed;
      cv::addWeighted(view, 0.6, cv::Mat(view.size(), view.type(), color), 0.4,
                      0, mixed);
      mixed.copyTo(view, item.mask);
      if (options.contours) {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(item.mask.clone(), contours, cv::RETR_EXTERNAL,
                         cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(view, contours, -1, color, 1);
      }
    }
    cv::rectangle(canvas, cv::Point(x, y),
                  cv::Point(int(item.box[2]), int(item.box[3])), color, 2);
    cv::putText(canvas,
                inputs.labels[item.label] + " " +
                    cv::format("%.3f", item.score),
                cv::Point(x, std::max(15, y - 5)), cv::FONT_HERSHEY_SIMPLEX,
                0.5, color, 1);
    if (i)
      report << ",\n";
    report << "    {\"class_id\": " << item.label
           << ", \"label\": " << quote(inputs.labels[item.label])
           << ", \"score\": " << item.score << ", \"box\": [" << item.box[0]
           << ", " << item.box[1] << ", " << item.box[2] << ", " << item.box[3]
           << "], \"mask_shape\": [" << item.mask.rows << ", " << item.mask.cols
           << "], \"mask\": " << (mask_name.empty() ? "null" : quote(mask_name))
           << "}";
  }
  report << "\n  ]\n}\n";
  write_image(output / "annotated.png", canvas);
  auto temporary = output / "report.json.tmp";
  std::ofstream file(temporary, std::ios::binary);
  file << report.str();
  file.close();
  if (!file)
    throw std::runtime_error("Cannot write result report");
  fs::rename(temporary, output / "report.json");
}
} // namespace yoloe
