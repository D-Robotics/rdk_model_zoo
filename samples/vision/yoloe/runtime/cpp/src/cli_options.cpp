// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "cli_io.h"
#include "preflight.h"
#include <map>
#include <set>
namespace yoloe {
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
} // namespace yoloe
