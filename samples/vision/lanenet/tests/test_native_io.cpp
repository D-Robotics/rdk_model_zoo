#include "cli_io.hpp"
#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
using namespace lanenet;
template <class F> void rejects(F f) {
  bool bad = false;
  try {
    f();
  } catch (const std::exception &) {
    bad = true;
  }
  assert(bad);
}
int main(int argc, char **argv) {
  assert(argc == 2);
  const std::string root = argv[1];
  auto options = parse_options({"--model_path=a.hbm", "--test_img", "a.jpg",
                                "--output", root + "/out",
                                "--instance_save_path", root + "/extra.png"});
  assert(options.model_path == "a.hbm" && options.target == "s100");
  validate_output_paths(options);
  rejects([] {
    parse_options(
        {"--model-path", "a", "--test-img", "b", "--target", "s100p"});
  });
  auto bad = options;
  bad.binary_path = options.instance_path;
  rejects([&] { validate_output_paths(bad); });
  bad = options;
  bad.instance_path = options.output_directory + "/binary.npy";
  rejects([&] { validate_output_paths(bad); });
  std::vector<std::int64_t> integers = {0, 1, 9007199254740993LL, -2};
  std::vector<unsigned char> bytes(integers.size() * 8);
  std::memcpy(bytes.data(), integers.data(), bytes.size());
  write_npy(root + "/integers.npy", ScalarType::Int64, {2, 2}, bytes);
  std::vector<float> floats = {-.5f, .1f, 1.2f};
  bytes.resize(floats.size() * 4);
  std::memcpy(bytes.data(), floats.data(), bytes.size());
  write_npy(root + "/floats.npy", ScalarType::Float32, {3}, bytes);
  write_npy(root + "/labels.npy", ScalarType::UInt8, {1, 2}, {0, 1});
  rejects(
      [&] { write_npy(root + "/bad.npy", ScalarType::Int64, {2, 3}, bytes); });
  std::ofstream(root + "/escaped.json")
      << "{\"value\":" << json_quote("a\"\\\n\t") << "}";
}
