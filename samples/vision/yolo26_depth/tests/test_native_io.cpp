#include "cli_io.hpp"
#include <cassert>
#include <iostream>
#include <stdexcept>
using namespace yolo26_depth;
template <class F> void rejects(F f) {
  bool threw = false;
  try {
    f();
  } catch (const std::exception &) {
    threw = true;
  }
  assert(threw);
}
int main(int argc, char **argv) {
  assert(argc == 2);
  const auto old = parse_options({"a.bin", "b.jpg", "out"});
  assert(old.model_path == "a.bin" && old.warmup == 0);
  const auto modern =
      parse_options({"--target", "x5", "--model-path", "a.bin", "--test-img",
                     "b.jpg", "--output", "out", "--warmup", "3"});
  assert(modern.warmup == 3 && modern.target == "x5");
  assert(parse_options({"--help"}).help);
  rejects([] { parse_options({"--target", "s100"}); });
  rejects([] { parse_options({"--warmup", "-1"}); });
  rejects([] { parse_options({"--warmup", "2oops"}); });
  rejects([] { parse_options({"--unknown", "x"}); });
  rejects([] { parse_options({"--model-path"}); });
  const std::string prefix = argv[1];
  write_npy(prefix + ".npy", {1, 2, 3, 4, 5, 6}, 2, 3);
  write_f32(prefix + ".f32", {1, 2, 3, 4, 5, 6});
  rejects([&] { write_npy(prefix + "-bad.npy", {1}, 2, 3); });
  std::cout << "{\"quoted\":" << json_quote("path\"\\\n\t") << "}";
}
