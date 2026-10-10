// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "asr.hpp"
#include <cassert>
#include <fstream>
#include <stdexcept>
#include <unistd.h>
template <class F> void rejects(F f) {
  bool caught = false;
  try {
    f();
  } catch (const std::exception &) {
    caught = true;
  }
  assert(caught);
}
int main(int argc, char **argv) {
  assert(argc == 2);
  char path[] = "/tmp/asr-preflight-XXXXXX";
  int fd = mkstemp(path);
  assert(fd >= 0);
  close(fd);
  {
    std::ofstream out(path);
    out << "abc";
  }
  const std::string sha =
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
  for (const auto &target : {"s100", "s600"})
    asr::verify_preflight({path, target}, sha, argv[1], {target, "", "", ""});
  rejects([&] {
    asr::verify_preflight({path, "s100"}, sha, argv[1],
                          {"s100", "RDK S100P", "", ""});
  });
  rejects([&] {
    asr::verify_preflight({path, "s100p"}, sha, argv[1], {"s100p", "", "", ""});
  });
  rejects([&] { asr::verify_preflight({path, "s100"}, sha, argv[1], {}); });
  rejects([&] {
    asr::verify_preflight({path, "s100"}, std::string(64, '0'), argv[1],
                          {"s100", "", "", ""});
  });
  rejects([&] {
    asr::verify_preflight({path, "s100"}, sha, path, {"s100", "", "", ""});
  });
  rejects([&] { asr::make_preflight("unknown", argv[1]); });
  std::remove(path);
  rejects([&] {
    asr::verify_preflight({path, "s100"}, sha, argv[1], {"s100", "", "", ""});
  });
}
