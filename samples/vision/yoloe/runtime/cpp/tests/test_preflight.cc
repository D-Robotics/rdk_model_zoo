// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "preflight.h"
#include "sha256.h"
#include <fstream>
#include <iostream>
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
int main(int argc, char **argv) {
  expect(argc == 4);
  std::string model_path = argv[1], labels = argv[2], expected = argv[3];
  yoloe::SdkModel model{model_path, "s100p", "26n"};
  rdk::NativeIdentity actual{"S100\n", "RDK S100P\n", "", ""};
  yoloe::verify_preflight(model, expected, labels, actual);
  auto uppercase = expected;
  std::transform(uppercase.begin(), uppercase.end(), uppercase.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  yoloe::verify_preflight(model, uppercase, labels, actual);
  actual.board = "s100";
  rejects([&] { yoloe::verify_preflight(model, expected, labels, actual); });
  actual.board = "s100p";
  rejects([&] {
    yoloe::verify_preflight(model, std::string(64, '0'), labels, actual);
  });
  rejects([&] { yoloe::make_preflight("invalid", labels); });
  rejects(
      [&] { yoloe::verify_preflight(model, expected, model_path, actual); });
  rejects([&] { yoloe::verify_preflight(model, expected, labels, {}); });
  auto gate = yoloe::make_preflight(expected, labels);
  if (rdk::identify_target(rdk::read_native_identity()).empty())
    rejects([&] { gate(model); });
  expect(rdk::identify_target({"s100", "s100p", "", ""}) == "s100p");
  expect(rdk::identify_target({"s100", "RDK S100P", "", ""}) == "s100p");
  expect(rdk::identify_target({"s100p", "", "", ""}) == "s100p");
  expect(rdk::identify_target({"s600", "", "", ""}) == "s600");
  expect(rdk::identify_target({"", "", "X5U", ""}) == "x5");
  expect(rdk::identify_target({"", "", "x5h", ""}) == "x5");
  expect(rdk::identify_target({"", "", "x5m", ""}) == "x5");
  expect(rdk::identify_target({"", "", "", "D-Robotics RDK X5 V1.0"}) == "x5");
  expect(rdk::identify_target({"unknown", "", "x5u", "D-Robotics RDK X5 V1.0"})
             .empty());
  expect(rdk::identify_target({"", "", "unknown", "D-Robotics RDK X5 V1.0"})
             .empty());
  expect(rdk::identify_target({"", "", "", "d-robotics rdk x5 v1.0"}).empty());
  expect(rdk::sha256_hex("", 0) ==
         "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  expect(rdk::sha256_hex("abc", 3) ==
         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  std::string million(1000000, 'a');
  expect(rdk::sha256_hex(million.data(), million.size()) ==
         "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  std::cout << rdk::sha256_file(model_path) << '\n';
}
