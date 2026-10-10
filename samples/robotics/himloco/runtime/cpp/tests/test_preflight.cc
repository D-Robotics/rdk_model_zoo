// Production preflight with only the board identity reader replaced for host
// tests.
#include "platform_identity.h"
#include "policy.hpp"
#include <cassert>
#include <fstream>
#include <iostream>
#include <stdexcept>
namespace {
rdk::NativeIdentity identity;
}
namespace rdk {
NativeIdentity read_native_identity() { return identity; }
} // namespace rdk
int main(int argc, char **argv) {
  assert(argc == 2);
  auto rejects = [&](const std::string &path, const std::string &message) {
    bool failed = false;
    try {
      himloco::verify_native_model(path);
    } catch (const std::exception &e) {
      failed = std::string(e.what()).find(message) != std::string::npos;
    }
    assert(failed);
  };
  for (const auto &target : {"", "s100", "s100p", "s600"}) {
    identity = {target, "", "", ""};
    rejects("absent.bin", "requires actual X5");
  }
  identity = {"s100", "RDK S100P", "", ""};
  rejects("absent.bin", "requires actual X5");
  identity = {"x5", "", "", ""};
  rejects("absent.hbm", ".bin");
  rejects("absent.bin", "SHA-256");
  {
    std::ofstream file(argv[1]);
    file << "not a published model";
  }
  rejects(argv[1], "SHA-256");
  std::cout << "production preflight: board aliases, suffix, absent/wrong "
               "digest rejection passed\n";
}
