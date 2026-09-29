// Calls the production identity reader; no identity or SDK doubles.
#include "preflight.h"
#include <iostream>
int main() {
  const auto actual = rdk::identify_target(rdk::read_native_identity());
  if (!actual.empty()) {
    std::cerr << "This negative host probe requires an unrecognized host\n";
    return 3;
  }
  try {
    paraformer::make_preflight({}, "not-loaded.json");
  } catch (const std::invalid_argument &error) {
    const std::string message = error.what();
    std::cout << message << '\n';
    return message == "Paraformer requires actual local S100 identity" ? 0 : 4;
  }
  return 5;
}
