#include "tensor_contract.hpp"
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
using namespace lanenet;
template <class F> void rejects(F f) {
  bool threw = false;
  try {
    f();
  } catch (const std::exception &) {
    threw = true;
  }
  assert(threw);
}
TensorSpec spec(ScalarType type, std::vector<std::size_t> shape,
                std::size_t item) {
  TensorSpec s{type, shape, std::vector<std::size_t>(shape.size()), 0};
  std::size_t stride = item;
  for (std::size_t i = shape.size(); i-- > 0;) {
    s.strides[i] = stride;
    stride *= shape[i];
  }
  s.capacity = stride;
  return s;
}
int main() {
  auto embedding = spec(ScalarType::Float32, {1, 3, 256, 512}, 4);
  // Non-contiguous W as well as row/channel padding. Padding must not leak.
  embedding.strides = {3 * 256 * 4104, 256 * 4104, 4104, 8};
  embedding.capacity = embedding.strides[0];
  auto binary = spec(ScalarType::Int64, {1, 1, 256, 512}, 8);
  auto aux = spec(ScalarType::Float32, {1, 2, 256, 512}, 4);
  auto roles = bind_roles({aux, binary, embedding});
  assert(roles.embedding == 2 && roles.binary == 1);
  rejects([&] { bind_roles({embedding, embedding, binary}); });
  rejects([&] { bind_roles({embedding, aux}); });
  auto bad = embedding;
  bad.strides[2] = 4;
  rejects([&] { validate_layout(bad); });
  bad = embedding;
  bad.capacity = 10;
  rejects([&] { validate_layout(bad); });
  bad = embedding;
  bad.strides[0] = std::numeric_limits<std::size_t>::max();
  rejects([&] { validate_layout(bad); });
  RawTensor e{embedding, std::vector<unsigned char>(embedding.capacity, 0xa5)},
      b{binary, std::vector<unsigned char>(binary.capacity, 0)};
  for (std::size_t c = 0; c < 3; c++)
    for (std::size_t h = 0; h < 256; h++)
      for (std::size_t w = 0; w < 512; w++) {
        float v = float(c) + float(w) / 511 - 1;
        std::memcpy(e.bytes.data() + c * embedding.strides[1] +
                        h * embedding.strides[2] + w * embedding.strides[3],
                    &v, 4);
        std::int64_t label = w % 2;
        std::memcpy(b.bytes.data() + h * binary.strides[2] + w * 8, &label, 8);
      }
  auto result = decode_outputs({e, b});
  assert(result.embedding.size() == 3 * 256 * 512 &&
         result.binary.size() == 256 * 512);
  assert(result.embedding[0] == -1 && result.embedding.back() == 2 &&
         result.binary[1] == 1);
  std::int64_t invalid = 2;
  std::memcpy(b.bytes.data(), &invalid, 8);
  rejects([&] { decode_outputs({e, b}); });
  invalid = 0;
  std::memcpy(b.bytes.data(), &invalid, 8);
  float nan = std::numeric_limits<float>::quiet_NaN();
  std::memcpy(e.bytes.data(), &nan, 4);
  rejects([&] { decode_outputs({e, b}); });
  assert(display_component(-.1f) == 0 && display_component(.1f) == 26 &&
         display_component(.5f) == 128 && display_component(1.2f) == 255);
  auto input = spec(ScalarType::Float32, {1, 3, 256, 512}, 4);
  input.strides = {3 * 256 * 2056, 256 * 2056, 2056, 4};
  input.capacity = input.strides[0];
  std::vector<unsigned char> bytes(input.capacity, 0xff);
  std::vector<float> values(3 * 256 * 512, 1.25f);
  write_input(values, input, bytes.data());
  auto compact = compact_bytes(RawTensor{input, bytes});
  float v = 0;
  std::memcpy(&v, compact.data() + 4 * 512, 4);
  assert(v == 1.25f);
  assert(bytes[2048] == 0);
  values.pop_back();
  rejects([&] { write_input(values, input, bytes.data()); });
  assert(is_s100(" S100\n", "RDK S100"));
  assert(!is_s100("s100", "RDK S100P"));
  assert(!is_s100("s100", "s100p"));
  assert(!is_s100("unknown", ""));
  assert(!is_s100("s600", ""));
}
