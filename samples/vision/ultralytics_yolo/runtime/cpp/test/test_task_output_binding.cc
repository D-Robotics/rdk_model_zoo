// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cstdlib>
#include <limits>

#include "common/task_output_binding.h"
#define expect(v)                           \
  do {                                      \
    if (!(v)) throw std::runtime_error(#v); \
  } while (0)
std::vector<hbDNNTensorProperties> metadata;
int allocated = 0, freed = 0, fail_alloc = -1, fail_flush = 0, fail_query = -1;
int hbDNNGetOutputCount(int32_t* n, hbDNNHandle_t) {
  *n = metadata.size();
  return 0;
}
int hbDNNGetOutputTensorProperties(hbDNNTensorProperties* p, hbDNNHandle_t,
                                   int i) {
  if (i == fail_query) return -1;
  *p = metadata.at(i);
  return 0;
}
int fake_allocate(TestMemory* m, int n) {
  if (allocated == fail_alloc) return -1;
  m->virAddr = std::calloc(1, n);
  ++allocated;
  return m->virAddr ? 0 : -1;
}
int fake_free(TestMemory* m) {
  std::free(m->virAddr);
  ++freed;
  return 0;
}
int fake_flush(TestMemory*, int) { return fail_flush; }
template <class F>
void rejects(F fn) {
  bool bad = false;
  try {
    fn();
  } catch (const std::exception&) {
    bad = true;
  }
  expect(bad);
}
hbDNNTensorProperties descriptor(int h, int w, int c) {
  hbDNNTensorProperties p{};
  p.tensorType = HB_DNN_TENSOR_TYPE_F32;
  p.quantiType = NONE;
  p.tensorLayout = HB_DNN_LAYOUT_NHWC;
  p.validShape = {4, {1, h, w, c}};
  p.alignedShape = {4, {1, h, w, c + 1}};
  p.stride[3] = 4;
  p.stride[2] = (c + 1) * 4;
  p.stride[1] = w * p.stride[2];
  p.stride[0] = h * p.stride[1];
  p.alignedByteSize = p.stride[0];
  return p;
}
void setup(bool segment) {
  metadata.clear();
  allocated = freed = 0;
  fail_alloc = -1;
  fail_query = -1;
  fail_flush = 0;
  for (int stride : {8, 16, 32})
    for (int c : {segment ? 80 : 1, 4, segment ? 32 : 51})
      metadata.push_back(descriptor(64 / stride, 64 / stride, c));
  if (segment) metadata.push_back(descriptor(16, 16, 32));
  std::reverse(metadata.begin(), metadata.end());
}
int main() {
  for (bool segment : {false, true}) {
    setup(segment);
    {
      yolo::TaskOutputs outputs;
      outputs.bind(nullptr, 64, 64, segment);
      outputs.allocate();
      auto copied = outputs.read();
      expect(copied.size() == metadata.size());
      expect(copied[outputs.heads.box[0]].size() == 8 * 8 * 4);
      fail_flush = -1;
      rejects([&] { outputs.read(); });
      fail_flush = 0;
      float* raw =
          static_cast<float*>(YOLO_SYS_MEM(outputs.tensors()[0])->virAddr);
      raw[0] = std::numeric_limits<float>::quiet_NaN();
      rejects([&] { outputs.read(); });
      // views() exposes the padded physical layout without copying or
      // prescanning; decoders check the values they consume.
      auto views = outputs.views();
      expect(views.size() == metadata.size());
      expect(views[0].data == raw);
      const yolo::TensorView& box = views[outputs.heads.box[0]];
      expect(box.h == 8 && box.w == 8 && box.channels == 4);
      expect(box.cell_step == 5 && box.row_step == 8 * 5);
      fail_flush = -1;
      rejects([&] { outputs.views(); });
      fail_flush = 0;
    }
    expect(allocated == static_cast<int>(metadata.size()) &&
           freed == allocated);
    setup(segment);
    fail_alloc = 3;
    rejects([&] {
      yolo::TaskOutputs outputs;
      outputs.bind(nullptr, 64, 64, segment);
      outputs.allocate();
    });
    expect(allocated == 3 && freed == 3);
    setup(segment);
    fail_query = 2;
    rejects([&] {
      yolo::TaskOutputs outputs;
      outputs.bind(nullptr, 64, 64, segment);
    });
    expect(allocated == 0 && freed == 0);
    setup(segment);
    metadata[0].quantiType = 1;
    rejects([&] {
      yolo::TaskOutputs outputs;
      outputs.bind(nullptr, 64, 64, segment);
    });
    expect(allocated == 0);
    setup(segment);
    metadata[0].validShape.dimensionSize[0] = 2;
    rejects([&] {
      yolo::TaskOutputs outputs;
      outputs.bind(nullptr, 64, 64, segment);
    });
  }
  setup(false);
  metadata[0] = metadata[1];
  yolo::TaskOutputs invalid;
  rejects([&] { invalid.read(); });
  rejects([&] { invalid.views(); });
  rejects([&] { invalid.bind(nullptr, 64, 64, false); });
  rejects([&] { invalid.allocate(); });
  expect(allocated == 0 && freed == 0);
}
