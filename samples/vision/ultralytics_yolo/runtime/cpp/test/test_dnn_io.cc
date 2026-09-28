// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "common/dnn_io.h"
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>
#define expect(v)                                                              \
  do {                                                                         \
    if (!(v))                                                                  \
      throw std::runtime_error(#v);                                            \
  } while (0)
std::vector<hbDNNTensorProperties> metadata;
int allocated = 0, freed = 0, fail_alloc = -1, fail_flush = 0, fail_query = -1;
int created = 0, waited = 0, released = 0, submitted = 0, create_rc = 0,
    wait_rc = 0, release_rc = 0, submit_rc = 0;
bool null_task = false;
int hbDNNGetInputCount(int32_t *n, hbDNNHandle_t) {
  *n = metadata.size();
  return 0;
}
int hbDNNGetInputTensorProperties(hbDNNTensorProperties *p, hbDNNHandle_t,
                                  int i) {
  if (i == fail_query)
    return -2;
  *p = metadata.at(i);
  return 0;
}
int test_allocate(TestMemory *m, int n) {
  if (allocated == fail_alloc)
    return -3;
  expect(n > 0);
  m->virAddr = std::malloc(n);
  m->memSize = n;
  expect(m->virAddr);
  ++allocated;
  return 0;
}
int test_free(TestMemory *m) {
  expect(m->virAddr);
  std::free(m->virAddr);
  m->virAddr = nullptr;
  ++freed;
  return 0;
}
int test_flush(TestMemory *m, int) {
  expect(m->virAddr);
  return fail_flush;
}
int test_create(void **t) {
  ++created;
  *t = null_task ? nullptr : reinterpret_cast<void *>(1);
  return create_rc;
}
int test_wait(void *t) {
  expect(t);
  ++waited;
  return wait_rc;
}
int test_release(void *t) {
  expect(t);
  ++released;
  return release_rc;
}
int test_submit(void *t, hbUCPSchedParam *s) {
  expect(t && s->backend == HB_UCP_BPU_CORE_ANY);
  ++submitted;
  return submit_rc;
}
void reset() {
  expect(allocated == freed);
  allocated = freed = 0;
  fail_alloc = fail_query = -1;
  fail_flush = 0;
  created = waited = released = submitted = create_rc = wait_rc = release_rc =
      submit_rc = 0;
  null_task = false;
  metadata.clear();
}
void split(bool dynamic = false) {
  reset();
  for (int i = 0; i < 2; ++i) {
    hbDNNTensorProperties p{};
#ifdef YOLO_DNN_STACK_X5
    p.tensorType = HB_DNN_TENSOR_TYPE_S8;
#else
    p.tensorType = HB_DNN_TENSOR_TYPE_U8;
#endif
    p.quantiType = NONE;
    p.tensorLayout = HB_DNN_LAYOUT_NHWC;
    int rows = i == 0 ? 4 : 2, cols = i == 0 ? 6 : 3, channels = i == 0 ? 1 : 2;
    p.validShape = {4, {1, rows, cols, channels}};
    p.alignedShape = p.validShape;
    p.stride[3] = 1;
    p.stride[2] = channels;
    p.stride[1] = 16;
    p.stride[0] = 16 * rows;
    p.alignedByteSize = 16 * rows;
    if (dynamic) {
      p.alignedByteSize = -1;
      for (int &s : p.stride)
        s = -1;
    }
    metadata.push_back(p);
  }
}
yolo::InputPlan plan() {
  std::string error;
  return yolo::probe_input_protocol(nullptr, &error);
}
void rejects_metadata() {
  expect(plan().protocol == yolo::InputProtocol::kUnknown);
  expect(allocated == freed);
}
void inference_failures() {
  hbDNNTensor inputs[2]{}, outputs[1]{};
  void *model = reinterpret_cast<void *>(2);
  reset();
  expect(yolo::infer_sync(outputs, inputs, 2, model) == 0);
  expect(created == 1 && waited == 1 && released == 1);
  reset();
  create_rc = -11;
  expect(yolo::infer_sync(outputs, inputs, 2, model) == -11);
  expect(waited == 0 && released == 1);
  reset();
  null_task = true;
  expect(yolo::infer_sync(outputs, inputs, 2, model) != 0);
  expect(waited == 0 && released == 0);
  reset();
  wait_rc = -12;
  release_rc = -13;
  expect(yolo::infer_sync(outputs, inputs, 2, model) == -12);
  expect(released == 1);
  reset();
  release_rc = -13;
  expect(yolo::infer_sync(outputs, inputs, 2, model) == -13);
#ifndef YOLO_DNN_STACK_X5
  reset();
  submit_rc = -14;
  expect(yolo::infer_sync(outputs, inputs, 2, model) == -14);
  expect(waited == 0 && released == 1);
#endif
  reset();
  expect(yolo::infer_sync(outputs, inputs, 0, model) != 0);
  expect(created == 0);
  expect(yolo::infer_sync(nullptr, inputs, 2, model) != 0);
  expect(created == 0);
  hbDNNTensor raw_inputs[4]{};
  expect(yolo::infer_sync(outputs, raw_inputs, 4, model) != 0);
  expect(created == 0);
  expect(yolo::infer_tensors_sync(outputs, raw_inputs, 0, model) != 0);
  expect(created == 0);
  expect(yolo::infer_tensors_sync(outputs, raw_inputs, 4, model) == 0);
  expect(created == 1 && waited == 1 && released == 1);
}
int main() {
  inference_failures();
  split();
  auto p = plan();
  expect(p.protocol == yolo::InputProtocol::kSplitNv12 && p.y_stride == 16);
  std::vector<uint8_t> y(24), uv(12);
  for (size_t i = 0; i < y.size(); ++i)
    y[i] = i;
  for (size_t i = 0; i < uv.size(); ++i)
    uv[i] = 100 + i;
  {
    yolo::Nv12Input input;
    expect(input.allocate(nullptr, p));
    expect(input.upload_planes(p, y.data(), y.size(), uv.data(), uv.size()));
    for (int plane = 0; plane < 2; ++plane) {
      auto *mem = YOLO_SYS_MEM(input.tensors()[plane]);
      auto *bytes = static_cast<uint8_t *>(mem->virAddr);
      for (int row = 0; row < (plane == 0 ? 4 : 2); ++row) {
        const auto &source = plane == 0 ? y : uv;
        expect(std::equal(source.begin() + row * 6,
                          source.begin() + (row + 1) * 6, bytes + row * 16));
        for (int col = 6; col < 16; ++col)
          expect(bytes[row * 16 + col] == 0);
      }
    }
    expect(
        !input.upload_planes(p, y.data(), y.size() - 1, uv.data(), uv.size()));
    auto foreign = p;
    foreign.y_stride = 6;
    expect(!input.upload_planes(foreign, y.data(), y.size(), uv.data(),
                                uv.size()));
    fail_flush = -1;
    expect(!input.upload_planes(p, y.data(), y.size(), uv.data(), uv.size()));
    fail_flush = 0;
    std::vector<uint8_t> i420(36);
    for (size_t i = 0; i < i420.size(); ++i)
      i420[i] = i;
    expect(input.upload(p, i420.data()));
    const auto *packed_uv =
        static_cast<const uint8_t *>(YOLO_SYS_MEM(input.tensors()[1])->virAddr);
    for (int i = 0; i < 6; ++i) {
      const int offset = (i / 3) * 16 + (i % 3) * 2;
      expect(packed_uv[offset] == 24 + i);
      expect(packed_uv[offset + 1] == 30 + i);
    }
  }
  expect(freed == 2);
  split(true);
  p = plan();
  expect(p.y_stride == 64);
  {
    yolo::Nv12Input input;
    expect(input.allocate(nullptr, p));
    expect(YOLO_SYS_MEM(input.tensors()[0])->memSize == 256);
  }
  // A static padded row pitch must drive dynamic capacity, not align(width).
  split();
  for (auto &d : metadata) {
    d.alignedByteSize = -1;
    d.stride[0] = -1;
  }
  p = plan();
  {
    yolo::Nv12Input input;
    expect(input.allocate(nullptr, p));
    expect(YOLO_SYS_MEM(input.tensors()[0])->memSize == 64);
  }
  split();
  for (auto &d : metadata) {
    d.stride[1] = 128;
    d.stride[0] = -1;
    d.alignedByteSize = -1;
  }
  p = plan();
  {
    yolo::Nv12Input input;
    expect(input.allocate(nullptr, p));
    expect(YOLO_SYS_MEM(input.tensors()[0])->memSize == 512);
    expect(input.upload_planes(p, y.data(), 24, uv.data(), 12));
  }
  split();
  p = plan();
  fail_alloc = 1;
  {
    yolo::Nv12Input input;
    expect(!input.allocate(nullptr, p));
    expect(allocated == freed);
    expect(!input.upload_planes(p, y.data(), 24, uv.data(), 12));
  }
  split();
  p = plan();
  metadata[0].stride[1] = 5;
  {
    yolo::Nv12Input input;
    expect(!input.allocate(nullptr, p));
    expect(allocated == 0);
  }
  split();
  metadata[0].stride[1] = 5;
  rejects_metadata();
  split();
  metadata[0].stride[2] = 2;
  rejects_metadata();
  split();
  metadata[0].stride[3] = 0;
  rejects_metadata();
  split();
  metadata[0].alignedByteSize = 1;
  rejects_metadata();
  split();
  metadata[0].quantiType = 1;
  rejects_metadata();
  split();
  metadata[0].validShape.dimensionSize[1] = 0;
  rejects_metadata();
  split();
  metadata[0].stride[1] = std::numeric_limits<int>::max();
  rejects_metadata();
  split();
  fail_query = 1;
  rejects_metadata();
#ifdef YOLO_DNN_STACK_X5
  for (int layout : {HB_DNN_LAYOUT_NCHW, HB_DNN_LAYOUT_NHWC}) {
    reset();
    hbDNNTensorProperties d{};
    d.tensorType = HB_DNN_IMG_TYPE_NV12;
    d.quantiType = NONE;
    d.tensorLayout = layout;
    d.validShape = layout == HB_DNN_LAYOUT_NCHW
                       ? hbDNNTensorShape{4, {1, 3, 4, 6}}
                       : hbDNNTensorShape{4, {1, 4, 6, 3}};
    d.alignedShape = d.validShape;
    d.alignedByteSize = 36;
    metadata = {d};
    p = plan();
    expect(p.protocol == yolo::InputProtocol::kPackedNv12);
    {
      yolo::Nv12Input input;
      expect(input.allocate(nullptr, p));
      expect(input.upload_planes(p, y.data(), 24, uv.data(), 12));
      auto *bytes =
          static_cast<uint8_t *>(YOLO_SYS_MEM(input.tensors()[0])->virAddr);
      expect(std::equal(y.begin(), y.end(), bytes));
      expect(std::equal(uv.begin(), uv.end(), bytes + 24));
    }
    metadata[0].alignedByteSize = 24;
    rejects_metadata();
  }
#endif
  reset();
}
