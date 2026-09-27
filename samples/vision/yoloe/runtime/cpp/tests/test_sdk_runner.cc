// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
// Runs production adapter/control flow against API doubles, never real SDK ABI.
#include "common/dnn_io.h"
#include "sdk_runner.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#define expect(v)                                                              \
  do {                                                                         \
    if (!(v))                                                                  \
      throw std::runtime_error(#v);                                            \
  } while (0)
std::vector<hbDNNTensorProperties> ins, outs;
int allocated = 0, freed = 0, models = 0, released_models = 0, tasks = 0,
    released_tasks = 0;
int fail_alloc = -1, fail_query = -1, fail_flush = 0, fail_infer = 0,
    model_count = 1, generation = 0;
bool nan_output = false;
int initialize_rc = 0;
int hbDNNInitializeFromFiles(yolo_packed_handle_t *p, const char **, int) {
  ++models;
  *p = reinterpret_cast<void *>(1);
  return initialize_rc;
}
int hbDNNRelease(yolo_packed_handle_t) {
  ++released_models;
  return 0;
}
int hbDNNGetModelNameList(const char ***names, int *n, yolo_packed_handle_t) {
  static const char *name = "fixture";
  *names = &name;
  *n = model_count;
  return 0;
}
int hbDNNGetModelHandle(hbDNNHandle_t *p, yolo_packed_handle_t, const char *) {
  *p = reinterpret_cast<void *>(2);
  return 0;
}
int hbDNNGetInputCount(int32_t *n, hbDNNHandle_t) {
  *n = ins.size();
  return 0;
}
int hbDNNGetInputTensorProperties(hbDNNTensorProperties *p, hbDNNHandle_t,
                                  int i) {
  *p = ins.at(i);
  return 0;
}
int hbDNNGetOutputCount(int32_t *n, hbDNNHandle_t) {
  *n = outs.size();
  return 0;
}
int hbDNNGetOutputTensorProperties(hbDNNTensorProperties *p, hbDNNHandle_t,
                                   int i) {
  if (i == fail_query)
    return -2;
  *p = outs.at(i);
  return 0;
}
int test_allocate(TestMemory *m, int n) {
  if (allocated == fail_alloc)
    return -3;
  m->virAddr = std::calloc(1, n);
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
int test_flush(TestMemory *, int) { return fail_flush; }
int test_create(void **) {
  throw std::runtime_error("use tensor-aware inference fixture");
}
int test_wait(void *p) {
  expect(p);
  return 0;
}
int test_release(void *p) {
  expect(p);
  ++released_tasks;
  return 0;
}
int test_submit(void *p, hbUCPSchedParam *s) {
  expect(p && s->backend == HB_UCP_BPU_CORE_ANY);
  return 0;
}
int test_infer(void **task, hbDNNTensor *output, hbDNNTensor *input) {
  ++tasks;
  *task = reinterpret_cast<void *>(3);
  expect(static_cast<uint8_t *>(YOLO_SYS_MEM(input[0])->virAddr)[0] == 82);
  if (fail_infer)
    return fail_infer;
  for (size_t i = 0; i < outs.size(); ++i) {
    auto *values = static_cast<float *>(YOLO_SYS_MEM(output[i])->virAddr);
    values[0] = nan_output ? std::numeric_limits<float>::quiet_NaN()
                           : float(i + 1 + generation);
  }
  return 0;
}
template <class F> void rejects(F fn) {
  bool failed = false;
  try {
    fn();
  } catch (const std::exception &) {
    failed = true;
  }
  expect(failed);
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
void setup(bool e26) {
  expect(allocated == freed && models == released_models &&
         tasks == released_tasks);
  allocated = freed = models = released_models = tasks = released_tasks =
      generation = 0;
  fail_alloc = fail_query = -1;
  fail_flush = fail_infer = 0;
  model_count = 1;
  initialize_rc = 0;
  nan_output = false;
  ins.clear();
  outs.clear();
#ifdef YOLO_DNN_STACK_X5
  hbDNNTensorProperties p{};
  p.tensorType = HB_DNN_IMG_TYPE_NV12;
  p.quantiType = NONE;
  p.tensorLayout = HB_DNN_LAYOUT_NCHW;
  p.validShape = {4, {1, 3, 640, 640}};
  p.alignedShape = p.validShape;
  p.alignedByteSize = 640 * 640 * 3 / 2;
  ins.push_back(p);
#else
  for (int i = 0; i < 2; ++i) {
    hbDNNTensorProperties p{};
    p.tensorType = HB_DNN_TENSOR_TYPE_U8;
    p.quantiType = NONE;
    p.validShape = {4, {1, i ? 320 : 640, i ? 320 : 640, i ? 2 : 1}};
    p.alignedByteSize = -1;
    for (int &s : p.stride)
      s = -1;
    ins.push_back(p);
  }
#endif
  for (int grid : {80, 40, 20})
    for (int c : {4585, e26 ? 4 : 64, 32})
      outs.push_back(descriptor(grid, grid, c));
  outs.push_back(descriptor(160, 160, 32));
  std::reverse(outs.begin(), outs.end());
}
int main(int argc, char **argv) {
  expect(argc == 2);
  std::string path = argv[1];
  {
    std::ofstream f(path, std::ios::binary);
    f << "host-only model placeholder";
  }
#ifdef YOLO_DNN_STACK_X5
  const bool e26 = false;
  yoloe::SdkModel spec{path, "x5", "11s"};
#else
  const bool e26 = true;
  yoloe::SdkModel spec{path, "s100p", "26n"};
#endif
  auto gate = [&](const yoloe::SdkModel &s) {
    expect(s.path == path && models == 0);
  };
  setup(e26);
  {
    yoloe::SdkRunner runner(spec, gate);
    expect(runner.protocol() ==
           (e26 ? yoloe::Protocol::E26 : yoloe::Protocol::E11));
    yoloe::Nv12Input input;
    input.y.assign(640 * 640, 82);
    input.uv.assign(640 * 320, 128);
    auto first = runner.infer(input);
    expect(first[0][0] == 10 && first[9][0] == 1);
    generation = 20;
    auto second = runner.infer(input);
    expect(second[0][0] == 30 && first[0][0] == 10);
    fail_flush = -4;
    int before = tasks;
    rejects([&] { runner.infer(input); });
    expect(tasks == before);
    fail_flush = 0;
    fail_infer = -5;
    rejects([&] { runner.infer(input); });
    fail_infer = 0;
    expect(tasks == released_tasks);
    nan_output = true;
    rejects([&] { runner.infer(input); });
    nan_output = false;
    input.y.pop_back();
    before = tasks;
    rejects([&] { runner.infer(input); });
    expect(tasks == before);
  }
  setup(e26);
  rejects([&] { yoloe::SdkRunner runner(spec, {}); });
  expect(models == 0);
  rejects([&] {
    yoloe::SdkRunner runner(spec, [](const yoloe::SdkModel &) {
      throw std::runtime_error("identity/hash mismatch");
    });
  });
  expect(models == 0);
  auto bad = spec;
  bad.target = "s600";
  rejects([&] { yoloe::SdkRunner runner(bad, gate); });
  expect(models == 0);
  bad = spec;
  bad.variant = "11m";
  bad.target = "s100p";
  rejects([&] { yoloe::SdkRunner runner(bad, gate); });
  expect(models == 0);
  setup(e26);
  outs[0].quantiType = 1;
  rejects([&] { yoloe::SdkRunner runner(spec, gate); });
  expect(allocated == 0);
  setup(e26);
  outs[0] = outs[1];
  rejects([&] { yoloe::SdkRunner runner(spec, gate); });
  expect(allocated == 0);
  setup(e26);
  fail_query = 5;
  rejects([&] { yoloe::SdkRunner runner(spec, gate); });
  expect(allocated == 0);
  setup(e26);
  fail_alloc = 3;
  rejects([&] { yoloe::SdkRunner runner(spec, gate); });
  expect(allocated == freed);
  setup(e26);
  initialize_rc = -17;
  rejects([&] { yoloe::SdkRunner runner(spec, gate); });
  expect(models == released_models && allocated == 0);
  setup(e26);
  model_count = 2;
  rejects([&] { yoloe::SdkRunner runner(spec, gate); });
  expect(allocated == 0);
#ifndef YOLO_DNN_STACK_X5
  setup(false);
  {
    auto s11 = spec;
    s11.target = "s100";
    s11.variant = "11s";
    yoloe::SdkRunner runner(s11, gate);
    expect(runner.protocol() == yoloe::Protocol::E11);
  }
#endif
  setup(e26);
  std::remove(path.c_str());
  rejects([&] { yoloe::SdkRunner runner(spec, gate); });
  expect(models == 0);
  setup(e26);
}
