// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
// Production control flow with an API double; never real SDK ABI evidence.
#include "backend.hpp"
#include "asr.hpp"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unistd.h>
hbDNNTensorProperties in_prop{}, out_prop{};
int models = 0, models_freed = 0, allocs = 0, frees = 0, tasks = 0,
    tasks_freed = 0;
int model_count = 1, input_count = 1, output_count = 1, fail_alloc = -1,
    fail_query = 0;
int fail_init = 0, fail_infer = 0, fail_submit = 0, fail_wait = 0,
    fail_release = 0, fail_flush = 0;
bool partial_alloc = false, null_alloc = false;
int generation = 0;
int hbDNNInitializeFromFiles(void **p, const char **, int) {
  *p = reinterpret_cast<void *>(1);
  ++models;
  return fail_init;
}
int hbDNNRelease(void *) {
  ++models_freed;
  return 0;
}
int hbDNNGetModelNameList(const char ***p, int *n, void *) {
  static const char *name = "fixture";
  *p = &name;
  *n = model_count;
  return 0;
}
int hbDNNGetModelHandle(void **p, void *, const char *) {
  *p = reinterpret_cast<void *>(2);
  return 0;
}
int hbDNNGetInputCount(int32_t *n, void *) {
  *n = input_count;
  return 0;
}
int hbDNNGetOutputCount(int32_t *n, void *) {
  *n = output_count;
  return 0;
}
int hbDNNGetInputTensorProperties(hbDNNTensorProperties *p, void *, int) {
  *p = in_prop;
  return fail_query;
}
int hbDNNGetOutputTensorProperties(hbDNNTensorProperties *p, void *, int) {
  *p = out_prop;
  return fail_query;
}
int test_allocate(TestMemory *m, int n) {
  if (null_alloc)
    return 0;
  if (allocs == fail_alloc && !partial_alloc)
    return -3;
  m->virAddr = std::malloc(n);
  m->memSize = n;
  assert(m->virAddr);
  ++allocs;
  return partial_alloc ? -3 : 0;
}
int test_free(TestMemory *m) {
  assert(m->virAddr);
  std::free(m->virAddr);
  m->virAddr = nullptr;
  ++frees;
  return 0;
}
int test_flush(TestMemory *, int flag) { return flag == fail_flush ? -4 : 0; }
int test_create(void **) { throw std::runtime_error("unexpected create"); }
int test_wait(void *) { return fail_wait; }
int test_submit(void *, hbUCPSchedParam *s) {
  assert(s->backend == HB_UCP_BPU_CORE_ANY);
  return fail_submit;
}
int test_release(void *) {
  ++tasks_freed;
  return fail_release;
}
int test_infer(void **p, hbDNNTensor *output, hbDNNTensor *input) {
  *p = reinterpret_cast<void *>(3);
  ++tasks;
  auto *bytes = static_cast<unsigned char *>(input->sysMem.virAddr);
  for (int i = 0; i < 30000; ++i) {
    float v;
    std::memcpy(&v, bytes + i * in_prop.stride[1], 4);
    assert(v == float(i));
  }
  assert(bytes[4] == 0); // physical padding is cleared
  auto *dst = static_cast<unsigned char *>(output->sysMem.virAddr);
  std::memset(dst, 0x7f, output->sysMem.memSize);
  for (int t = 0; t < 4; ++t)
    for (int v = 0; v < 3503; ++v) {
      float value = float(t * 3503 + v + generation);
      std::memcpy(dst + t * out_prop.stride[1] + v * out_prop.stride[2], &value,
                  4);
    }
  return fail_infer;
}
void balanced() {
  assert(models == models_freed && allocs == frees && tasks == tasks_freed);
}
void setup() {
  balanced();
  models = models_freed = allocs = frees = tasks = tasks_freed = 0;
  model_count = input_count = output_count = 1;
  fail_alloc = -1;
  fail_query = fail_init = fail_infer = fail_submit = fail_wait = fail_release =
      fail_flush = 0;
  partial_alloc = null_alloc = false;
  generation = 0;
  in_prop = {};
  in_prop.tensorType = HB_DNN_TENSOR_TYPE_F32;
  in_prop.quantiType = NONE;
  in_prop.validShape = {2, {1, 30000}};
  in_prop.stride[1] = 8;
  in_prop.stride[0] = 240000;
  in_prop.alignedByteSize = 240000;
  out_prop = {};
  out_prop.tensorType = HB_DNN_TENSOR_TYPE_F32;
  out_prop.quantiType = NONE;
  out_prop.validShape = {3, {1, 4, 3503}};
  out_prop.stride[2] = 8;
  out_prop.stride[1] = 3504 * 8;
  out_prop.stride[0] = 4 * out_prop.stride[1];
  out_prop.alignedByteSize = out_prop.stride[0];
}
template <class F> void rejects(F f) {
  bool caught = false;
  try {
    f();
  } catch (const std::exception &) {
    caught = true;
  }
  assert(caught);
}
int main() {
  char name[] = "/tmp/asr-sdk-XXXXXX";
  int fd = mkstemp(name);
  assert(fd >= 0);
  close(fd);
  {
    std::ofstream f(name);
    f << "fixture";
  }
  const asr::SdkModel model{name, "s100"};
  auto gate = [](const asr::SdkModel &) {};
  setup();
  rejects([&] { asr::SdkRunner r(model, {}); });
  assert(models == 0);
  rejects([&] {
    asr::SdkRunner r(model, [](const auto &) {
      throw std::invalid_argument("identity mismatch");
    });
  });
  assert(models == 0);
  rejects([&] { asr::SdkRunner r({name, "s100p"}, gate); });
  assert(models == 0);
  for (int target = 0; target < 2; ++target) {
    setup();
    {
      asr::SdkRunner r({name, target ? "s600" : "s100"}, gate);
      assert(r.metadata().steps == 4);
      std::vector<float> input(30000);
      for (int i = 0; i < 30000; ++i)
        input[i] = float(i);
      auto first = r.infer(input);
      assert(first.size() == 4 * 3503 && first.back() == 14011);
      generation = 10;
      auto second = r.infer(input);
      assert(second[0] == 10 && first[0] == 0);
      rejects([&] { r.infer({1}); });
      input[0] = std::numeric_limits<float>::infinity();
      rejects([&] { r.infer(input); });
    }
    balanced();
  }
  for (int failure = 0; failure < 18; ++failure) {
    setup();
    switch (failure) {
    case 0:
      fail_init = -1;
      break;
    case 1:
      model_count = 2;
      break;
    case 2:
      input_count = 2;
      break;
    case 3:
      output_count = 2;
      break;
    case 4:
      fail_query = -1;
      break;
    case 5:
      in_prop.tensorType = HB_DNN_TENSOR_TYPE_S8;
      break;
    case 6:
      out_prop.validShape.dimensionSize[2] = 3502;
      break;
    case 7:
      out_prop.stride[1] = 4;
      break;
    case 8:
      out_prop.alignedByteSize = 4;
      break;
    case 9:
      fail_alloc = 1;
      break;
    case 10:
      partial_alloc = true;
      break;
    case 11:
      null_alloc = true;
      break;
    case 12:
      out_prop.quantiType = 1;
      break;
    case 13:
      out_prop.tensorType = HB_DNN_TENSOR_TYPE_S8;
      break;
    case 14:
      in_prop.stride[1] = -1;
      break;
    case 15:
      in_prop.validShape.dimensionSize[1] = 29999;
      break;
    case 16:
      out_prop.validShape.dimensionSize[1] = 0;
      break;
    case 17:
      out_prop.stride[2] = 6;
      break;
    }
    rejects([&] { asr::SdkRunner r(model, gate); });
    balanced();
  }
  for (int failure = 0; failure < 6; ++failure) {
    setup();
    {
      asr::SdkRunner r(model, gate);
      switch (failure) {
      case 0:
        fail_flush = HB_SYS_MEM_CACHE_CLEAN;
        break;
      case 1:
        fail_flush = HB_SYS_MEM_CACHE_INVALIDATE;
        break;
      case 2:
        fail_infer = -2;
        break;
      case 3:
        fail_submit = -2;
        break;
      case 4:
        fail_wait = -2;
        break;
      case 5:
        fail_release = -2;
        break;
      }
      std::vector<float> input(30000);
      for (int i = 0; i < 30000; ++i)
        input[i] = float(i);
      rejects([&] { r.infer(input); });
    }
    balanced();
  }
  std::remove(name);
}
