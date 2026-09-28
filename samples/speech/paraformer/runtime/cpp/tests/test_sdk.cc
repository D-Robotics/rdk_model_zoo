// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
// Explicit host API double; not real vendor ABI or model evidence.
#include "common/dnn_io.h"
#include "sdk_runner.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unistd.h>
using namespace paraformer;
struct Physical {
  std::string name;
  hbDNNTensorProperties prop;
};
std::vector<Physical> inputs, outputs;
int models = 0, released = 0, allocations = 0, freed = 0, tasks = 0,
    tasks_freed = 0;
int fail_init = 0, fail_query = 0, fail_alloc = -1, fail_infer = 0,
    fail_submit = 0, fail_wait = 0, fail_flush = 0;
bool partial_alloc = false;
int generation = 0;
int hbDNNInitializeFromFiles(void **p, const char **, int n) {
  assert(n == 1);
  *p = (void *)1;
  ++models;
  return fail_init;
}
int hbDNNRelease(void *) {
  ++released;
  return 0;
}
int hbDNNGetModelNameList(const char ***p, int *n, void *) {
  static const char *v = "fixture";
  *p = &v;
  *n = 1;
  return 0;
}
int hbDNNGetModelHandle(void **p, void *, const char *) {
  *p = (void *)2;
  return 0;
}
int hbDNNGetInputCount(int32_t *n, void *) {
  *n = inputs.size();
  return 0;
}
int hbDNNGetOutputCount(int32_t *n, void *) {
  *n = outputs.size();
  return 0;
}
int hbDNNGetInputName(const char **n, void *, int i) {
  *n = inputs.at(i).name.c_str();
  return 0;
}
int hbDNNGetOutputName(const char **n, void *, int i) {
  *n = outputs.at(i).name.c_str();
  return 0;
}
int hbDNNGetInputTensorProperties(hbDNNTensorProperties *p, void *, int i) {
  *p = inputs.at(i).prop;
  return fail_query;
}
int hbDNNGetOutputTensorProperties(hbDNNTensorProperties *p, void *, int i) {
  *p = outputs.at(i).prop;
  return fail_query;
}
int test_allocate(TestMemory *m, int n) {
  if (allocations == fail_alloc && !partial_alloc)
    return -1;
  m->virAddr = std::malloc(n);
  m->memSize = n;
  ++allocations;
  return partial_alloc ? -1 : 0;
}
int test_free(TestMemory *m) {
  std::free(m->virAddr);
  m->virAddr = nullptr;
  ++freed;
  return 0;
}
int test_flush(TestMemory *, int flag) { return fail_flush == flag ? -1 : 0; }
int test_create(void **) { throw std::runtime_error("unexpected create"); }
int test_wait(void *) { return fail_wait; }
int test_release(void *) {
  ++tasks_freed;
  return 0;
}
int test_submit(void *, hbUCPSchedParam *p) {
  assert(p->backend == HB_UCP_BPU_CORE_ANY);
  return fail_submit;
}
size_t count(const hbDNNTensorProperties &p) {
  size_t n = 1;
  for (int i = 0; i < p.validShape.numDimensions; ++i)
    n *= p.validShape.dimensionSize[i];
  return n;
}
size_t offset(size_t flat, const hbDNNTensorProperties &p) {
  size_t o = 0;
  for (int i = p.validShape.numDimensions - 1; i >= 0; --i) {
    o += (flat % p.validShape.dimensionSize[i]) * p.stride[i];
    flat /= p.validShape.dimensionSize[i];
  }
  return o;
}
int test_infer(void **t, hbDNNTensor *out, hbDNNTensor *in) {
  *t = (void *)3;
  ++tasks;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &p = inputs[i].prop;
    auto *bytes = (unsigned char *)in[i].sysMem.virAddr;
    for (size_t j = 0; j < count(p); ++j) {
      if (p.tensorType == HB_DNN_TENSOR_TYPE_S32) {
        int32_t v;
        std::memcpy(&v, bytes + offset(j, p), 4);
        assert(v == 3);
      } else {
        float v;
        std::memcpy(&v, bytes + offset(j, p), 4);
        assert(v == float(j % 17));
      }
    }
    assert(bytes[4] == 0); // All fixtures deliberately pad each element.
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto &p = outputs[i].prop;
    auto *bytes = (unsigned char *)out[i].sysMem.virAddr;
    std::memset(bytes, 0x7f, out[i].sysMem.memSize);
    for (size_t j = 0; j < count(p); ++j) {
      if (p.tensorType == HB_DNN_TENSOR_TYPE_S32) {
        int32_t v = 3;
        std::memcpy(bytes + offset(j, p), &v, 4);
      } else {
        float v = float(j + generation);
        std::memcpy(bytes + offset(j, p), &v, 4);
      }
    }
  }
  return fail_infer;
}
Physical tensor(std::string name, std::vector<int> shape,
                bool integer = false) {
  hbDNNTensorProperties p{};
  p.tensorType = integer ? HB_DNN_TENSOR_TYPE_S32 : HB_DNN_TENSOR_TYPE_F32;
  p.quantiType = NONE;
  p.validShape.numDimensions = shape.size();
  int span = 8;
  for (int i = int(shape.size()) - 1; i >= 0; --i) {
    p.validShape.dimensionSize[i] = shape[i];
    p.stride[i] = span;
    span *= shape[i];
  }
  p.alignedByteSize = span;
  return {std::move(name), p};
}
void balanced() {
  assert(models == released && allocations == freed && tasks == tasks_freed);
}
void setup(Stage stage) {
  balanced();
  models = released = allocations = freed = tasks = tasks_freed = 0;
  fail_init = fail_query = fail_infer = fail_submit = fail_wait = fail_flush =
      0;
  fail_alloc = -1;
  partial_alloc = false;
  generation = 0;
  const std::string context = "/encoder/after_norm/Add_1_output_0";
  if (stage == Stage::Encoder) {
    inputs = {tensor("speech", {1, 400, 560})};
    outputs = {tensor(context, {1, 400, 512})};
  }
  if (stage == Stage::Predictor) {
    inputs = {tensor(context, {1, 400, 512})};
    outputs = {tensor("/predictor/Concat_5_output_0", {1, 401, 512}),
               tensor("/predictor/Add_output_0", {1, 401})};
  }
  if (stage == Stage::Decoder) {
    inputs = {tensor("shape_8609", {1, 100, 512}),
              tensor("bias_embed", {1, 1, 512}), tensor("token_num", {1}, true),
              tensor(context, {1, 400, 512})};
    outputs = {tensor("token_num", {1}, true),
               tensor("logits", {1, 100, 8404})};
  }
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
RawTensors values(const SdkMetadata &m) {
  RawTensors r;
  for (const auto &p : m.inputs) {
    size_t n = 1;
    for (int d : p.shape)
      n *= d;
    if (p.dtype == "int32")
      r[p.role] = std::vector<int32_t>(n, 3);
    else {
      std::vector<float> v(n);
      for (size_t j = 0; j < n; ++j)
        v[j] = float(j % 17);
      r[p.role] = std::move(v);
    }
  }
  return r;
}
int main() {
  char path[] = "/tmp/paraformer-sdk-XXXXXX";
  int fd = mkstemp(path);
  assert(fd >= 0);
  close(fd);
  {
    std::ofstream f(path);
    f << "fixture";
  }
  auto gate = [](const SdkModel &) {};
  setup(Stage::Encoder);
  rejects([&] { SdkRunner r({path, "s100", Stage::Encoder}, {}); });
  assert(models == 0);
  rejects([&] { SdkRunner r({path, "s100p", Stage::Encoder}, gate); });
  assert(models == 0);
  rejects([&] {
    SdkRunner r({path, "s100", Stage::Encoder},
                [](const auto &) { throw std::runtime_error("identity"); });
  });
  assert(models == 0);
  for (auto stage : {Stage::Encoder, Stage::Predictor, Stage::Decoder}) {
    setup(stage);
    {
      SdkRunner r({path, "s100", stage}, gate);
      auto in = values(r.metadata());
      auto first = r.infer(in);
      generation = 7;
      auto second = r.infer(in);
      for (const auto &p : r.metadata().outputs)
        if (p.dtype == "float32") {
          assert(std::get<std::vector<float>>(first.at(p.role))[0] == 0);
          assert(std::get<std::vector<float>>(second.at(p.role))[0] == 7);
        }
      const int before = tasks;
      auto bad = in;
      bad["extra"] = std::vector<float>{1};
      rejects([&] { r.infer(bad); });
      assert(tasks == before);
      bad = in;
      bad.begin()->second = std::vector<int32_t>{-1};
      rejects([&] { r.infer(bad); });
      assert(tasks == before);
      bad = in;
      for (auto &entry : bad) {
        if (auto *floats = std::get_if<std::vector<float>>(&entry.second)) {
          (*floats)[0] = std::numeric_limits<float>::infinity();
          break;
        }
      }
      rejects([&] { r.infer(bad); });
      assert(tasks == before);
      if (stage == Stage::Decoder) {
        bad = in;
        bad["count"] = std::vector<int32_t>{101};
        rejects([&] { r.infer(bad); });
        assert(tasks == before);
        bad["count"] = std::vector<float>{3};
        rejects([&] { r.infer(bad); });
        assert(tasks == before);
      }
    }
    balanced();
  }
  // Optional decoder pass-through may be absent; both documented aliases work.
  setup(Stage::Decoder);
  outputs.erase(outputs.begin());
  inputs[0].name = "onnx::Shape_8609";
  {
    SdkRunner r({path, "s100", Stage::Decoder}, gate);
    assert(r.infer(values(r.metadata())).size() == 1);
  }
  balanced();
  setup(Stage::Decoder);
  inputs[1] = tensor("onnx::Shape_8609", {1, 100, 512});
  rejects([&] { SdkRunner r({path, "s100", Stage::Decoder}, gate); });
  balanced();
  setup(Stage::Decoder);
  outputs[0].prop.tensorType = HB_DNN_TENSOR_TYPE_F32;
  rejects([&] { SdkRunner r({path, "s100", Stage::Decoder}, gate); });
  balanced();
  for (int failure = 0; failure < 12; ++failure) {
    setup(Stage::Encoder);
    switch (failure) {
    case 0:
      inputs[0].name = "wrong";
      break;
    case 1:
      outputs.push_back(outputs[0]);
      break;
    case 2:
      inputs[0].prop.tensorType = HB_DNN_TENSOR_TYPE_S8;
      break;
    case 3:
      outputs[0].prop.quantiType = 1;
      break;
    case 4:
      inputs[0].prop.stride[2] = 6;
      break;
    case 5:
      outputs[0].prop.alignedByteSize = 4;
      break;
    case 6:
      inputs[0].prop.validShape.dimensionSize[1] = 399;
      break;
    case 7:
      fail_init = -1;
      break;
    case 8:
      fail_query = -1;
      break;
    case 9:
      fail_alloc = 1;
      break;
    case 10:
      partial_alloc = true;
      break;
    case 11:
      inputs[0].prop.stride[0] = -1;
      break;
    }
    rejects([&] { SdkRunner r({path, "s100", Stage::Encoder}, gate); });
    balanced();
  }
  for (int failure = 0; failure < 5; ++failure) {
    setup(Stage::Encoder);
    {
      SdkRunner r({path, "s100", Stage::Encoder}, gate);
      switch (failure) {
      case 0:
        fail_flush = HB_SYS_MEM_CACHE_CLEAN;
        break;
      case 1:
        fail_infer = -1;
        break;
      case 2:
        fail_submit = -1;
        break;
      case 3:
        fail_wait = -1;
        break;
      case 4:
        fail_flush = HB_SYS_MEM_CACHE_INVALIDATE;
        break;
      }
      rejects([&] { r.infer(values(r.metadata())); });
    }
    balanced();
  }
  std::remove(path);
}
