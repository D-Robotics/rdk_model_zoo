// Independent host doubles: no vendor SDK or model execution.
#include "gemma4_vision_engine.hpp"
#include "hb_utils.hpp"
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <set>
#include <string>
std::string fail;
bool vision_mode = false;
std::set<void *> buffers, tasks, models;
int allocations = 0, releases = 0, clean = 0, invalidate = 0;
int error(const char *name) { return fail == name ? -1 : 0; }
const char *hbDNNGetErrorDesc(int) { return "fixture"; }
const char *hbUCPGetErrorDesc(int) { return "fixture"; }
int hbUCPMallocCached(hbUCPSysMem *m, int64_t n, int) {
  ++allocations;
  if (fail == "alloc_null")
    return 0;
  m->virAddr = std::malloc(n);
  buffers.insert(m->virAddr);
  return error("alloc") || (fail == "second_alloc" && allocations == 2) ? -1
                                                                        : 0;
}
int hbUCPFree(hbUCPSysMem *m) {
  assert(buffers.erase(m->virAddr) == 1);
  std::free(m->virAddr);
  m->virAddr = nullptr;
  return 0;
}
int hbUCPMemFlush(hbUCPSysMem *, int mode) {
  if (mode == HB_SYS_MEM_CACHE_CLEAN) {
    ++clean;
    return error("clean");
  }
  ++invalidate;
  return error("invalidate");
}
int hbUCPSubmitTask(hbUCPTaskHandle_t, hbUCPSchedParam *) {
  return error("submit");
}
int hbUCPWaitTaskDone(hbUCPTaskHandle_t, int) { return error("wait"); }
int hbUCPReleaseTask(hbUCPTaskHandle_t t) {
  ++releases;
  assert(tasks.erase(t) == 1);
  delete static_cast<int *>(t);
  return error("release");
}
int hbDNNInferV2(hbUCPTaskHandle_t *t, hbDNNTensor *out, const hbDNNTensor *in,
                 hbDNNHandle_t) {
  if (fail == "infer_null")
    return 0;
  if (vision_mode) {
    const auto &ip = in[0].properties;
    const auto *pixels =
        static_cast<const unsigned char *>(in[0].sysMem.virAddr);
    uint16_t first = 0, second = 0;
    std::memcpy(&first, pixels, 2);
    std::memcpy(&second, pixels + ip.stride[0], 2);
    assert(first == 0x3c00 && second == 0x3800);
    assert(pixels[2] == 0 && pixels[3] == 0 && pixels[768 * ip.stride[1]] == 0);
    uint16_t last_pixel = 0;
    std::memcpy(&last_pixel, pixels + 767 * ip.stride[1], 2);
    assert(last_pixel == 0x3800);
    const auto &op = out[0].properties;
    auto *dst = static_cast<unsigned char *>(out[0].sysMem.virAddr);
    std::memset(dst, 0xcd, op.alignedByteSize);
    for (int row = 0; row < 280; ++row)
      for (int col = 0; col < 1536; ++col) {
        const bool last = row == 279 && col == 1535;
        auto *cell = dst + row * op.stride[0] + col * op.stride[1];
        if (op.tensorType == HB_DNN_TENSOR_TYPE_F16) {
          uint16_t value = last ? 0x3400 : 0xb800;
          std::memcpy(cell, &value, 2);
        } else {
          float value = last ? 0.25f : -0.5f;
          std::memcpy(cell, &value, 4);
        }
      }
    if (fail == "nonfinite") {
      float value = std::numeric_limits<float>::infinity();
      std::memcpy(dst, &value, 4);
    }
  }
  *t = new int;
  tasks.insert(*t);
  return error("infer");
}
int hbDNNGetCompileBpuCoreNum(int32_t *n, hbDNNHandle_t) {
  *n = 2;
  return error("cores");
}
int hbDNNGetTaskOutputTensorProperties(hbDNNTensorProperties *p,
                                       hbUCPTaskHandle_t, int, int) {
  if (fail == "after_capacity")
    p->alignedByteSize += 4;
  if (fail == "after_shape")
    p->validShape.dimensionSize[0] += 1;
  return error("task_props");
}
int hbDNNInitializeFromFiles(hbDNNPackedHandle_t *p, const char **, int) {
  if (fail == "model_null")
    return 0;
  *p = new int;
  models.insert(*p);
  return error("load");
}
int hbDNNRelease(hbDNNPackedHandle_t p) {
  assert(models.erase(p) == 1);
  delete static_cast<int *>(p);
  return 0;
}
int hbDNNGetModelHandle(hbDNNHandle_t *h, hbDNNPackedHandle_t p, const char *) {
  if (fail != "handle_null")
    *h = p;
  return error("handle");
}
int hbDNNGetInputCount(int *n, hbDNNHandle_t) {
  *n = 1;
  return error("input_count");
}
int hbDNNGetOutputCount(int *n, hbDNNHandle_t) {
  *n = 1;
  return error("output_count");
}
hbDNNTensorProperties vision_properties(bool input) {
  hbDNNTensorProperties p{};
  p.tensorType = input ? HB_DNN_TENSOR_TYPE_F16 : HB_DNN_TENSOR_TYPE_F32;
  p.validShape.numDimensions = 2;
  p.validShape.dimensionSize[0] = input ? 2520 : 280;
  p.validShape.dimensionSize[1] = input ? 768 : 1536;
  p.stride[1] = input ? 2 : 4;
  p.stride[0] = p.stride[1] * p.validShape.dimensionSize[1];
  p.alignedByteSize = p.stride[0] * p.validShape.dimensionSize[0];
  if (vision_mode) {
    if (!input && fail == "f16")
      p.tensorType = HB_DNN_TENSOR_TYPE_F16;
    p.stride[1] = (p.tensorType == HB_DNN_TENSOR_TYPE_F16 ? 2 : 4) * 2;
    p.stride[0] = p.stride[1] * p.validShape.dimensionSize[1] + 32;
    p.alignedByteSize = p.stride[0] * p.validShape.dimensionSize[0];
    if (input && fail == "input_type")
      p.tensorType = HB_DNN_TENSOR_TYPE_F32;
    if (!input && fail == "output_type")
      p.tensorType = HB_DNN_TENSOR_TYPE_S32;
    if (fail == "quantized")
      p.quantiType = SCALE;
    if (fail == "bad_shape")
      p.validShape.dimensionSize[0] += 1;
    if (fail == "overlap")
      p.stride[0] = 2;
    if (fail == "bad_rank")
      p.validShape.numDimensions = 99;
  }
  return p;
}
int hbDNNGetInputTensorProperties(hbDNNTensorProperties *p, hbDNNHandle_t,
                                  int) {
  *p = vision_properties(true);
  if (fail == "zero_bytes")
    p->alignedByteSize = 0;
  return error("input_props");
}
int hbDNNGetOutputTensorProperties(hbDNNTensorProperties *p, hbDNNHandle_t,
                                   int) {
  *p = vision_properties(false);
  return error("output_props");
}

int main(int argc, char **argv) {
  assert(argc == 3);
  std::string action = argv[1];
  fail = argv[2];
  vision_mode = action == "vision";
  bool threw = false;
  try {
    if (action == "tensor") {
      auto t = MakeTensor(nullptr, true, 0);
      std::vector<hbDNNTensor> ts{t};
      FreeTensors(ts);
    } else if (action == "vision") {
      gemma4::VisionEngine engine("fixture.hbm");
      std::vector<float> patches(2520 * 768, 0.5f);
      patches[0] = 1.f;
      auto features = engine.Infer(patches);
      assert(features.size() == 280u * 1536);
      for (size_t i = 0; i + 1 < features.size(); ++i)
        assert(features[i] == -0.5f);
      assert(features.back() == 0.25f);
    } else if (action == "constructor") {
      gemma4::VisionEngine engine("fixture.hbm");
    } else {
      std::vector<hbDNNTensor> inputs(2), outputs(2);
      if (action == "all")
        RunInfer(nullptr, inputs, outputs);
      else
        RunInferSelective(nullptr, inputs, outputs, {1}, {0});
    }
  } catch (const std::exception &) {
    threw = true;
  }
  assert(threw == (fail != "none" && fail != "f16"));
  assert(buffers.empty());
  assert(tasks.empty());
  assert(models.empty());
  if ((action == "all" || action == "selective") && fail == "none") {
    assert(releases == 1);
    assert(clean == (action == "all" ? 2 : 1));
    assert(invalidate == (action == "all" ? 2 : 1));
  }
  std::cout << action << "/" << fail << " passed\n";
}
