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
}
int main(int argc,char**) {
 if(argc==1) { reset(); create_rc=-11; hbDNNTensor inputs[2]{},outputs[1]{};
 expect(yolo::infer_sync(outputs,inputs,2,reinterpret_cast<void*>(2))==-11);
 expect(released==1);
 } else {split();for(auto& d:metadata){d.stride[1]=128;d.stride[0]=-1;d.alignedByteSize=-1;}
 auto p=plan();yolo::Nv12Input input;expect(input.allocate(nullptr,p));
 expect(YOLO_SYS_MEM(input.tensors()[0])->memSize>=128*4);}
}
