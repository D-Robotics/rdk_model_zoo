# Host-side unit tests for the pure helpers in ../common. These do not
# require board headers: only benchmark.cc and nv12_geometry.cc (and the
# header-only decode/tensor_view logic) are compiled.
#
#   cmake -S . -B build-host && cmake --build build-host && ctest --test-dir build-host --output-on-failure

cmake_minimum_required(VERSION 3.10)
project(yolo_cpp_common_tests)

enable_testing()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

foreach(test_name test_decode test_head_probe test_nv12_geometry test_benchmark)
  add_executable(${test_name} ${test_name}.cc
                 ../common/nv12_geometry.cc
                 ../common/benchmark.cc)
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
