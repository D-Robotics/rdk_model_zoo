# Host tests for shared helpers and production SDK plumbing. No vendor headers
# are required: narrow X5/UCP API doubles exercise descriptor/lifetime logic;
# they do not prove real SDK ABI compatibility.
#
#   cmake -S . -B build-host && cmake --build build-host && ctest --test-dir build-host --output-on-failure

cmake_minimum_required(VERSION 3.13)
project(yolo_cpp_common_tests)

enable_testing()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

foreach(test_name test_decode test_head_probe test_nv12_geometry test_benchmark test_classification test_task_outputs test_obb_decode)
  add_executable(${test_name} ${test_name}.cc
                 ../common/nv12_geometry.cc
                 ../common/benchmark.cc)
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Compile both descriptor adapters against a narrow host test double. These
# exercise production binding/resource code, not actual SDK ABI compatibility.
foreach(stack X5 UCP)
  set(target test_classification_binding_${stack})
  add_executable(${target} test_classification_binding.cc)
  target_include_directories(${target} BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/fake_classification)
  target_compile_definitions(${target} PRIVATE YOLO_DNN_STACK_${stack}=1)
  add_test(NAME ${target} COMMAND ${target})
endforeach()

foreach(stack X5 UCP)
  set(target test_task_output_binding_${stack})
  add_executable(${target} test_task_output_binding.cc)
  target_include_directories(${target} BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/fake_task_outputs)
  target_compile_definitions(${target} PRIVATE YOLO_DNN_STACK_${stack}=1)
  add_test(NAME ${target} COMMAND ${target})
endforeach()

# Compile the production input/task implementation against both API layouts.
foreach(stack x5 ucp)
  set(target test_dnn_io_${stack})
  add_executable(${target} test_dnn_io.cc ../common/dnn_io.cc ../common/nv12_geometry.cc)
  target_include_directories(${target} BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/fake_dnn_io/${stack})
  target_compile_options(${target} PRIVATE -Wall -Wextra -Werror)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(${target} PRIVATE -fsanitize=address,undefined)
  endif()
  add_test(NAME ${target} COMMAND ${target})
endforeach()
