# Host tests for the consolidated C++ runtime vocabulary and production SDK
# plumbing. No vendor headers are required: the fake_dnn_io X5/UCP doubles
# select the stack through the genuine header probe in inc/backend.hpp; they
# exercise descriptor/lifetime/binding logic, not real SDK ABI compatibility.
#
#   cmake -S . -B build-host && cmake --build build-host && ctest --test-dir build-host --output-on-failure

cmake_minimum_required(VERSION 3.13)
project(yolo_cpp_runtime_tests)

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../inc)

# Pure vocabulary tests: everything they exercise is header-only in
# inc/yolo.hpp / inc/cli.hpp (decode math, plans, NV12 packing, benchmark
# bookkeeping, oriented-box geometry).
foreach(test_name test_decode test_head_probe test_nv12_geometry test_benchmark test_classification test_task_outputs test_obb_decode)
  add_executable(${test_name} ${test_name}.cc)
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Compile both descriptor adapters against the fake stacks. These exercise
# production binding/resource code (inc/backend.hpp), not actual SDK ABI
# compatibility.
foreach(stack X5 UCP)
  set(target test_classification_binding_${stack})
  add_executable(${target} test_classification_binding.cc)
  target_include_directories(${target} BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/fake_dnn_io/${stack})
  add_test(NAME ${target} COMMAND ${target})
endforeach()

foreach(stack X5 UCP)
  set(target test_task_output_binding_${stack})
  add_executable(${target} test_task_output_binding.cc)
  target_include_directories(${target} BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/fake_dnn_io/${stack})
  add_test(NAME ${target} COMMAND ${target})
endforeach()

# Compile the production input/task implementation (src/backend.cpp) against
# both API layouts.
foreach(stack x5 ucp)
  set(target test_dnn_io_${stack})
  add_executable(${target} test_dnn_io.cc ../src/backend.cpp)
  target_include_directories(${target} BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/fake_dnn_io/${stack})
  target_compile_options(${target} PRIVATE -Wall -Wextra -Werror)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(${target} PRIVATE -fsanitize=address,undefined)
  endif()
  add_test(NAME ${target} COMMAND ${target})
endforeach()

# Benchmark resource-shape regression: drives the production detect model and
# the CLI benchmark helper end-to-end against the fake stacks, so it needs
# real OpenCV (discovered via CMake config or pkg-config opencv5/opencv4;
# skipped only when neither is available).
find_package(OpenCV QUIET)
if(NOT OpenCV_FOUND)
  foreach(pc_name opencv5 opencv4)
    execute_process(COMMAND pkg-config --modversion ${pc_name}
                    RESULT_VARIABLE pc_result OUTPUT_QUIET ERROR_QUIET)
    if(pc_result EQUAL 0)
      set(OPENCV_PC_NAME ${pc_name})
      break()
    endif()
  endforeach()
endif()
if(OpenCV_FOUND OR OPENCV_PC_NAME)
  find_package(Threads REQUIRED)
  foreach(stack x5 ucp)
    set(target test_benchmark_streams_${stack})
    # cli.cpp drives every task benchmark, so all five task models link here.
    add_executable(${target} test_benchmark_streams.cc
                                ../src/backend.cpp
                                ../src/detect.cpp
                                ../src/segment.cpp
                                ../src/pose.cpp
                                ../src/classify.cpp
                                ../src/obb.cpp
                                ../src/cli.cpp)
    target_include_directories(${target} BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/fake_dnn_io/${stack})
    # Tensor-aware infer double: scenarios fabricate deterministic non-empty
    # task outputs the way test_full_models does.
    target_compile_definitions(${target} PRIVATE TEST_TENSOR_AWARE_INFER=1)
    if(OpenCV_FOUND)
      target_include_directories(${target} PRIVATE ${OpenCV_INCLUDE_DIRS})
      target_link_libraries(${target} PRIVATE ${OpenCV_LIBS})
    else()
      execute_process(COMMAND pkg-config --cflags ${OPENCV_PC_NAME}
                      OUTPUT_VARIABLE OPENCV_CFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
      execute_process(COMMAND pkg-config --libs ${OPENCV_PC_NAME}
                      OUTPUT_VARIABLE OPENCV_PC_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
      separate_arguments(OPENCV_CFLAGS)
      separate_arguments(OPENCV_PC_LIBS)
      target_compile_options(${target} PRIVATE ${OPENCV_CFLAGS})
      target_link_libraries(${target} PRIVATE ${OPENCV_PC_LIBS})
    endif()
    target_link_libraries(${target} PRIVATE Threads::Threads)
    add_test(NAME ${target} COMMAND ${target})

    # Full-model fixture: all five production task models driven through
    # preprocess/infer/postprocess/predict with tensor-aware fabricated
    # outputs (deterministic box/mask/keypoints/top-k/rotated box plus
    # ownership checks).
    set(target test_full_models_${stack})
    add_executable(${target} test_full_models.cc
                                ../src/backend.cpp
                                ../src/detect.cpp
                                ../src/segment.cpp
                                ../src/pose.cpp
                                ../src/classify.cpp
                                ../src/obb.cpp)
    target_include_directories(${target} BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/fake_dnn_io/${stack})
    target_compile_definitions(${target} PRIVATE TEST_TENSOR_AWARE_INFER=1)
    if(OpenCV_FOUND)
      target_include_directories(${target} PRIVATE ${OpenCV_INCLUDE_DIRS})
      target_link_libraries(${target} PRIVATE ${OpenCV_LIBS})
    else()
      target_compile_options(${target} PRIVATE ${OPENCV_CFLAGS})
      target_link_libraries(${target} PRIVATE ${OPENCV_PC_LIBS})
    endif()
    target_link_libraries(${target} PRIVATE Threads::Threads)
    add_test(NAME ${target} COMMAND ${target})
  endforeach()
else()
  message(STATUS "OpenCV not found: skipping test_benchmark_streams and test_full_models")
endif()
