# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.16)
project(yolov5_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(YOLOV5_TARGET "x5" CACHE STRING "Explicit runtime target: x5, s100, s100p or s600")
set_property(CACHE YOLOV5_TARGET PROPERTY STRINGS x5 s100 s100p s600)

add_executable(yolov5_cpp src/cli_main.cpp src/yolov5_decode.cpp src/yolov5_visualize.cpp
                          src/yolov5_gate.cpp src/yolov5_dump.cpp)
target_include_directories(yolov5_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_options(yolov5_cpp PRIVATE -Wall -Wextra -Wpedantic)

if(YOLOV5_TARGET STREQUAL "x5")
  target_sources(yolov5_cpp PRIVATE src/x5_adapter.cpp)
  # The compiled target identity lets the binary refuse a mismatching --target.
  target_compile_definitions(yolov5_cpp PRIVATE YOLOV5_TARGET_X5=1 YOLOV5_TARGET_NAME="x5")
  find_package(OpenCV REQUIRED)
  target_include_directories(yolov5_cpp PRIVATE ${OpenCV_INCLUDE_DIRS} /usr/hobot/include /usr/include/hobot/dnn)
  target_link_directories(yolov5_cpp PRIVATE /usr/hobot/lib)
  target_link_libraries(yolov5_cpp PRIVATE dnn ${OpenCV_LIBS})
elseif(YOLOV5_TARGET STREQUAL "s100" OR YOLOV5_TARGET STREQUAL "s100p" OR YOLOV5_TARGET STREQUAL "s600")
  target_sources(yolov5_cpp PRIVATE src/s_adapter.cpp)
  target_compile_definitions(yolov5_cpp PRIVATE YOLOV5_TARGET_S=1 YOLOV5_TARGET_NAME="${YOLOV5_TARGET}")
  if(YOLOV5_TARGET STREQUAL "s600")
    target_compile_definitions(yolov5_cpp PRIVATE SOC_S600=1)
  elseif(YOLOV5_TARGET STREQUAL "s100")
    target_compile_definitions(yolov5_cpp PRIVATE SOC_S100=1)
  else()
    target_compile_definitions(yolov5_cpp PRIVATE SOC_S100P=1)
  endif()
  find_package(OpenCV REQUIRED)
  target_include_directories(yolov5_cpp PRIVATE ${OpenCV_INCLUDE_DIRS} /usr/hobot/include /usr/include/hobot/dnn
                             "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../utils/c_utils/inc")
  target_link_directories(yolov5_cpp PRIVATE /usr/hobot/lib)
  target_sources(yolov5_cpp PRIVATE
                 "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../utils/c_utils/src/preprocess.cpp"
                 "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../utils/c_utils/src/postprocess.cpp"
                 "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../utils/c_utils/src/nn_math.cpp")
  target_link_libraries(yolov5_cpp PRIVATE dnn hbucp ${OpenCV_LIBS})
else()
  message(FATAL_ERROR "YOLOV5_TARGET must be x5, s100, s100p or s600")
endif()
