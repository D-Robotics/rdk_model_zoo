# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.16)
project(unetmobilenet_cpp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(UNETMOBILENET_TARGET "" CACHE STRING "Explicit target: s100 or s600")
if(NOT UNETMOBILENET_TARGET STREQUAL "s100" AND NOT UNETMOBILENET_TARGET STREQUAL "s600")
  message(FATAL_ERROR "Set -DUNETMOBILENET_TARGET=s100 or s600; no implicit SoC fallback")
endif()
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
add_executable(unetmobilenet_cpp src/main.cpp src/model_runner.cpp src/unetmobilenet.cpp
                               src/tensor_contract.cpp src/visualization.cpp)
target_compile_definitions(unetmobilenet_cpp PRIVATE UNETMOBILENET_TARGET_NAME="${UNETMOBILENET_TARGET}")
target_compile_options(unetmobilenet_cpp PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(unetmobilenet_cpp PRIVATE inc ${OpenCV_INCLUDE_DIRS}
                           /usr/hobot/include /usr/include/hobot /usr/include/hobot/dnn)
target_link_directories(unetmobilenet_cpp PRIVATE /usr/hobot/lib)
target_link_libraries(unetmobilenet_cpp PRIVATE dnn hbucp ${OpenCV_LIBS})
