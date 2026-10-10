# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.16)
project(ultralytics_yolo_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(YOLO_TARGET "auto" CACHE STRING "Explicit runtime target: auto, x5, s100, s100p or s600")
set_property(CACHE YOLO_TARGET PROPERTY STRINGS auto x5 s100 s100p s600)

# The consolidated layout: main constructs the task model and reports, cli
# owns options/image loading/render/benchmark, one task source per task, and
# the shared board backend (backend.cpp) owns the SDK lifecycle. The X5/S
# stack is selected by the header probe in inc/backend.hpp; the target below
# only selects which SDK include roots are visible and never reads the host
# SoC.
add_executable(ultralytics_yolo_cpp
    src/main.cpp
    src/cli.cpp
    src/backend.cpp
    src/detect.cpp
    src/segment.cpp
    src/pose.cpp
    src/classify.cpp
    src/obb.cpp)
target_include_directories(ultralytics_yolo_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/inc")
target_compile_options(ultralytics_yolo_cpp PRIVATE -Wall -Wextra)

find_package(OpenCV REQUIRED)

# X5 keeps the DNN headers under /usr/include/dnn; S-series boards keep them
# under /usr/include/hobot/dnn with the library in /usr/hobot/lib (mirrors
# the released per-task CMakeLists).
set(DNN_INCLUDES "")
set(DNN_LIB_DIRS "")
set(DNN_LIBS "")
if(YOLO_TARGET STREQUAL "x5" OR YOLO_TARGET STREQUAL "auto")
  if(EXISTS "/usr/include/dnn/hb_dnn.h")
    set(DNN_INCLUDES /usr/include/dnn)
    set(DNN_LIB_DIRS /usr/lib)
    set(DNN_LIBS dnn)
  endif()
endif()
if(NOT DNN_LIBS AND (YOLO_TARGET STREQUAL "s100" OR YOLO_TARGET STREQUAL "s100p"
                     OR YOLO_TARGET STREQUAL "s600" OR YOLO_TARGET STREQUAL "auto"))
  if(EXISTS "/usr/include/hobot/dnn/hb_dnn.h")
    # UCP headers sit at /usr/include/hobot, DNN headers under .../dnn, and
    # some images additionally expose SDK headers via /usr/hobot/include.
    set(DNN_INCLUDES /usr/include/hobot /usr/include/hobot/dnn /usr/hobot/include)
    set(DNN_LIB_DIRS /usr/hobot/lib)
    set(DNN_LIBS dnn hbucp)
  endif()
endif()

if(NOT DNN_LIBS)
  if(YOLO_TARGET STREQUAL "auto")
    message(FATAL_ERROR "No DNN SDK headers found (looked for /usr/include/dnn/hb_dnn.h and /usr/include/hobot/dnn/hb_dnn.h); pass -DYOLO_TARGET=x5|s100|s100p|s600 on the board")
  else()
    message(FATAL_ERROR "YOLO_TARGET ${YOLO_TARGET} requires the board DNN SDK headers (/usr/include/dnn or /usr/include/hobot)")
  endif()
endif()

target_include_directories(ultralytics_yolo_cpp PRIVATE ${OpenCV_INCLUDE_DIRS} ${DNN_INCLUDES})
target_link_directories(ultralytics_yolo_cpp PRIVATE ${DNN_LIB_DIRS})
target_link_libraries(ultralytics_yolo_cpp PRIVATE ${DNN_LIBS} ${OpenCV_LIBS} pthread rt dl)
