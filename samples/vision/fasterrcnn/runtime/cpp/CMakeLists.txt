# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.16)
project(fasterrcnn_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(FRCNN_TARGET "" CACHE STRING "Explicit runtime target: host, x5, s100, s100p or s600")
set_property(CACHE FRCNN_TARGET PROPERTY STRINGS host x5 s100 s100p s600)

# The shared CPU ByteTrack core is reused without modification: with no
# BYTETRACK_TARGET_* macro defined (this sample never defines one) its
# tracking/lapjv/parallel sources compile SDK-free, and the person filter
# lives at the detector boundary in detect.cpp.
set(FRCNN_BYTETRACK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../bytetrack/runtime/cpp")

find_package(Threads REQUIRED)
add_library(fasterrcnn_core STATIC
  src/detect.cpp
  "${FRCNN_BYTETRACK_DIR}/src/tracking.cpp"
  "${FRCNN_BYTETRACK_DIR}/src/lapjv.cpp"
  "${FRCNN_BYTETRACK_DIR}/src/parallel.cpp")
target_include_directories(fasterrcnn_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/inc"
                                                  "${FRCNN_BYTETRACK_DIR}/inc")
# -ffp-contract=off keeps the C++ arithmetic free of FMA contraction so the
# host/board builds stay bit-comparable with the Python runtime.
target_compile_options(fasterrcnn_core PRIVATE -Wall -Wextra -ffp-contract=off)
target_link_libraries(fasterrcnn_core PUBLIC Threads::Threads)
# The vendored lap 0.5.12 solver is compiled verbatim; its one unused
# parameter warning is upstream's, not ours to edit away.
set_source_files_properties("${FRCNN_BYTETRACK_DIR}/src/lapjv.cpp"
                            PROPERTIES COMPILE_OPTIONS "-Wno-unused-parameter")

add_executable(fasterrcnn src/main.cpp src/cli.cpp)
target_link_libraries(fasterrcnn PRIVATE fasterrcnn_core)
target_compile_options(fasterrcnn PRIVATE -Wall -Wextra -ffp-contract=off)

# Default model/label/test-data paths resolve from the sample root next to
# this runtime (model/<target>/..., model/coco91_labels.txt, test_data/...);
# detect.cpp's descriptor_model_dir is compiled inside the core, so both
# targets need the definition.
foreach(frcnn_tgt fasterrcnn_core fasterrcnn)
  target_compile_definitions(${frcnn_tgt} PRIVATE
    FRCNN_SAMPLE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../..")
endforeach()

set(FRCNN_IS_BOARD OFF)
if(NOT FRCNN_TARGET)
  message(FATAL_ERROR "Set FRCNN_TARGET explicitly: host, x5, s100, s100p or s600")
elseif(FRCNN_TARGET STREQUAL "host")
  # SDK-free build: --help/--list-models/--dry-run plus the board-free math
  # stages exercised by tests/cpp/run_host_checks.py.
elseif(FRCNN_TARGET STREQUAL "x5")
  # X5: HB-DNN .bin models, packed NV12 320x320 input, float pooled PartB
  # (featuremap input; the graph quantizes on chip).
  set(FRCNN_IS_BOARD ON)
  set(FRCNN_BOARD_DEFS FRCNN_TARGET_X5=1 FRCNN_TARGET_NAME="x5")
  set(FRCNN_BOARD_LIBS dnn)
elseif(FRCNN_TARGET MATCHES "^(s100|s100p|s600)$")
  # S targets: UCP SDK .hbm models, split NV12 320x320, packed float32 ROI
  # bank. The SoC macro the S backend expects is uppercase (SOC_S100/
  # SOC_S100P/SOC_S600, selecting the 64-byte S600 stride alignment) while
  # the target name reaches the runtime verbatim.
  string(TOUPPER "${FRCNN_TARGET}" FRCNN_SOC)
  set(FRCNN_IS_BOARD ON)
  set(FRCNN_BOARD_DEFS FRCNN_TARGET_S=1 FRCNN_TARGET_NAME="${FRCNN_TARGET}"
                       SOC_${FRCNN_SOC}=1)
  set(FRCNN_BOARD_LIBS dnn hbucp)
else()
  message(FATAL_ERROR "FRCNN_TARGET must be host, x5, s100, s100p or s600")
endif()

if(FRCNN_IS_BOARD)
  # The board backends (detect.cpp's X5/S sessions) and the CLI's image/video
  # paths need OpenCV plus the board SDK headers; main.cpp and cli.cpp see
  # the same target macros as detect.cpp. Standard cache overrides for the
  # SDK locations (cpp-runtime-code §5); defaults are the on-image paths
  # verified on the 2026-10-12 board runs: S images put libdnn/libhbucp in
  # /usr/hobot/lib with the HB-DNN headers under /usr/include/hobot, while
  # X5 images put libdnn in /usr/lib with the HB-DNN headers under
  # /usr/include/dnn.
  find_package(OpenCV REQUIRED)
  if(FRCNN_TARGET STREQUAL "x5")
    set(FRCNN_HOBOT_LIB_DIR "/usr/lib" CACHE PATH "Board DNN/UCP SDK library directory")
    set(FRCNN_HOBOT_INCLUDE_DIR "/usr/include" CACHE PATH "Board SDK header root")
  else()
    set(FRCNN_HOBOT_LIB_DIR "/usr/hobot/lib" CACHE PATH "Board DNN/UCP SDK library directory")
    set(FRCNN_HOBOT_INCLUDE_DIR "/usr/hobot/include" CACHE PATH "Board SDK header root")
  endif()
  set(FRCNN_SYSTEM_HOBOT_INCLUDE_DIR "/usr/include/hobot" CACHE PATH "System hobot header root")
  set(FRCNN_SDK_INC ${OpenCV_INCLUDE_DIRS} "${FRCNN_HOBOT_INCLUDE_DIR}"
                    "${FRCNN_SYSTEM_HOBOT_INCLUDE_DIR}/dnn")
  if(FRCNN_TARGET MATCHES "^(s100|s100p|s600)$")
    # The S backend includes "hobot/dnn/hb_dnn.h" relative to the system
    # hobot root on the S images.
    list(APPEND FRCNN_SDK_INC "${FRCNN_SYSTEM_HOBOT_INCLUDE_DIR}")
  endif()
  foreach(frcnn_tgt fasterrcnn_core fasterrcnn)
    target_compile_definitions(${frcnn_tgt} PRIVATE ${FRCNN_BOARD_DEFS})
    target_include_directories(${frcnn_tgt} PRIVATE ${FRCNN_SDK_INC})
    target_link_directories(${frcnn_tgt} PRIVATE "${FRCNN_HOBOT_LIB_DIR}")
    target_link_libraries(${frcnn_tgt} PRIVATE ${FRCNN_BOARD_LIBS} ${OpenCV_LIBS})
  endforeach()
endif()
