# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.16)
project(bytetrack_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BYTETRACK_TARGET "" CACHE STRING "Explicit runtime target: host, x5, s100, s100p or s600")
set_property(CACHE BYTETRACK_TARGET PROPERTY STRINGS host x5 s100 s100p s600)
option(BYTETRACK_BUILD_TESTS "Build the host check/bench harnesses" OFF)

# tracking owns the BYTETracker port (detections -> tracks, C++17, no SDK);
# lapjv is the vendored lap 0.5.12 solver; parallel owns the fixed-slot
# runner used for per-track prediction and IoU rows. main/cli stay thin.
find_package(Threads REQUIRED)
add_library(bytetrack_core STATIC src/tracking.cpp src/lapjv.cpp src/parallel.cpp)
target_include_directories(bytetrack_core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/inc")
target_compile_options(bytetrack_core PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(bytetrack_core PUBLIC Threads::Threads)
# The vendored lap 0.5.12 solver is compiled verbatim; its one unused
# parameter warning is upstream's, not ours to edit away.
set_source_files_properties(src/lapjv.cpp PROPERTIES COMPILE_OPTIONS "-Wno-unused-parameter")

add_executable(bytetrack src/main.cpp src/cli.cpp)
target_link_libraries(bytetrack PRIVATE bytetrack_core)
target_include_directories(bytetrack PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/inc")
target_compile_options(bytetrack PRIVATE -Wall -Wextra -Wpedantic)

set(BT_IS_BOARD OFF)
if(NOT BYTETRACK_TARGET)
  message(FATAL_ERROR "Set BYTETRACK_TARGET explicitly: host, x5, s100, s100p or s600")
elseif(BYTETRACK_TARGET STREQUAL "host")
  # SDK-free build: replay only (tracker parity and quick comparisons).
  target_compile_definitions(bytetrack_core PRIVATE BYTETRACK_TARGET_HOST=1)
elseif(BYTETRACK_TARGET STREQUAL "x5")
  # X5: HB-DNN .bin models, packed NV12 640x640 (same macros the committed
  # YOLOv5 C++ runtime uses for its x5 build).
  set(BT_IS_BOARD ON)
  set(BT_BOARD_DEF BYTETRACK_TARGET_X5=1)
  set(BT_NAME_DEF BYTETRACK_TARGET_NAME="x5")
  set(BT_YOLO_DEFS YOLOV5_TARGET_X5=1 YOLOV5_TARGET_NAME="x5")
  set(BT_BOARD_LIBS dnn)
elseif(BYTETRACK_TARGET MATCHES "^(s100|s100p|s600)$")
  # S targets: UCP SDK .hbm models, split NV12 672x672. The SoC macro the S
  # backend and utils/c_utils expect is uppercase (SOC_S100/SOC_S100P/
  # SOC_S600) while the target name reaches the detector gate verbatim.
  string(TOUPPER "${BYTETRACK_TARGET}" BT_SOC)
  set(BT_IS_BOARD ON)
  set(BT_BOARD_DEF BYTETRACK_TARGET_S=1)
  set(BT_NAME_DEF BYTETRACK_TARGET_NAME="${BYTETRACK_TARGET}")
  set(BT_YOLO_DEFS YOLOV5_TARGET_S=1 YOLOV5_TARGET_NAME="${BYTETRACK_TARGET}" SOC_${BT_SOC}=1)
  set(BT_BOARD_LIBS dnn hbucp)
else()
  message(FATAL_ERROR "BYTETRACK_TARGET must be host, x5, s100, s100p or s600")
endif()

if(BT_IS_BOARD)
  # The board macro guards ByteTrackModel and the video/render paths; main
  # and cli compile those paths too, so they need the SAME definitions the
  # core gets. ByteTrackModel lives in tracking.cpp, i.e. in the core
  # library, so the YOLOv5 composition (detect.cpp + SDK libs) is linked
  # into the core too -- every core consumer (including the test harnesses)
  # pulls that object in.
  find_package(OpenCV REQUIRED)
  # Standard cache overrides for the board SDK locations (cpp-runtime-code
  # §5); the defaults are the on-image paths the committed YOLOv5 runtime
  # uses. X5 images expose the HB-DNN headers under /usr/include/hobot/dnn,
  # S images under /usr/include/hobot.
  set(BYTETRACK_HOBOT_LIB_DIR "/usr/hobot/lib" CACHE PATH "Board DNN/UCP SDK library directory")
  set(BYTETRACK_HOBOT_INCLUDE_DIR "/usr/hobot/include" CACHE PATH "Board SDK header root")
  set(BYTETRACK_SYSTEM_HOBOT_INCLUDE_DIR "/usr/include/hobot" CACHE PATH "System hobot header root (dnn headers live below it)")
  set(BT_YOLO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../yolov5/runtime/cpp")
  set(BT_SDK_INC ${OpenCV_INCLUDE_DIRS} ${BYTETRACK_HOBOT_INCLUDE_DIR}
                 "${BYTETRACK_SYSTEM_HOBOT_INCLUDE_DIR}/dnn")
  target_compile_definitions(bytetrack_core PRIVATE ${BT_BOARD_DEF} ${BT_NAME_DEF} ${BT_YOLO_DEFS})
  target_compile_definitions(bytetrack PRIVATE ${BT_BOARD_DEF} ${BT_NAME_DEF} ${BT_YOLO_DEFS})
  target_sources(bytetrack_core PRIVATE "${BT_YOLO_DIR}/src/detect.cpp")
  target_include_directories(bytetrack_core PRIVATE "${BT_YOLO_DIR}/inc" ${BT_SDK_INC})
  target_include_directories(bytetrack PRIVATE "${BT_YOLO_DIR}/inc" ${BT_SDK_INC})
  target_link_directories(bytetrack_core PRIVATE "${BYTETRACK_HOBOT_LIB_DIR}")
  target_link_directories(bytetrack PRIVATE "${BYTETRACK_HOBOT_LIB_DIR}")
  target_link_libraries(bytetrack_core PRIVATE ${BT_BOARD_LIBS} ${OpenCV_LIBS})
  target_link_libraries(bytetrack PRIVATE ${BT_BOARD_LIBS} ${OpenCV_LIBS})
  if(BYTETRACK_TARGET MATCHES "^(s100|s100p|s600)$")
    # The S backend's c_utils helpers (preprocess/postprocess/nn_math), with
    # their headers under the system hobot root on the S images.
    set(BT_C_UTILS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../utils/c_utils")
    target_include_directories(bytetrack_core PRIVATE "${BYTETRACK_SYSTEM_HOBOT_INCLUDE_DIR}"
                                                 "${BT_C_UTILS}/inc")
    target_include_directories(bytetrack PRIVATE "${BYTETRACK_SYSTEM_HOBOT_INCLUDE_DIR}" "${BT_C_UTILS}/inc")
    target_sources(bytetrack_core PRIVATE "${BT_C_UTILS}/src/preprocess.cpp"
                                          "${BT_C_UTILS}/src/postprocess.cpp"
                                          "${BT_C_UTILS}/src/nn_math.cpp")
  endif()
endif()

if(BYTETRACK_BUILD_TESTS)
  # Behaviour checks and the timing harness live with the sample tests; the
  # production CLI deliberately carries no benchmark flags. Both reuse the
  # production CLI's replay-file reader.
  add_executable(bytetrack_checks "../../tests/cpp/bytetrack_checks.cpp" src/cli.cpp)
  target_link_libraries(bytetrack_checks PRIVATE bytetrack_core)
  target_include_directories(bytetrack_checks PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/inc")
  target_compile_options(bytetrack_checks PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(bytetrack_bench "../../tests/cpp/bytetrack_bench.cpp" src/cli.cpp)
  target_link_libraries(bytetrack_bench PRIVATE bytetrack_core)
  target_include_directories(bytetrack_bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/inc")
  target_compile_options(bytetrack_bench PRIVATE -Wall -Wextra -Wpedantic)

  # Component goldens compile the production tracking.cpp translation unit
  # directly (to reach the internal IoU/assignment primitives), so it links
  # only the solver and the parallel runner.
  add_executable(bytetrack_components "../../tests/cpp/bytetrack_components.cpp"
                                    src/lapjv.cpp src/parallel.cpp)
  target_include_directories(bytetrack_components PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/inc"
                                                          "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_compile_options(bytetrack_components PRIVATE -Wall -Wextra -Wpedantic)

  if(BT_IS_BOARD)
    # Evidence tool: dump the detections the composition feeds the tracker.
    # Its translation unit sees ByteTrackModel, so it needs the same board
    # and YOLOv5 definitions the executable gets.
    add_executable(bytetrack_detector_dump "../../tests/cpp/bytetrack_detector_dump.cpp")
    target_link_libraries(bytetrack_detector_dump PRIVATE bytetrack_core)
    target_compile_definitions(bytetrack_detector_dump PRIVATE ${BT_BOARD_DEF} ${BT_NAME_DEF} ${BT_YOLO_DEFS})
    target_include_directories(bytetrack_detector_dump PRIVATE
                               "${CMAKE_CURRENT_SOURCE_DIR}/inc" "${BT_YOLO_DIR}/inc" ${BT_SDK_INC})
    target_link_directories(bytetrack_detector_dump PRIVATE /usr/hobot/lib)
    target_link_libraries(bytetrack_detector_dump PRIVATE ${BT_BOARD_LIBS} ${OpenCV_LIBS})
    target_compile_options(bytetrack_detector_dump PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endif()
