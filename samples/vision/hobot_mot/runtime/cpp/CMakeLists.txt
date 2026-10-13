# HobotMot (IOU 2.0) C++ runtime and C ABI binding.
#
# Builds against the pinned D-Robotics/mot source prepared by
# samples/vision/hobot_mot/model/download.sh (wrappers are compiled here; the
# prebuilt ARM64 iou_based_mot core and its dependency archives are linked
# from the prepared tree). The tracker is CPU-only: no BPU model, no OpenCV,
# and one identical library for every supported board, so the build does not
# gate on board identity. RDK_TARGET records the intended board for
# bookkeeping; auto (the default) is only allowed on native builds because a
# cross build would read the build host's identity.
cmake_minimum_required(VERSION 3.16)
project(hobot_mot_sample CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(RDK_TARGET "auto" CACHE STRING "Intended board: auto (native only), x5, s100, s100p or s600; the CPU-only library is identical for all of them")

# Prepared upstream source tree (see model/download.sh).
set(HOBOT_MOT_SOURCE_ROOT "" CACHE PATH "Extracted D-Robotics/mot source tree")

if(NOT HOBOT_MOT_SOURCE_ROOT)
  # Default next to the model/ directory that prepared it.
  set(HOBOT_MOT_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../model/vendor/mot-0120205146d88c3474bfc2936522e349e169fa8d")
endif()

string(TOLOWER "${RDK_TARGET}" RDK_TARGET)
string(STRIP "${RDK_TARGET}" RDK_TARGET)
if(RDK_TARGET STREQUAL "auto")
  if(CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR "RDK_TARGET=auto is not allowed in cross builds; pass an explicit target (x5, s100, s100p, s600)")
  endif()
  # Informational only: the produced CPU library does not depend on it.
  foreach(soc_file /sys/class/boardinfo/soc_name /sys/class/socinfo/soc_name)
    if(EXISTS "${soc_file}")
      file(READ "${soc_file}" soc_name)
      string(TOLOWER "${soc_name}" soc_name)
      string(STRIP "${soc_name}" soc_name)
      if(soc_name MATCHES "x5|s100p|s100|s600")
        message(STATUS "HobotMot sample: native build on ${soc_name}")
      else()
        message(STATUS "HobotMot sample: unrecognized SoC '${soc_name}'; CPU-only build continues (pass -DRDK_TARGET=<x5|s100|s100p|s600> to record the board)")
      endif()
      break()
    endif()
  endforeach()
else()
  if(NOT RDK_TARGET MATCHES "^(x5|s100|s100p|s600)$")
    message(FATAL_ERROR "Unsupported RDK_TARGET '${RDK_TARGET}'; use x5, s100, s100p or s600")
  endif()
  message(STATUS "HobotMot sample target: ${RDK_TARGET} (identical CPU-only build for all targets)")
endif()

if(NOT EXISTS "${HOBOT_MOT_SOURCE_ROOT}/src/hobot_mot.cpp")
  message(FATAL_ERROR
    "Upstream source not found under ${HOBOT_MOT_SOURCE_ROOT}. "
    "Run samples/vision/hobot_mot/model/download.sh first or pass "
    "-DHOBOT_MOT_SOURCE_ROOT=<prepared mot tree>.")
endif()

add_definitions(-DHR_POSIX -DENABLE_IOU2)
add_compile_options(-Wall -Wextra)

include_directories(
  ${HOBOT_MOT_SOURCE_ROOT}
  ${HOBOT_MOT_SOURCE_ROOT}/include
  ${HOBOT_MOT_SOURCE_ROOT}/deps/iou_based_mot/include/include
  ${HOBOT_MOT_SOURCE_ROOT}/deps/hobotlog/include
  ${HOBOT_MOT_SOURCE_ROOT}/deps/jsoncpp/include
  ${HOBOT_MOT_SOURCE_ROOT}/deps/hobot/include
  ${CMAKE_CURRENT_SOURCE_DIR}/inc
)
link_directories(
  ${HOBOT_MOT_SOURCE_ROOT}/deps/iou_based_mot/lib
  ${HOBOT_MOT_SOURCE_ROOT}/deps/hobotlog/lib
  ${HOBOT_MOT_SOURCE_ROOT}/deps/jsoncpp/lib
  ${HOBOT_MOT_SOURCE_ROOT}/deps/hobot/lib
)

set(UPSTREAM_SOURCES
  ${HOBOT_MOT_SOURCE_ROOT}/src/hobot_mot.cpp
  ${HOBOT_MOT_SOURCE_ROOT}/src/MOTMethod.cpp
  ${HOBOT_MOT_SOURCE_ROOT}/src/MOT/IOU2.cpp
)

# Shared C ABI consumed by runtime/python (ctypes) and by tools that need the
# tracker without linking C++ types. Owns the complete C boundary.
add_library(hobot_mot_capi SHARED
  src/mot_c_api.cpp
  src/track.cpp
  ${UPSTREAM_SOURCES}
)
target_compile_definitions(hobot_mot_capi PRIVATE
  HOBOT_MOT_SAMPLE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/../..")
target_link_libraries(hobot_mot_capi PRIVATE
  iou_based_mot hobotlog hobotsdk hobotthread jsoncpp rt dl pthread)

# Replay executable: JSONL detections in, tracked identities out.
add_executable(hobot_mot_replay
  src/main.cpp
  src/cli.cpp
  src/track.cpp
  ${UPSTREAM_SOURCES}
)
target_compile_definitions(hobot_mot_replay PRIVATE
  HOBOT_MOT_SAMPLE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/../..")
target_link_libraries(hobot_mot_replay PRIVATE
  iou_based_mot hobotlog hobotsdk hobotthread jsoncpp rt dl pthread)
