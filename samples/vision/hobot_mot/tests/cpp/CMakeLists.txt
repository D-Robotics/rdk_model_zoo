# Board-only build of the parity reference (parity_reference.cpp): it links
# the prebuilt ARM64 iou_based_mot core, so it is never built by host
# validation. Build on a board after model/download.sh:
#   cmake -B build -S . && cmake --build build
cmake_minimum_required(VERSION 3.16)
project(hobot_mot_parity_reference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(HOBOT_MOT_SOURCE_ROOT "" CACHE PATH "Extracted D-Robotics/mot source tree")
if(NOT HOBOT_MOT_SOURCE_ROOT)
  set(HOBOT_MOT_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../model/vendor/mot-0120205146d88c3474bfc2936522e349e169fa8d")
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
)
link_directories(
  ${HOBOT_MOT_SOURCE_ROOT}/deps/iou_based_mot/lib
  ${HOBOT_MOT_SOURCE_ROOT}/deps/hobotlog/lib
  ${HOBOT_MOT_SOURCE_ROOT}/deps/jsoncpp/lib
  ${HOBOT_MOT_SOURCE_ROOT}/deps/hobot/lib
)

add_executable(parity_reference
  parity_reference.cpp
  ${HOBOT_MOT_SOURCE_ROOT}/src/hobot_mot.cpp
  ${HOBOT_MOT_SOURCE_ROOT}/src/MOTMethod.cpp
  ${HOBOT_MOT_SOURCE_ROOT}/src/MOT/IOU2.cpp
)
target_link_libraries(parity_reference
  iou_based_mot hobotlog hobotsdk hobotthread jsoncpp rt dl pthread)
