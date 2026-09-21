cmake_minimum_required(VERSION 3.10)

project(ultralytics_yolo_segment)

# C++11 standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
# libdnn.so depends on system software dynamic link library
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wl,-unresolved-symbols=ignore-in-shared-libs")
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# D-Robotics BPU DNN library
# X5 keeps the DNN headers under /usr/include/dnn; S-series boards keep
# them under /usr/include/hobot/dnn with the library in /usr/hobot/lib.
set(DNN_PATH "/usr/include/dnn")
set(DNN_LIB_PATH "/usr/lib/")
set(DNN_EXTRA_INCLUDES "")
set(DNN_EXTRA_LIBS "")
if (NOT EXISTS "${DNN_PATH}/hb_dnn.h")
    set(DNN_PATH "/usr/include/hobot")
    set(DNN_LIB_PATH "/usr/hobot/lib")
    # Mirror the include roots used by the platform S samples: the UCP
    # headers sit at /usr/include/hobot, the DNN headers under .../dnn, and
    # some images additionally expose SDK headers via /usr/hobot/include.
    set(DNN_EXTRA_INCLUDES "/usr/include/hobot/dnn" "/usr/hobot/include")
    set(DNN_EXTRA_LIBS hbucp)
endif()

include_directories(${DNN_PATH} ${DNN_EXTRA_INCLUDES})
link_directories(${DNN_LIB_PATH})

# Add executable (with the shared helpers in ../common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
add_executable(ultralytics_yolo_segment
    main.cc
    ../common/dnn_io.cc
    ../common/nv12_geometry.cc
)

# Link libraries
target_link_libraries(ultralytics_yolo_segment
    ${OpenCV_LIBS}
    dnn
    ${DNN_EXTRA_LIBS}
    pthread
    rt
    dl
)
