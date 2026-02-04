cmake_minimum_required(VERSION 3.10)

project(ultralytics_yolo_classify)

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
set(DNN_PATH "/usr/include/dnn")
set(DNN_LIB_PATH "/usr/lib/")

include_directories(${DNN_PATH})
link_directories(${DNN_LIB_PATH})

# Add executable
add_executable(ultralytics_yolo_classify main.cc)

# Link libraries
target_link_libraries(ultralytics_yolo_classify
    ${OpenCV_LIBS}
    dnn
    pthread
    rt
    dl
)
