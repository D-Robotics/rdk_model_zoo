cmake_minimum_required(VERSION 3.5)
project(ultralytics_yolo_segment)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# BPU DNN library
include_directories(/usr/include/horizon/dnn)
link_directories(/usr/lib/hbmedia)

# Source files
set(SOURCES main.cc)

# Executable
add_executable(ultralytics_yolo_segment ${SOURCES})

# Link libraries
target_link_libraries(ultralytics_yolo_segment
    ${OpenCV_LIBS}
    dnn
)

# Set build type to Release for optimization
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
