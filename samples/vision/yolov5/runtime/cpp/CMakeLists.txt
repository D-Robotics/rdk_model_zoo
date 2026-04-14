cmake_minimum_required(VERSION 2.8)

project(rdk_yolov8_detect)


# 设置OpenCV包
find_package(OpenCV REQUIRED)

# # 添加可执行文件
# add_executable(main main.cc)
# # 链接OpenCV库
# target_link_libraries(main ${OpenCV_LIBS})



# libdnn.so depends on system software dynamic link library, use -Wl,-unresolved-symbols=ignore-in-shared-libs to shield during compilation
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wl,-unresolved-symbols=ignore-in-shared-libs")

set(CMAKE_CXX_FLAGS_DEBUG " -Wall -Werror -g -O0 ")
set(CMAKE_C_FLAGS_DEBUG " -Wall -Werror -g -O0 ")
set(CMAKE_CXX_FLAGS_RELEASE " -Wall -Werror -O3 ")
set(CMAKE_C_FLAGS_RELEASE " -Wall -Werror -O3 ")
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
# define dnn lib path
set(DNN_PATH "/usr/include/dnn")

set(DNN_LIB_PATH "/usr/lib/")

include_directories(${DNN_PATH})
link_directories(${DNN_LIB_PATH})

add_executable(main main.cc)
target_link_libraries(main
                      ${OpenCV_LIBS}
                      dnn
                      pthread
                      rt
                      dl)