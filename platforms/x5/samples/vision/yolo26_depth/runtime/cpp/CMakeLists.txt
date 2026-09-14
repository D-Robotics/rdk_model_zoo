cmake_minimum_required(VERSION 3.10)
project(yolo26_depth LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wl,-unresolved-symbols=ignore-in-shared-libs")

find_package(OpenCV REQUIRED)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc /usr/include/dnn)
link_directories(/usr/lib)

add_executable(yolo26_depth
  src/main.cpp
  src/yolo26_depth.cpp
)
target_link_libraries(yolo26_depth PRIVATE ${OpenCV_LIBS} dnn pthread rt dl)
