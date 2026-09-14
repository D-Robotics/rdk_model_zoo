cmake_minimum_required(VERSION 3.10)
project(himloco_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wl,-unresolved-symbols=ignore-in-shared-libs")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc /usr/include)
link_directories(/usr/lib)
find_library(GFLAGS_LIBRARY NAMES gflags)
if(NOT GFLAGS_LIBRARY)
  message(FATAL_ERROR "libgflags is required; install libgflags-dev")
endif()

add_executable(himloco_cpp
  src/main.cpp
  src/himloco.cc
)
target_link_libraries(himloco_cpp PRIVATE dnn pthread rt dl ${GFLAGS_LIBRARY})
