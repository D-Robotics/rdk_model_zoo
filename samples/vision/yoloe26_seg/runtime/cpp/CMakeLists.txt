cmake_minimum_required(VERSION 3.16)
project(yoloe26seg LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
find_package(OpenCV REQUIRED)
find_package(gflags QUIET)
find_path(UCP_INCLUDE_DIR hobot/hb_ucp.h PATHS /usr/include /usr/hobot/include REQUIRED)
find_library(DNN_LIBRARY dnn PATHS /usr/hobot/lib /usr/lib REQUIRED)
find_library(UCP_LIBRARY NAMES hbucp ucp PATHS /usr/hobot/lib /usr/lib REQUIRED)
if(NOT TARGET gflags::gflags)
    find_library(GFLAGS_LIBRARY gflags PATHS /usr/lib /usr/local/lib REQUIRED)
endif()

set(UTILS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../utils/c_utils")
add_executable(yoloe26seg
    src/main.cpp
    src/yoloe26seg.cpp)
target_include_directories(yoloe26seg PRIVATE
    inc
    "${UTILS_DIR}/inc"
    ${UCP_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS})
target_compile_definitions(yoloe26seg PRIVATE
    YOLOE26_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(yoloe26seg PRIVATE
    ${DNN_LIBRARY}
    ${UCP_LIBRARY}
    ${OpenCV_LIBS})
if(TARGET gflags::gflags)
    target_link_libraries(yoloe26seg PRIVATE gflags::gflags)
else()
    target_link_libraries(yoloe26seg PRIVATE ${GFLAGS_LIBRARY})
endif()

option(YOLOE26_BUILD_CPP_CONTRACT_TEST
       "Build the optional board-backed C++ contract regression" OFF)
if(YOLOE26_BUILD_CPP_CONTRACT_TEST)
    add_executable(yoloe26seg_cpp_contract_test
        "${CMAKE_CURRENT_SOURCE_DIR}/../../evaluator/cpp/test_cpp_contract.cpp"
        src/yoloe26seg.cpp)
    target_include_directories(yoloe26seg_cpp_contract_test PRIVATE
        inc
        "${UTILS_DIR}/inc"
        ${UCP_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS})
    target_compile_definitions(yoloe26seg_cpp_contract_test PRIVATE
        YOLOE26_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(yoloe26seg_cpp_contract_test PRIVATE
        ${DNN_LIBRARY}
        ${UCP_LIBRARY}
        ${OpenCV_LIBS})
endif()
