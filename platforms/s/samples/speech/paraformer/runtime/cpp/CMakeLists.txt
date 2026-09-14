cmake_minimum_required(VERSION 3.10)
project(paraformer LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    /usr/hobot/include
    /usr/include/hobot
    /usr/include/hobot/dnn
    inc
)
link_directories(/usr/hobot/lib)

add_executable(paraformer
    src/main.cpp
    src/paraformer.cpp
)
target_link_libraries(paraformer PRIVATE dnn hbucp)
