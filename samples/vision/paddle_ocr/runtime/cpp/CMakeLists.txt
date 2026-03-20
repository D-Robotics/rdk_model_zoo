cmake_minimum_required(VERSION 3.0)
project(paddle_ocr)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# SOC macro
file(READ "/sys/class/boardinfo/soc_name" SOC_NAME_RAW)
string(TOUPPER "${SOC_NAME_RAW}" SOC_NAME_UPPER)
add_definitions(-DSOC_${SOC_NAME_UPPER})

include_directories(
    /usr/hobot/include
    /usr/include/hobot/dnn
    /usr/include/hobot
    ../../../../../utils/c_utils/inc
    inc/
)

link_directories(/usr/hobot/lib)

add_executable(paddle_ocr
    src/main.cpp
    src/paddle_ocr.cpp
    ../../../../../utils/c_utils/src/file_io.cpp
    ../../../../../utils/c_utils/src/preprocess.cpp
    ../../../../../utils/c_utils/src/postprocess.cpp
    ../../../../../utils/c_utils/src/visualize.cpp
)

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
find_library(CLIPPER_LIBRARY NAMES polyclipping)
find_package(Freetype REQUIRED)

SET(LINK_libs dnn hbucp gflags ${OpenCV_LIBS} ${FREETYPE_LIBRARIES})
target_include_directories(paddle_ocr PRIVATE ${FREETYPE_INCLUDE_DIRS})
target_link_libraries(paddle_ocr ${LINK_libs} ${CLIPPER_LIBRARY})
