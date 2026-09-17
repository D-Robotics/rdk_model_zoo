cmake_minimum_required(VERSION 3.0)
project(paddle_ocr)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The canonical C++ implementation is the audited S100 PP-OCRv6 path.  Keep
# the source's S16 detector-output and F32 detector-output branches intact;
# the runtime chooses the branch from the tensor metadata at execution time.
# A missing board-info file is allowed during host-side configure checks.
if(EXISTS "/sys/class/boardinfo/soc_name")
    file(READ "/sys/class/boardinfo/soc_name" SOC_NAME_RAW)
    string(STRIP "${SOC_NAME_RAW}" SOC_NAME_STRIPPED)
    string(TOUPPER "${SOC_NAME_STRIPPED}" SOC_NAME_UPPER)
    if(SOC_NAME_UPPER STREQUAL "(NULL)")
        set(SOC_NAME_UPPER "")
    endif()
    if(SOC_NAME_UPPER)
        add_definitions(-DSOC_${SOC_NAME_UPPER})
    endif()
endif()

# The C++ utility sources are the existing S utility contract.  They remain
# outside this sample because they are consumed by other S C++ samples too.
set(RDK_S_UTILS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../platforms/s/utils/c_utils")

include_directories(
    /usr/hobot/include
    /usr/include/hobot/dnn
    /usr/include/hobot
    "${RDK_S_UTILS}/inc"
    "${CMAKE_CURRENT_SOURCE_DIR}/inc"
)

link_directories(/usr/hobot/lib)

add_executable(paddle_ocr
    src/main.cpp
    src/paddle_ocr.cpp
    "${RDK_S_UTILS}/src/file_io.cpp"
    "${RDK_S_UTILS}/src/preprocess.cpp"
    "${RDK_S_UTILS}/src/postprocess.cpp"
    "${RDK_S_UTILS}/src/visualize.cpp"
)

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
find_library(CLIPPER_LIBRARY NAMES polyclipping)
find_package(Freetype REQUIRED)

set(LINK_libs dnn hbucp gflags ${OpenCV_LIBS} ${FREETYPE_LIBRARIES})
target_include_directories(paddle_ocr PRIVATE ${FREETYPE_INCLUDE_DIRS})
target_link_libraries(paddle_ocr ${LINK_libs} ${CLIPPER_LIBRARY})
