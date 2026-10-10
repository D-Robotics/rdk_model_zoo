cmake_minimum_required(VERSION 3.10)
project(paddle_ocr)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Board target. "auto" resolves the SoC only when configuring natively on the
# board; cross compilation must pass an explicit target and never reads the
# build host's SoC identity. This runtime ships S100 and S600 artifacts only.
set(RDK_TARGET "auto" CACHE STRING "Board target: s100|s600|auto (auto only for native on-board configuration)")
if(RDK_TARGET STREQUAL "auto")
    if(CMAKE_CROSSCOMPILING)
        message(FATAL_ERROR "RDK_TARGET=auto would read the build host SoC identity while cross-compiling; pass an explicit target: -DRDK_TARGET=s100 or -DRDK_TARGET=s600")
    endif()
    file(READ "/sys/class/boardinfo/soc_name" SOC_NAME_RAW)
    string(STRIP "${SOC_NAME_RAW}" SOC_NAME_RAW)
    string(TOUPPER "${SOC_NAME_RAW}" SOC_NAME_UPPER)
else()
    string(TOUPPER "${RDK_TARGET}" SOC_NAME_UPPER)
endif()
if(NOT SOC_NAME_UPPER MATCHES "^(S100|S600)$")
    message(FATAL_ERROR "Unsupported RDK board target '${RDK_TARGET}' (resolved '${SOC_NAME_UPPER}'): this runtime supports s100 and s600")
endif()
add_definitions(-DSOC_${SOC_NAME_UPPER})

# The C++ utility sources are the existing S utility contract.  They remain
# outside this sample because they are consumed by other S C++ samples too.
set(RDK_S_UTILS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../utils/c_utils")

include_directories(
    /usr/hobot/include
    /usr/include/hobot/dnn
    /usr/include/hobot
    "${RDK_S_UTILS}/inc"
    "${CMAKE_CURRENT_SOURCE_DIR}/inc"
)

link_directories(/usr/hobot/lib)

# The canonical C++ implementation is the audited S100 PP-OCRv6 path.  Keep
# the source's S16 detector-output and F32 detector-output branches intact;
# the runtime chooses the branch from the tensor metadata at execution time.
add_executable(paddle_ocr
    src/main.cpp
    src/ocr.cpp
    src/cli.cpp
    "${RDK_S_UTILS}/src/file_io.cpp"
    "${RDK_S_UTILS}/src/preprocess.cpp"
    "${RDK_S_UTILS}/src/postprocess.cpp"
    "${RDK_S_UTILS}/src/visualize.cpp"
)

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
find_library(CLIPPER_LIBRARY NAMES polyclipping)
find_package(Freetype REQUIRED)

set(LINK_libs dnn hbucp ${OpenCV_LIBS} ${FREETYPE_LIBRARIES} ${CLIPPER_LIBRARY})
target_include_directories(paddle_ocr PRIVATE ${FREETYPE_INCLUDE_DIRS})
target_link_libraries(paddle_ocr ${LINK_libs})
