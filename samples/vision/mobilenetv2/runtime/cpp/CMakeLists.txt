cmake_minimum_required(VERSION 3.10)
project(mobilenetv2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Board target. "auto" resolves the SoC only when configuring natively on the
# board; cross compilation must pass an explicit target and never reads the
# build host's SoC identity. This runtime ships S100 and S600 artifacts only
# (the launcher's identity gate rejects S100P before the build is reached).
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

# Include directories for header file search
include_directories(
    /usr/hobot/include
    /usr/include/hobot/dnn
    /usr/include/hobot
    ../../../../../utils/c_utils/inc
    ../../../../../utils/tools/mobilenet/cpp
    inc/
)

# Add library search path for linker
link_directories(/usr/hobot/lib)

# Define the executable target and its source files
add_executable(mobilenetv2
    src/main.cpp
    src/classify.cpp
    src/cli.cpp
    ../../../../../utils/c_utils/src/file_io.cpp
    ../../../../../utils/c_utils/src/nn_math.cpp
    ../../../../../utils/c_utils/src/preprocess.cpp
    ../../../../../utils/c_utils/src/postprocess.cpp
    ../../../../../utils/c_utils/src/visualize.cpp
)

# Find package (required)
find_package(OpenCV REQUIRED)

# Libraries to link with the target
SET(LINK_libs dnn hbucp fmt ${OpenCV_LIBS})

target_link_libraries(mobilenetv2 ${LINK_libs})
