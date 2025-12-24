cmake_minimum_required(VERSION 3.0)
project(yolov5)
set(CMAKE_BUILD_TYPE Release)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set device model macro definition
file(READ "/sys/class/boardinfo/soc_name" SOC_NAME_RAW)
string(TOUPPER "${SOC_NAME_RAW}" SOC_NAME_UPPER)
add_definitions(-DPLATFORM_${SOC_NAME_UPPER})

# Include directories for header file search
include_directories(
    /usr/hobot/include
    /usr/include/hobot/dnn
    /usr/include/hobot
    ../../../../../utils/c_utils/inc
    inc/
)

# Add library search path for linker
link_directories(/usr/hobot/lib)

# Define the executable target and its source files
add_executable(yolov5
    src/main.cc
    src/yolov5.cc
    ../../../../../utils/c_utils/src/file_io.cc
    ../../../../../utils/c_utils/src/nn_math.cc
    ../../../../../utils/c_utils/src/preprocess.cc
    ../../../../../utils/c_utils/src/postprocess.cc
    ../../../../../utils/c_utils/src/visualize.cc
)

# Find package (required)
find_package(OpenCV REQUIRED)

# Libraries to link with the target
SET(LINK_libs dnn hbucp gflags fmt ${OpenCV_LIBS})

# Link libraries to the executable target
target_link_libraries(yolov5 ${LINK_libs})
