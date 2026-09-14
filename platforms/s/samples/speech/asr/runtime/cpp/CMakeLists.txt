cmake_minimum_required(VERSION 3.0)
project(asr)
set(CMAKE_BUILD_TYPE Release)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set device model macro definition
file(READ "/sys/class/boardinfo/soc_name" SOC_NAME_RAW)
string(TOUPPER "${SOC_NAME_RAW}" SOC_NAME_UPPER)
add_definitions(-DSOC_${SOC_NAME_UPPER})

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
add_executable(asr
    src/main.cpp
    src/asr.cpp
    src/audio_chunk_reader.cpp
    ../../../../../utils/c_utils/src/file_io.cpp
    ../../../../../utils/c_utils/src/preprocess.cpp
)

# Find packages (required)
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE    REQUIRED sndfile)
pkg_check_modules(SAMPLERATE REQUIRED samplerate)

# Libraries to link with the target
SET(LINK_libs dnn hbucp gflags ${OpenCV_LIBS})

# Link libraries to the executable target
target_link_libraries(asr ${LINK_libs})
target_link_libraries(asr ${SNDFILE_LIBRARIES})
target_link_libraries(asr ${SAMPLERATE_LIBRARIES})
