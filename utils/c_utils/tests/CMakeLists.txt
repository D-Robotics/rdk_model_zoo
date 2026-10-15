cmake_minimum_required(VERSION 3.18)
project(c_utils_file_io_tests LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(C_UTILS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
get_filename_component(REPO_ROOT "${C_UTILS_ROOT}/../.." ABSOLUTE)

# Host-only: the shared file-I/O helpers plus their label-reader test. The
# ImageNet names file and the plain one-label-per-line oracle
# (samples/vision/resnet/test_data/imagenet1000_labels.txt) are read from
# the checkout; nothing is downloaded and no vendor SDK is involved.
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs)
find_path(NLOHMANN_JSON_INCLUDE nlohmann/json.hpp REQUIRED)

enable_testing()
add_executable(test_file_io test_file_io.cc "${C_UTILS_ROOT}/src/file_io.cpp")
target_include_directories(test_file_io PRIVATE
    "${C_UTILS_ROOT}/inc" ${OpenCV_INCLUDE_DIRS} "${NLOHMANN_JSON_INCLUDE}")
target_compile_definitions(test_file_io PRIVATE REPO_ROOT="${REPO_ROOT}")
# file_io.cpp references cv::imread/cv::Mat, so the OpenCV libraries found
# above must be linked into this target, not only used for include paths.
target_link_libraries(test_file_io PRIVATE ${OpenCV_LIBS})
target_compile_options(test_file_io PRIVATE -Wall -Wextra -Werror)
add_test(NAME file_io COMMAND test_file_io)
