cmake_minimum_required(VERSION 3.15)
project(resnet18_legacy_launcher LANGUAGES CXX)

# The S18 source is maintained in the canonical sample. Keep this historical
# configure path buildable and preserve its old build/resnet18 output location
# without maintaining a second implementation or a second set of flags.
get_filename_component(REPOSITORY_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/../../../../../../../" ABSOLUTE)
add_subdirectory(
    "${REPOSITORY_ROOT}/samples/vision/resnet/runtime/cpp"
    "${CMAKE_BINARY_DIR}/canonical-resnet18"
)
set_target_properties(resnet18 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
