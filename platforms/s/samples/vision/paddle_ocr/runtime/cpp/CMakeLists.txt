cmake_minimum_required(VERSION 3.0)
project(paddle_ocr_legacy_entrypoint)

# Keep the historical configure/build directory and executable name while
# delegating every dependency, compiler flag, and source list to the
# canonical sample.  There is deliberately no second C++ build recipe here.
set(CANONICAL_PADDLE_OCR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../samples/vision/paddle_ocr/runtime/cpp")
add_subdirectory("${CANONICAL_PADDLE_OCR}" canonical_paddle_ocr)

# Older instructions expect <old-build>/paddle_ocr after `cmake ..`.
set_target_properties(paddle_ocr PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
