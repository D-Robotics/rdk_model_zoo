cmake_minimum_required(VERSION 3.15)
project(gemma4_cpp_runtime CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -pipe -fPIC")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -pipe -fPIC")

# ── Horizon BPU SDK discovery ─────────────────────────────────────────────────
find_path(HOBOT_INCLUDE_DIR hobot/dnn/hb_dnn.h
  PATHS /usr/include /usr/hobot/include
  NO_DEFAULT_PATH)
if(NOT HOBOT_INCLUDE_DIR)
  find_path(HOBOT_INCLUDE_DIR hobot/dnn/hb_dnn.h)
endif()

find_library(DNN_LIB dnn PATHS /usr/hobot/lib /usr/lib/aarch64-linux-gnu)
find_library(HBUCP_LIB hbucp PATHS /usr/hobot/lib /usr/lib/aarch64-linux-gnu)

if(NOT HOBOT_INCLUDE_DIR OR NOT DNN_LIB OR NOT HBUCP_LIB)
  message(FATAL_ERROR
    "Horizon BPU SDK not found.\n"
    "Expected headers in /usr/hobot/include/hobot/dnn/hb_dnn.h\n"
    "Expected libs in /usr/hobot/lib/libdnn.so and libhbucp.so\n"
    "Please install the OE-LLM board runtime package first.")
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

include_directories(${HOBOT_INCLUDE_DIR})

# ── Native C++ tokenizer (tokenizers-cpp) ─────────────────────────────────────
# Provides tokenizers_cpp + tokenizers_c (Rust HF tokenizers C binding) +
# sentencepiece. Matches the OpenExplorer_LLM-s600 reference stack.
# Source is NOT vendored in git; it is downloaded by install_tokenizers_cpp.sh
# (pinned commit). run.sh calls it before cmake; this block makes manual
# `cmake ..` also work by auto-fetching if the dir is missing.
set(TOKENIZERS_CPP_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/tokenizers-cpp)
if(NOT EXISTS "${TOKENIZERS_CPP_ROOT}/CMakeLists.txt")
  message(STATUS "tokenizers-cpp not found, running installer...")
  execute_process(
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/install_tokenizers_cpp.sh
    RESULT_VARIABLE _install_rc)
  if(NOT _install_rc EQUAL 0)
    message(FATAL_ERROR
      "Failed to download tokenizers-cpp.\n"
      "Run manually: bash third_party/install_tokenizers_cpp.sh")
  endif()
endif()
add_subdirectory(${TOKENIZERS_CPP_ROOT} ${CMAKE_CURRENT_BINARY_DIR}/tokenizers-cpp)

# ── Static library: core inference engine ────────────────────────────────────
add_library(gemma4_runtime STATIC
  gemma4_native_tokenizer.cc
  gemma4_tokenizer.cpp
  gemma4_embeddings.cpp
  gemma4_kv_cache.cpp
  gemma4_text_engine.cpp
  gemma4_vision_preprocess.cpp
  gemma4_vision_engine.cpp
)

target_include_directories(gemma4_runtime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${TOKENIZERS_CPP_ROOT}/include
)
target_link_libraries(gemma4_runtime PUBLIC
  ${DNN_LIB} ${HBUCP_LIB} ${OpenCV_LIBS}
  tokenizers_cpp
)

# ── Executables ──────────────────────────────────────────────────────────────

# Interactive VLM chat (Model Zoo standard entry: main.cpp -> main)
add_executable(main main.cpp)
target_link_libraries(main PRIVATE gemma4_runtime)

# HTTP API server
add_executable(gemma4_server gemma4_server.cpp)
target_link_libraries(gemma4_server PRIVATE gemma4_runtime)

# Text-only benchmark
add_executable(gemma4_text_bench gemma4_text_bench.cpp)
target_link_libraries(gemma4_text_bench PRIVATE gemma4_runtime)

# Single-shot VLM diagnostic (image + prompt → text)
add_executable(gemma4_demo gemma4_demo.cpp)
target_link_libraries(gemma4_demo PRIVATE gemma4_runtime)

# Golden mask/KV alignment verification tool
add_executable(gemma4_golden_verify gemma4_golden_verify.cpp)
target_link_libraries(gemma4_golden_verify PRIVATE gemma4_runtime)

set_target_properties(
  main gemma4_server gemma4_text_bench gemma4_demo gemma4_golden_verify
  PROPERTIES
  BUILD_RPATH "/usr/hobot/lib"
  INSTALL_RPATH "/usr/hobot/lib"
)
