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

# ── Native C++ tokenizer (tokenizers-cpp, vendored) ──────────────────────────
# Provides tokenizers_cpp + tokenizers_c (Rust HF tokenizers C binding) +
# sentencepiece. Matches the OpenExplorer_LLM-s600 reference stack.
set(TOKENIZERS_CPP_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/tokenizers-cpp)
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

# Interactive VLM chat (primary entry point)
add_executable(gemma4_chat gemma4_chat.cpp)
target_link_libraries(gemma4_chat PRIVATE gemma4_runtime)

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
  gemma4_chat gemma4_server gemma4_text_bench gemma4_demo gemma4_golden_verify
  PROPERTIES
  BUILD_RPATH "/usr/hobot/lib"
  INSTALL_RPATH "/usr/hobot/lib"
)
