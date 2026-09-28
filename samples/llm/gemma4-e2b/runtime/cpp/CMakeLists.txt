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

# Exact target is supplied by the launcher after board identity resolution.
set(GEMMA4_TARGET "" CACHE STRING "s100, s100p or s600")
if(NOT GEMMA4_TARGET MATCHES "^(s100|s100p|s600)$")
  message(FATAL_ERROR "Set -DGEMMA4_TARGET=s100/s100p/s600 (matching board and HBM)")
endif()
string(TOUPPER "${GEMMA4_TARGET}" GEMMA4_TARGET_UPPER)
add_compile_definitions(SOC_${GEMMA4_TARGET_UPPER})

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

# gflags is the standard CLI parser per Model Zoo Guideline §c/c++ 规范.
find_package(gflags REQUIRED)

include_directories(${HOBOT_INCLUDE_DIR})

# ── Native C++ tokenizer (tokenizers-cpp) ─────────────────────────────────────
# Provides tokenizers_cpp + tokenizers_c (Rust HF tokenizers C binding) +
# sentencepiece. Matches the OpenExplorer_LLM-s600 reference stack.
# Dependency acquisition is an explicit preparation step, never a CMake side effect.
set(TOKENIZERS_CPP_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/tokenizers-cpp)
if(NOT EXISTS "${TOKENIZERS_CPP_ROOT}/CMakeLists.txt")
  message(FATAL_ERROR
    "Prepare tokenizers-cpp first: bash third_party/install_tokenizers_cpp.sh (from sample root)")
endif()
add_subdirectory(${TOKENIZERS_CPP_ROOT} ${CMAKE_CURRENT_BINARY_DIR}/tokenizers-cpp)

# ── Static library: core inference engine ────────────────────────────────────
add_library(gemma4_runtime STATIC
  src/gemma4_native_tokenizer.cpp
  src/gemma4_tokenizer.cpp
  src/gemma4_embeddings.cpp
  src/gemma4_kv_cache.cpp
  src/gemma4_text_engine.cpp
  src/gemma4_vision_preprocess.cpp
  src/gemma4_vision_engine.cpp
)

target_include_directories(gemma4_runtime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/inc
  ${TOKENIZERS_CPP_ROOT}/include
)
target_link_libraries(gemma4_runtime PUBLIC
  ${DNN_LIB} ${HBUCP_LIB} ${OpenCV_LIBS}
  ${CMAKE_DL_LIBS}
  tokenizers_cpp
)

# ── Executables ──────────────────────────────────────────────────────────────

# Interactive VLM chat (Model Zoo standard entry: main.cpp -> main)
add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE gemma4_runtime gflags)

# HTTP API server
add_executable(gemma4_server src/gemma4_server.cpp)
target_link_libraries(gemma4_server PRIVATE gemma4_runtime gflags)

# Text-only benchmark
add_executable(gemma4_text_bench src/gemma4_text_bench.cpp)
target_link_libraries(gemma4_text_bench PRIVATE gemma4_runtime gflags)

# Single-shot VLM diagnostic (image + prompt → text)
add_executable(gemma4_demo src/gemma4_demo.cpp)
target_link_libraries(gemma4_demo PRIVATE gemma4_runtime gflags)

# Golden mask/KV alignment verification tool
add_executable(gemma4_golden_verify src/gemma4_golden_verify.cpp)
target_link_libraries(gemma4_golden_verify PRIVATE gemma4_runtime gflags)

set_target_properties(
  main gemma4_server gemma4_text_bench gemma4_demo gemma4_golden_verify
  PROPERTIES
  BUILD_RPATH "/usr/hobot/lib"
  INSTALL_RPATH "/usr/hobot/lib"
)
