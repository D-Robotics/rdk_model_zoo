cmake_minimum_required(VERSION 3.18)
project(gemma4_vision_host_tests CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
get_filename_component(SAMPLE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
get_filename_component(REPO_ROOT "${SAMPLE_ROOT}/../../.." ABSOLUTE)
set(CPP "${SAMPLE_ROOT}/runtime/cpp")
# The historical platforms/ tree now lives only in Git. The legacy side of
# the preprocess comparison is extracted from the pinned commit into the
# build tree (same bytes the worktree copy used to carry); a shallow clone
# must fetch the pin first:
#   git fetch origin d2d2a4e0a898697bdfe5f68a9740a8c7d7cad57d
set(GEMMA_PLATFORMS_PIN "d2d2a4e0a898697bdfe5f68a9740a8c7d7cad57d")
set(LEGACY_ARCHIVE "${CMAKE_CURRENT_BINARY_DIR}/gemma-legacy-pin.tar")
set(LEGACY "${CMAKE_CURRENT_BINARY_DIR}/gemma-legacy-pin")
find_package(Git REQUIRED)
execute_process(
  COMMAND ${GIT_EXECUTABLE} -C "${REPO_ROOT}" archive --format=tar
          "${GEMMA_PLATFORMS_PIN}" "platforms/s/samples/llm/gemma4-e2b/runtime/cpp"
  OUTPUT_FILE "${LEGACY_ARCHIVE}"
  RESULT_VARIABLE _archive_result)
if(NOT _archive_result EQUAL 0)
  message(FATAL_ERROR
    "Cannot read pinned legacy sources (${GEMMA_PLATFORMS_PIN}). "
    "Fetch the commit first: git fetch origin ${GEMMA_PLATFORMS_PIN}")
endif()
file(ARCHIVE_EXTRACT INPUT "${LEGACY_ARCHIVE}" DESTINATION "${LEGACY}")
if(NOT EXISTS "${LEGACY}/cpp")
  file(RENAME "${LEGACY}/platforms/s/samples/llm/gemma4-e2b/runtime/cpp"
       "${LEGACY}/cpp")
endif()
add_library(source_preprocess OBJECT "${LEGACY}/cpp/src/gemma4_vision_preprocess.cpp")
target_include_directories(source_preprocess PRIVATE "${LEGACY}/cpp/inc" ${OpenCV_INCLUDE_DIRS})
target_compile_definitions(source_preprocess PRIVATE PreprocessImage=SourcePreprocessImage)
# gemma4_vision_engine.cpp now carries the whole vision pipeline in one
# translation unit: image IO, preprocessing, stage composition, the tensor
# contract and the SDK-driven engine (formerly separate
# gemma4_image_io/gemma4_vision_task/gemma4_vision_preprocess/
# gemma4_vision_tensor fragments); it reads the SDK through the sdk_fixtures
# header double like the text stage tests.
add_executable(vision_test vision_test.cpp
  "${CPP}/src/gemma4_vision_engine.cpp"
  $<TARGET_OBJECTS:source_preprocess>)
target_include_directories(vision_test PRIVATE sdk_fixtures "${CPP}/inc" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vision_test PRIVATE ${OpenCV_LIBS} ${CMAKE_DL_LIBS})
if(MSVC)
  target_compile_options(vision_test PRIVATE /UNDEBUG)
else()
  target_compile_options(vision_test PRIVATE -UNDEBUG)
endif()
enable_testing()
add_test(NAME vision_contract COMMAND vision_test)
add_test(NAME vision_source_images COMMAND vision_test "${SAMPLE_ROOT}/test_data")

# Metadata/packing tests use local declarations, not a vendor SDK or BPU.
# The contract helpers live in the folded engine source, so the binary
# compiles the full engine TU (OpenCV headers and SDK link stubs included).
add_executable(vision_tensor_test vision_tensor_test.cpp "${CPP}/src/gemma4_vision_engine.cpp")
target_include_directories(vision_tensor_test PRIVATE sdk_fixtures "${CPP}/inc" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vision_tensor_test PRIVATE ${OpenCV_LIBS})
if(MSVC)
  target_compile_options(vision_tensor_test PRIVATE /UNDEBUG)
else()
  target_compile_options(vision_tensor_test PRIVATE -UNDEBUG)
endif()
add_test(NAME vision_tensor_contract COMMAND vision_tensor_test)

add_executable(kv_cache_test kv_cache_test.cpp "${CPP}/src/gemma4_kv_cache.cpp")
target_include_directories(kv_cache_test PRIVATE sdk_fixtures "${CPP}/inc")
if(MSVC)
  target_compile_options(kv_cache_test PRIVATE /UNDEBUG)
else()
  target_compile_options(kv_cache_test PRIVATE -UNDEBUG)
endif()
foreach(KV_CASE reset transaction null invalid append alias reallocate)
  add_test(NAME "kv_${KV_CASE}" COMMAND kv_cache_test "${KV_CASE}")
endforeach()

# Fixed Text export descriptor contract: physical layout, padding, and
# argmax semantics; no SDK. The helpers live in the folded engine source, so
# the binary compiles the full engine TU against the text_fixture double.
add_executable(text_tensor_test text_tensor_test.cpp
  "${CPP}/src/gemma4_text_engine.cpp" "${CPP}/src/gemma4_kv_cache.cpp"
  "${CPP}/src/gemma4_text_session.cpp")
target_include_directories(text_tensor_test PRIVATE sdk_fixtures "${CPP}/inc")
target_link_libraries(text_tensor_test PRIVATE ${CMAKE_DL_LIBS})

# Production TextEngine against an SDK double speaking the fixed contract:
# generation flow, cache transport, post-inference drift and rejections.
add_executable(text_tensor_flow_test text_tensor_flow_test.cpp
  "${CPP}/src/gemma4_text_engine.cpp" "${CPP}/src/gemma4_kv_cache.cpp"
  "${CPP}/src/gemma4_text_session.cpp")
target_include_directories(text_tensor_flow_test PRIVATE sdk_fixtures "${CPP}/inc")
target_link_libraries(text_tensor_flow_test PRIVATE ${CMAKE_DL_LIBS})

add_executable(model_io_test model_io_test.cpp)
target_include_directories(model_io_test PRIVATE sdk_fixtures "${CPP}/inc")

# R2: tensor adoption under injected allocation failure. The test replaces
# global operator new and passes under the sanitizer runtime.
add_executable(model_io_adopt_test model_io_adopt_test.cpp)
target_include_directories(model_io_adopt_test PRIVATE sdk_fixtures "${CPP}/inc")
add_executable(text_resources_test text_resources_test.cpp
  "${CPP}/src/gemma4_text_engine.cpp" "${CPP}/src/gemma4_kv_cache.cpp"
  "${CPP}/src/gemma4_text_session.cpp")
target_include_directories(text_resources_test PRIVATE sdk_fixtures "${CPP}/inc")
target_link_libraries(text_resources_test PRIVATE ${CMAKE_DL_LIBS})
foreach(OWNER_TEST model_io_test model_io_adopt_test text_resources_test)
  if(MSVC)
    target_compile_options(${OWNER_TEST} PRIVATE /UNDEBUG)
  else()
    target_compile_options(${OWNER_TEST} PRIVATE -UNDEBUG)
  endif()
  add_test(NAME "${OWNER_TEST}" COMMAND ${OWNER_TEST})
endforeach()
foreach(TENSOR_TEST text_tensor_test text_tensor_flow_test)
  if(MSVC)
    target_compile_options(${TENSOR_TEST} PRIVATE /UNDEBUG)
  else()
    target_compile_options(${TENSOR_TEST} PRIVATE -UNDEBUG)
  endif()
  add_test(NAME "${TENSOR_TEST}" COMMAND ${TENSOR_TEST})
endforeach()

# ---- Text stage/session refactor (design: 2026-09-28-gemma-text-stages) ----
add_executable(text_session_test text_session_test.cpp
  "${CPP}/src/gemma4_text_session.cpp")
target_include_directories(text_session_test PRIVATE sdk_fixtures "${CPP}/inc")

# The folded text engine TU carries input preparation, the tensor contract
# and SDK transport alongside the orchestration; the session policy and KV
# cache remain independent algorithm sources.
set(TEXT_STAGE_SOURCES
  "${CPP}/src/gemma4_text_engine.cpp"
  "${CPP}/src/gemma4_text_session.cpp"
  "${CPP}/src/gemma4_kv_cache.cpp")
add_executable(text_stages_test text_stages_test.cpp ${TEXT_STAGE_SOURCES})
target_include_directories(text_stages_test PRIVATE sdk_fixtures "${CPP}/inc")
target_link_libraries(text_stages_test PRIVATE ${CMAKE_DL_LIBS})

add_executable(text_engine_flow_test text_engine_flow_test.cpp
  ${TEXT_STAGE_SOURCES})
target_include_directories(text_engine_flow_test PRIVATE sdk_fixtures "${CPP}/inc")
target_link_libraries(text_engine_flow_test PRIVATE ${CMAKE_DL_LIBS})

# The README stage/session example, compiled and run so the documented code
# and its exact output stay executable.
add_executable(readme_text_stages_example readme_text_stages_example.cpp
  ${TEXT_STAGE_SOURCES})
target_include_directories(readme_text_stages_example PRIVATE sdk_fixtures "${CPP}/inc")
target_link_libraries(readme_text_stages_example PRIVATE ${CMAKE_DL_LIBS})

foreach(STAGE_TEST text_session_test text_stages_test text_engine_flow_test
                   readme_text_stages_example)
  if(MSVC)
    target_compile_options(${STAGE_TEST} PRIVATE /UNDEBUG)
  else()
    target_compile_options(${STAGE_TEST} PRIVATE -UNDEBUG)
  endif()
  add_test(NAME "${STAGE_TEST}" COMMAND ${STAGE_TEST})
endforeach()
