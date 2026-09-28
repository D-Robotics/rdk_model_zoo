cmake_minimum_required(VERSION 3.15)
project(gemma4_vision_host_tests CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
get_filename_component(SAMPLE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
get_filename_component(REPO_ROOT "${SAMPLE_ROOT}/../../.." ABSOLUTE)
set(CPP "${SAMPLE_ROOT}/runtime/cpp")
set(LEGACY "${REPO_ROOT}/platforms/s/samples/llm/gemma4-e2b/runtime/cpp")
add_library(source_preprocess OBJECT "${LEGACY}/src/gemma4_vision_preprocess.cpp")
target_include_directories(source_preprocess PRIVATE "${LEGACY}/inc" ${OpenCV_INCLUDE_DIRS})
target_compile_definitions(source_preprocess PRIVATE PreprocessImage=SourcePreprocessImage)
add_executable(vision_test vision_test.cpp
  "${CPP}/src/gemma4_vision_preprocess.cpp"
  "${CPP}/src/gemma4_vision_task.cpp"
  "${CPP}/src/gemma4_image_io.cpp"
  $<TARGET_OBJECTS:source_preprocess>)
target_include_directories(vision_test PRIVATE "${CPP}/inc" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vision_test PRIVATE ${OpenCV_LIBS})
if(MSVC)
  target_compile_options(vision_test PRIVATE /UNDEBUG)
else()
  target_compile_options(vision_test PRIVATE -UNDEBUG)
endif()
enable_testing()
add_test(NAME vision_contract COMMAND vision_test)
add_test(NAME vision_source_images COMMAND vision_test "${SAMPLE_ROOT}/test_data")

# Metadata/packing tests use local declarations, not a vendor SDK or BPU.
add_executable(vision_tensor_test vision_tensor_test.cpp "${CPP}/src/gemma4_vision_tensor.cpp")
target_include_directories(vision_tensor_test PRIVATE sdk_fixtures "${CPP}/inc")
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
# argmax semantics; no SDK.
add_executable(text_tensor_test text_tensor_test.cpp
  "${CPP}/src/gemma4_text_tensor.cpp")
target_include_directories(text_tensor_test PRIVATE sdk_fixtures "${CPP}/inc")

# Production TextEngine against an SDK double speaking the fixed contract:
# generation flow, cache transport, post-inference drift and rejections.
add_executable(text_tensor_flow_test text_tensor_flow_test.cpp
  "${CPP}/src/gemma4_text_engine.cpp" "${CPP}/src/gemma4_kv_cache.cpp"
  "${CPP}/src/gemma4_text_tensor.cpp")
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
  "${CPP}/src/gemma4_text_tensor.cpp")
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
