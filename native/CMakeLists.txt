cmake_minimum_required(VERSION 3.20)
project(typopilot_inference LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED LLAMA_CPP_DIR)
  message(FATAL_ERROR "Pass -DLLAMA_CPP_DIR=<path-to-llama.cpp>")
endif()

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_CPP_DIR} llama.cpp-build EXCLUDE_FROM_ALL)

add_executable(typopilot-harness src/main.cpp)
target_link_libraries(typopilot-harness PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(typopilot-harness PRIVATE cxx_std_17)

