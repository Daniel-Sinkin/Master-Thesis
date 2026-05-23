cmake_minimum_required(VERSION 3.22)
project(peps_cuda_scaffold LANGUAGES CXX)

option(PEPS_CUDA_ENABLE_CUDA "Build CUDA kernels for the PEPS scaffold" OFF)
set(PEPS_CUDA_REAL_TYPE "double" CACHE STRING "CPU scalar type: double or float")
set_property(CACHE PEPS_CUDA_REAL_TYPE PROPERTY STRINGS double float)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(peps_cuda_cpu
  src/memory.cpp
  src/peps_cpu.cpp
)
target_include_directories(peps_cuda_cpu PUBLIC include)
target_compile_options(peps_cuda_cpu PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
)
if(PEPS_CUDA_REAL_TYPE STREQUAL "float")
  target_compile_definitions(peps_cuda_cpu PUBLIC PEPS_CUDA_REAL_FLOAT=1)
elseif(NOT PEPS_CUDA_REAL_TYPE STREQUAL "double")
  message(FATAL_ERROR "PEPS_CUDA_REAL_TYPE must be double or float")
endif()

add_executable(peps_cuda_demo
  src/main.cpp
)
target_link_libraries(peps_cuda_demo PRIVATE peps_cuda_cpu)

add_executable(peps_cuda_unit_tests
  src/tests.cpp
)
target_link_libraries(peps_cuda_unit_tests PRIVATE peps_cuda_cpu)

enable_testing()
add_test(NAME peps_cuda_unit_tests COMMAND peps_cuda_unit_tests)
add_test(NAME peps_cuda_demo_2x2 COMMAND peps_cuda_demo 2 2 2 4)
add_test(NAME peps_cuda_demo_3x3 COMMAND peps_cuda_demo 3 3 2 2)
add_test(NAME peps_cuda_demo_tfi COMMAND peps_cuda_demo 2 2 2 3 tfi)
add_test(NAME peps_cuda_demo_rydberg COMMAND peps_cuda_demo 2 2 2 3 rydberg)

if(PEPS_CUDA_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 90)
  endif()

  add_library(peps_cuda_gpu
    src/peps_cuda.cu
  )
  target_include_directories(peps_cuda_gpu PUBLIC include)
  target_link_libraries(peps_cuda_gpu PUBLIC CUDA::cudart CUDA::cublas)
  set_target_properties(peps_cuda_gpu PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  )
  target_compile_options(peps_cuda_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )

  add_executable(peps_cuda_gpu_smoke
    src/gpu_smoke.cu
  )
  target_link_libraries(peps_cuda_gpu_smoke PRIVATE peps_cuda_gpu CUDA::cudart)
  set_target_properties(peps_cuda_gpu_smoke PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  )
  add_test(NAME peps_cuda_gpu_smoke COMMAND peps_cuda_gpu_smoke)
endif()
