cmake_minimum_required(VERSION 3.22)
project(cuda_fp32_profiling_cases LANGUAGES CXX CUDA)

option(ENABLE_CUDA_LINEINFO "Enable CUDA lineinfo for Nsight Compute" ON)
option(FORCE_O3 "Compile with -O3 regardless of build type" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if (NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()
message(STATUS "Profiling build CUDA arch: sm_${CMAKE_CUDA_ARCHITECTURES}")

find_package(CUDAToolkit REQUIRED)

function(apply_common_cuda_flags target)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )

  if (FORCE_O3)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-O3>
      $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    )
  endif()

  if (ENABLE_CUDA_LINEINFO)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )
  endif()
endfunction()

add_executable(gemm_fp32_good_cublas gemm_fp32_good_cublas.cu)
target_link_libraries(gemm_fp32_good_cublas PRIVATE CUDA::cublas)
apply_common_cuda_flags(gemm_fp32_good_cublas)

add_executable(gemm_fp32_bad_naive gemm_fp32_bad_naive.cu)
target_link_libraries(gemm_fp32_bad_naive PRIVATE CUDA::cudart)
apply_common_cuda_flags(gemm_fp32_bad_naive)
