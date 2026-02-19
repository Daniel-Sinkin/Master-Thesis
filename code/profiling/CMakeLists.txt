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

add_executable(tn_two_site_good_batched_gemm tn_two_site_good_batched_gemm.cu)
target_link_libraries(tn_two_site_good_batched_gemm PRIVATE CUDA::cublas)
apply_common_cuda_flags(tn_two_site_good_batched_gemm)

add_executable(tn_two_site_bad_direct tn_two_site_bad_direct.cu)
target_link_libraries(tn_two_site_bad_direct PRIVATE CUDA::cudart)
apply_common_cuda_flags(tn_two_site_bad_direct)

add_executable(warp_divergence_good_uniform warp_divergence_good_uniform.cu)
target_link_libraries(warp_divergence_good_uniform PRIVATE CUDA::cudart)
apply_common_cuda_flags(warp_divergence_good_uniform)

add_executable(warp_divergence_bad_divergent warp_divergence_bad_divergent.cu)
target_link_libraries(warp_divergence_bad_divergent PRIVATE CUDA::cudart)
apply_common_cuda_flags(warp_divergence_bad_divergent)

add_executable(phenomenon_divergence_sweep phenomena/phenomenon_divergence_sweep.cu)
target_include_directories(phenomenon_divergence_sweep PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/phenomena)
target_link_libraries(phenomenon_divergence_sweep PRIVATE CUDA::cudart)
apply_common_cuda_flags(phenomenon_divergence_sweep)

add_executable(phenomenon_coalescing_sweep phenomena/phenomenon_coalescing_sweep.cu)
target_include_directories(phenomenon_coalescing_sweep PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/phenomena)
target_link_libraries(phenomenon_coalescing_sweep PRIVATE CUDA::cudart)
apply_common_cuda_flags(phenomenon_coalescing_sweep)

add_executable(phenomenon_shared_bank_conflict phenomena/phenomenon_shared_bank_conflict.cu)
target_include_directories(phenomenon_shared_bank_conflict PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/phenomena)
target_link_libraries(phenomenon_shared_bank_conflict PRIVATE CUDA::cudart)
apply_common_cuda_flags(phenomenon_shared_bank_conflict)

add_executable(phenomenon_launch_overhead phenomena/phenomenon_launch_overhead.cu)
target_include_directories(phenomenon_launch_overhead PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/phenomena)
target_link_libraries(phenomenon_launch_overhead PRIVATE CUDA::cudart)
apply_common_cuda_flags(phenomenon_launch_overhead)

add_executable(phenomenon_ilp_dependency phenomena/phenomenon_ilp_dependency.cu)
target_include_directories(phenomenon_ilp_dependency PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/phenomena)
target_link_libraries(phenomenon_ilp_dependency PRIVATE CUDA::cudart)
apply_common_cuda_flags(phenomenon_ilp_dependency)

add_executable(phenomenon_register_pressure phenomena/phenomenon_register_pressure.cu)
target_include_directories(phenomenon_register_pressure PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/phenomena)
target_link_libraries(phenomenon_register_pressure PRIVATE CUDA::cudart)
apply_common_cuda_flags(phenomenon_register_pressure)

add_executable(phenomenon_arithmetic_intensity_sweep phenomena/phenomenon_arithmetic_intensity_sweep.cu)
target_include_directories(phenomenon_arithmetic_intensity_sweep PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/phenomena)
target_link_libraries(phenomenon_arithmetic_intensity_sweep PRIVATE CUDA::cudart)
apply_common_cuda_flags(phenomenon_arithmetic_intensity_sweep)
