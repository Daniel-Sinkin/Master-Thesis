#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                          \
  do {                                                                            \
    cudaError_t err__ = (call);                                                   \
    if (err__ != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,         \
                   cudaGetErrorString(err__));                                    \
      std::exit(EXIT_FAILURE);                                                    \
    }                                                                             \
  } while (0)

struct GpuTimer {
  cudaEvent_t start{};
  cudaEvent_t stop{};

  GpuTimer() {
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
  }

  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void begin() {
    CHECK_CUDA(cudaEventRecord(start));
  }

  float end_ms() {
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

__global__ void init_buffer(float* x, int n, float base) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    x[tid] = base + static_cast<float>(tid % 257) * 1.0e-4f;
  }
}

inline void init_device_buffer(float* x, int n, float base) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  init_buffer<<<blocks, threads>>>(x, n, base);
  CHECK_CUDA(cudaGetLastError());
}
