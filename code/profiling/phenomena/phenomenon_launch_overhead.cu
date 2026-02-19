#include <array>
#include <cstdio>
#include <cstdlib>

#include "bench_common.cuh"

__global__ void tiny_work_kernel(float* x, int n) {
  int tid = threadIdx.x;
  if (tid < n) {
    float v = x[tid];
    v = fmaf(v, 1.000001f, 0.000001f);
    x[tid] = v;
  }
}

__global__ void fused_work_kernel(float* x, int n, int launches) {
  int tid = threadIdx.x;
  if (tid < n) {
    float v = x[tid];
#pragma unroll 1
    for (int i = 0; i < launches; ++i) {
      v = fmaf(v, 1.000001f, 0.000001f);
    }
    x[tid] = v;
  }
}

double run_many_launches(float* d_x, int n, int launches, int warmup, int iters) {
  auto launch = [&]() {
    for (int i = 0; i < launches; ++i) {
      tiny_work_kernel<<<1, 256>>>(d_x, n);
    }
  };

  for (int i = 0; i < warmup; ++i) {
    launch();
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  GpuTimer timer;
  timer.begin();
  for (int i = 0; i < iters; ++i) {
    launch();
  }
  float ms = timer.end_ms();
  CHECK_CUDA(cudaGetLastError());
  return static_cast<double>(ms) / iters;
}

double run_fused(float* d_x, int n, int launches, int warmup, int iters) {
  auto launch = [&]() { fused_work_kernel<<<1, 256>>>(d_x, n, launches); };

  for (int i = 0; i < warmup; ++i) {
    launch();
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  GpuTimer timer;
  timer.begin();
  for (int i = 0; i < iters; ++i) {
    launch();
  }
  float ms = timer.end_ms();
  CHECK_CUDA(cudaGetLastError());
  return static_cast<double>(ms) / iters;
}

int main(int argc, char** argv) {
  int n = 256;
  int warmup = 5;
  int iters = 30;

  if (argc == 4) {
    n = std::atoi(argv[1]);
    warmup = std::atoi(argv[2]);
    iters = std::atoi(argv[3]);
  } else if (argc != 1) {
    std::fprintf(stderr, "Usage: %s [n warmup iters]\n", argv[0]);
    return EXIT_FAILURE;
  }

  float* d_x = nullptr;
  CHECK_CUDA(cudaMalloc(&d_x, static_cast<size_t>(n) * sizeof(float)));
  init_device_buffer(d_x, n, 1.0f);
  CHECK_CUDA(cudaDeviceSynchronize());

  std::array<int, 7> batches{1, 10, 50, 100, 250, 500, 1000};
  std::printf("mode,batch,avg_ms\n");
  for (int batch : batches) {
    double many_ms = run_many_launches(d_x, n, batch, warmup, iters);
    double fused_ms = run_fused(d_x, n, batch, warmup, iters);
    std::printf("many_launches,%d,%.6f\n", batch, many_ms);
    std::printf("fused,%d,%.6f\n", batch, fused_ms);
  }

  CHECK_CUDA(cudaFree(d_x));
  return 0;
}
