#include <cstdio>
#include <cstdlib>
#include <utility>

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

__global__ void init_input(float* x, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    x[tid] = 1.0f + static_cast<float>(tid % 113) * 0.0001f;
  }
}

__device__ __forceinline__ float path_a(float v, int inner_iters) {
#pragma unroll 1
  for (int i = 0; i < inner_iters; ++i) {
    v = fmaf(v, 1.000001f, 0.000001f);
    v = fmaf(v, 0.999999f, 0.000002f);
  }
  return v;
}

__device__ __forceinline__ float path_b(float v, int inner_iters) {
#pragma unroll 1
  for (int i = 0; i < inner_iters; ++i) {
    v = fmaf(v, 0.999997f, 0.000003f);
    v = fmaf(v, 1.000003f, 0.000001f);
  }
  return v;
}

__global__ void warp_uniform_branch(const float* in, float* out, int n, int inner_iters) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    int warp_global = tid >> 5;
    bool take_a = ((warp_global & 1) == 0);
    float v = in[tid];
    if (take_a) {
      v = path_a(v, inner_iters);
    } else {
      v = path_b(v, inner_iters);
    }
    out[tid] = v;
  }
}

int main(int argc, char** argv) {
  int n = 1 << 22;
  int inner_iters = 256;
  int warmup = 20;
  int iters = 100;

  if (argc == 5) {
    n = std::atoi(argv[1]);
    inner_iters = std::atoi(argv[2]);
    warmup = std::atoi(argv[3]);
    iters = std::atoi(argv[4]);
  } else if (argc != 1) {
    std::fprintf(stderr, "Usage: %s [N inner_iters warmup iters]\n", argv[0]);
    return EXIT_FAILURE;
  }

  std::printf("Case: GOOD (warp-uniform branching)\n");
  std::printf("N=%d inner_iters=%d warmup=%d iters=%d\n", n, inner_iters, warmup, iters);

  float* d_in = nullptr;
  float* d_out = nullptr;
  CHECK_CUDA(cudaMalloc(&d_in, static_cast<size_t>(n) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, static_cast<size_t>(n) * sizeof(float)));

  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  init_input<<<blocks, threads>>>(d_in, n);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  auto launch = [&]() {
    warp_uniform_branch<<<blocks, threads>>>(d_in, d_out, n, inner_iters);
    std::swap(d_in, d_out);
  };

  for (int i = 0; i < warmup; ++i) {
    launch();
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    launch();
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaGetLastError());

  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  const double avg_ms = static_cast<double>(ms) / iters;
  const double flops_per_elem = static_cast<double>(inner_iters) * 4.0;
  const double flops = static_cast<double>(n) * flops_per_elem * iters;
  const double tflops = flops / (static_cast<double>(ms) * 1.0e9);

  float sample = 0.0f;
  CHECK_CUDA(cudaMemcpy(&sample, d_in, sizeof(float), cudaMemcpyDeviceToHost));

  std::printf("Total time: %.3f ms\n", static_cast<double>(ms));
  std::printf("Avg kernel: %.3f ms\n", avg_ms);
  std::printf("Effective throughput: %.2f TFLOP/s\n", tflops);
  std::printf("Sample output: %.6f\n", static_cast<double>(sample));

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_out));
  return 0;
}
