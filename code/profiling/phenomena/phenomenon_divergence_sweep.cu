#include <array>
#include <cstdio>
#include <cstdlib>
#include <utility>

#include "bench_common.cuh"

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

__global__ void divergence_fraction_kernel(const float* in,
                                           float* out,
                                           int n,
                                           int inner_iters,
                                           int divergent_warps_per_1024) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }

  unsigned int warp_global = static_cast<unsigned int>(tid >> 5);
  int lane = threadIdx.x & 31;
  bool warp_divergent = ((warp_global * 2654435761u) & 1023u) <
                        static_cast<unsigned int>(divergent_warps_per_1024);
  bool take_a = warp_divergent ? (lane < 16) : true;

  float v = in[tid];
  if (take_a) {
    v = path_a(v, inner_iters);
  } else {
    v = path_b(v, inner_iters);
  }
  out[tid] = v;
}

int main(int argc, char** argv) {
  int n = 1 << 22;
  int inner_iters = 256;
  int warmup = 10;
  int iters = 50;

  if (argc == 5) {
    n = std::atoi(argv[1]);
    inner_iters = std::atoi(argv[2]);
    warmup = std::atoi(argv[3]);
    iters = std::atoi(argv[4]);
  } else if (argc != 1) {
    std::fprintf(stderr, "Usage: %s [N inner_iters warmup iters]\n", argv[0]);
    return EXIT_FAILURE;
  }

  float* d_a = nullptr;
  float* d_b = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, static_cast<size_t>(n) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_b, static_cast<size_t>(n) * sizeof(float)));
  init_device_buffer(d_a, n, 1.0f);
  init_device_buffer(d_b, n, 0.0f);
  CHECK_CUDA(cudaDeviceSynchronize());

  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  std::array<int, 9> levels{0, 128, 256, 384, 512, 640, 768, 896, 1024};

  std::printf("divergent_fraction,avg_ms,effective_tflops\n");
  for (int divergent : levels) {
    auto launch = [&]() {
      divergence_fraction_kernel<<<blocks, threads>>>(d_a, d_b, n, inner_iters, divergent);
      std::swap(d_a, d_b);
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

    double avg_ms = static_cast<double>(ms) / iters;
    double flops = static_cast<double>(n) * inner_iters * 4.0 * iters;
    double tflops = flops / (static_cast<double>(ms) * 1.0e9);
    double frac = static_cast<double>(divergent) / 1024.0;
    std::printf("%.4f,%.6f,%.6f\n", frac, avg_ms, tflops);
  }

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  return 0;
}
