#include <array>
#include <cstdio>
#include <cstdlib>

#include "bench_common.cuh"

__global__ void strided_access_kernel(const float* in,
                                      float* out,
                                      int n_mask,
                                      int stride,
                                      int inner_iters) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = (tid * stride) & n_mask;
  float v = in[idx];
#pragma unroll 1
  for (int i = 0; i < inner_iters; ++i) {
    v = fmaf(v, 1.000001f, 0.000001f);
  }
  out[tid] = v;
}

int main(int argc, char** argv) {
  int n = 1 << 24;
  int inner_iters = 4;
  int warmup = 5;
  int iters = 20;

  if (argc == 5) {
    n = std::atoi(argv[1]);
    inner_iters = std::atoi(argv[2]);
    warmup = std::atoi(argv[3]);
    iters = std::atoi(argv[4]);
  } else if (argc != 1) {
    std::fprintf(stderr, "Usage: %s [N inner_iters warmup iters]\n", argv[0]);
    return EXIT_FAILURE;
  }

  if ((n & (n - 1)) != 0) {
    std::fprintf(stderr, "N must be a power-of-two for this benchmark.\n");
    return EXIT_FAILURE;
  }

  float* d_in = nullptr;
  float* d_out = nullptr;
  CHECK_CUDA(cudaMalloc(&d_in, static_cast<size_t>(n) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, static_cast<size_t>(n) * sizeof(float)));
  init_device_buffer(d_in, n, 1.0f);
  init_device_buffer(d_out, n, 0.0f);
  CHECK_CUDA(cudaDeviceSynchronize());

  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  int n_mask = n - 1;
  std::array<int, 8> strides{1, 2, 4, 8, 16, 32, 64, 128};

  std::printf("stride,avg_ms,effective_bw_gbps\n");
  for (int stride : strides) {
    auto launch = [&]() {
      strided_access_kernel<<<blocks, threads>>>(d_in, d_out, n_mask, stride, inner_iters);
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
    double bytes = static_cast<double>(n) * sizeof(float) * 2.0 * iters;
    double bw_gbps = bytes / (static_cast<double>(ms) * 1.0e6);
    std::printf("%d,%.6f,%.6f\n", stride, avg_ms, bw_gbps);
  }

  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_out));
  return 0;
}
