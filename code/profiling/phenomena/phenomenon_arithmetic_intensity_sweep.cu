#include <array>
#include <cstdio>
#include <cstdlib>

#include "bench_common.cuh"

__global__ void arithmetic_intensity_kernel(const float* x,
                                            const float* y,
                                            float* out,
                                            int n,
                                            int compute_iters) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }
  float a = x[tid];
  float b = y[tid];
#pragma unroll 1
  for (int i = 0; i < compute_iters; ++i) {
    a = fmaf(a, b, 0.000001f);
    b = fmaf(b, 1.000001f, 0.000001f);
  }
  out[tid] = a;
}

int main(int argc, char** argv) {
  int n = 1 << 24;
  int warmup = 5;
  int iters = 20;

  if (argc == 4) {
    n = std::atoi(argv[1]);
    warmup = std::atoi(argv[2]);
    iters = std::atoi(argv[3]);
  } else if (argc != 1) {
    std::fprintf(stderr, "Usage: %s [N warmup iters]\n", argv[0]);
    return EXIT_FAILURE;
  }

  float* d_x = nullptr;
  float* d_y = nullptr;
  float* d_out = nullptr;
  CHECK_CUDA(cudaMalloc(&d_x, static_cast<size_t>(n) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_y, static_cast<size_t>(n) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, static_cast<size_t>(n) * sizeof(float)));
  init_device_buffer(d_x, n, 1.0f);
  init_device_buffer(d_y, n, 2.0f);
  init_device_buffer(d_out, n, 0.0f);
  CHECK_CUDA(cudaDeviceSynchronize());

  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  std::array<int, 9> compute_levels{1, 2, 4, 8, 16, 32, 64, 128, 256};

  std::printf("compute_iters,arithmetic_intensity_flop_per_byte,avg_ms,effective_tflops,effective_bw_gbps\n");
  for (int compute_iters : compute_levels) {
    auto launch = [&]() {
      arithmetic_intensity_kernel<<<blocks, threads>>>(d_x, d_y, d_out, n, compute_iters);
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
    double flops_per_elem = static_cast<double>(compute_iters) * 4.0;
    double total_flops = static_cast<double>(n) * flops_per_elem * iters;
    double total_bytes = static_cast<double>(n) * 12.0 * iters;
    double intensity = flops_per_elem / 12.0;
    double tflops = total_flops / (static_cast<double>(ms) * 1.0e9);
    double bw = total_bytes / (static_cast<double>(ms) * 1.0e6);

    std::printf("%d,%.6f,%.6f,%.6f,%.6f\n",
                compute_iters, intensity, avg_ms, tflops, bw);
  }

  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_y));
  CHECK_CUDA(cudaFree(d_out));
  return 0;
}
