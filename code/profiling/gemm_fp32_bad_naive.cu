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

__global__ void init_matrix(float* x, size_t n, float base) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] = base + static_cast<float>(i % 17) * 0.001f;
  }
}

__global__ void naive_sgemm_fp32(const float* a,
                                 const float* b,
                                 float* c,
                                 int m, int n, int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < m && col < n) {
    float acc = 0.0f;
    for (int i = 0; i < k; ++i) {
      acc += a[row * k + i] * b[i * n + col];
    }
    c[row * n + col] = acc;
  }
}

int main(int argc, char** argv) {
  int m = 4096;
  int n = 4096;
  int k = 4096;
  int warmup = 5;
  int iters = 20;

  if (argc == 6) {
    m = std::atoi(argv[1]);
    n = std::atoi(argv[2]);
    k = std::atoi(argv[3]);
    warmup = std::atoi(argv[4]);
    iters = std::atoi(argv[5]);
  } else if (argc != 1) {
    std::fprintf(stderr, "Usage: %s [M N K warmup iters]\n", argv[0]);
    return EXIT_FAILURE;
  }

  std::printf("Case: BAD (FP32 naive global-memory GEMM)\n");
  std::printf("M=%d N=%d K=%d warmup=%d iters=%d\n", m, n, k, warmup, iters);

  const size_t size_a = static_cast<size_t>(m) * k;
  const size_t size_b = static_cast<size_t>(k) * n;
  const size_t size_c = static_cast<size_t>(m) * n;

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, size_a * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_b, size_b * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_c, size_c * sizeof(float)));

  const int init_threads = 256;
  const int blocks_a = static_cast<int>((size_a + init_threads - 1) / init_threads);
  const int blocks_b = static_cast<int>((size_b + init_threads - 1) / init_threads);
  const int blocks_c = static_cast<int>((size_c + init_threads - 1) / init_threads);
  init_matrix<<<blocks_a, init_threads>>>(d_a, size_a, 1.0f);
  init_matrix<<<blocks_b, init_threads>>>(d_b, size_b, 2.0f);
  init_matrix<<<blocks_c, init_threads>>>(d_c, size_c, 0.0f);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  dim3 block(16, 16);
  dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  auto launch = [&]() { naive_sgemm_fp32<<<grid, block>>>(d_a, d_b, d_c, m, n, k); };

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
  const double flops = 2.0 * static_cast<double>(m) * n * k * iters;
  const double tflops = flops / (static_cast<double>(ms) * 1.0e9);
  const double approx_a100_fp32_peak = 19.5;
  const double approx_peak_pct = 100.0 * tflops / approx_a100_fp32_peak;

  std::printf("Total time: %.3f ms\n", static_cast<double>(ms));
  std::printf("Avg GEMM:  %.3f ms\n", avg_ms);
  std::printf("Throughput: %.2f TFLOP/s\n", tflops);
  std::printf("Approx A100 FP32 peak: %.2f%% of 19.5 TFLOP/s\n", approx_peak_pct);

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));
  return 0;
}
