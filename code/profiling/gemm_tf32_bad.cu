#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call)                                                          \
  do {                                                                            \
    cudaError_t err__ = (call);                                                   \
    if (err__ != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,         \
                   cudaGetErrorString(err__));                                    \
      std::exit(EXIT_FAILURE);                                                    \
    }                                                                             \
  } while (0)

#define CHECK_CUBLAS(call)                                                        \
  do {                                                                            \
    cublasStatus_t st__ = (call);                                                 \
    if (st__ != CUBLAS_STATUS_SUCCESS) {                                          \
      std::fprintf(stderr, "cuBLAS error %s:%d: status=%d\n", __FILE__, __LINE__,\
                   static_cast<int>(st__));                                       \
      std::exit(EXIT_FAILURE);                                                    \
    }                                                                             \
  } while (0)

__global__ void init_matrix(float* x, size_t n, float base) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] = base + static_cast<float>(i % 17) * 0.001f;
  }
}

int main(int argc, char** argv) {
  int m = 8192;
  int n = 8192;
  int k = 8192;
  int warmup = 20;
  int iters = 120;

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

  std::printf("Case: BAD (pedantic FP32 path, tensor cores discouraged)\n");
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

  const int threads = 256;
  const int blocks_a = static_cast<int>((size_a + threads - 1) / threads);
  const int blocks_b = static_cast<int>((size_b + threads - 1) / threads);
  const int blocks_c = static_cast<int>((size_c + threads - 1) / threads);
  init_matrix<<<blocks_a, threads>>>(d_a, size_a, 1.0f);
  init_matrix<<<blocks_b, threads>>>(d_b, size_b, 2.0f);
  init_matrix<<<blocks_c, threads>>>(d_c, size_c, 0.0f);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  // Pedantic math mode avoids reduced-precision/tensor-core acceleration.
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

  const float alpha = 1.0f;
  const float beta = 0.0f;
  auto launch = [&]() {
    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             m, n, k,
                             &alpha,
                             d_a, m,
                             d_b, k,
                             &beta,
                             d_c, m));
  };

  for (int i = 0; i < warmup; ++i) {
    launch();
  }
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
  std::printf("Approx A100 FP32 peak: %.1f%% of 19.5 TFLOP/s\n", approx_peak_pct);
  std::printf("KPI target in Nsight Compute: tensor-pipe active should be LOW.\n");

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUBLAS(cublasDestroy(handle));
  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));
  return 0;
}
