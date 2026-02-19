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

__global__ void init_array(float* x, size_t n, float base) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] = base + static_cast<float>(i % 257) * 0.0001f;
  }
}

int main(int argc, char** argv) {
  int batch = 512;
  int d2 = 16;
  int chi2 = 256;
  int warmup = 20;
  int iters = 120;

  if (argc == 6) {
    batch = std::atoi(argv[1]);
    d2 = std::atoi(argv[2]);
    chi2 = std::atoi(argv[3]);
    warmup = std::atoi(argv[4]);
    iters = std::atoi(argv[5]);
  } else if (argc != 1) {
    std::fprintf(stderr, "Usage: %s [batch d2 chi2 warmup iters]\n", argv[0]);
    return EXIT_FAILURE;
  }

  std::printf("Case: GOOD (TN two-site ordered contraction via batched GEMM)\n");
  std::printf("batch=%d d2=%d chi2=%d warmup=%d iters=%d\n",
              batch, d2, chi2, warmup, iters);

  const size_t size_g = static_cast<size_t>(d2) * d2;
  const size_t size_m = static_cast<size_t>(chi2) * chi2;
  const size_t size_x = static_cast<size_t>(batch) * d2 * chi2;

  float* d_g = nullptr;
  float* d_m = nullptr;
  float* d_x = nullptr;
  float* d_tmp = nullptr;
  float* d_y = nullptr;
  CHECK_CUDA(cudaMalloc(&d_g, size_g * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_m, size_m * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_x, size_x * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_tmp, size_x * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_y, size_x * sizeof(float)));

  const int init_threads = 256;
  init_array<<<static_cast<int>((size_g + init_threads - 1) / init_threads), init_threads>>>(d_g, size_g, 0.25f);
  init_array<<<static_cast<int>((size_m + init_threads - 1) / init_threads), init_threads>>>(d_m, size_m, 0.50f);
  init_array<<<static_cast<int>((size_x + init_threads - 1) / init_threads), init_threads>>>(d_x, size_x, 1.00f);
  init_array<<<static_cast<int>((size_x + init_threads - 1) / init_threads), init_threads>>>(d_tmp, size_x, 0.00f);
  init_array<<<static_cast<int>((size_x + init_threads - 1) / init_threads), init_threads>>>(d_y, size_x, 0.00f);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

  const float alpha = 1.0f;
  const float beta = 0.0f;

  // All matrices are interpreted in column-major layout for cuBLAS.
  // X_b, T_b, Y_b are d2 x chi2 with leading dimension d2.
  const long long stride_x = static_cast<long long>(d2) * chi2;
  const long long stride_t = static_cast<long long>(d2) * chi2;
  const long long stride_y = static_cast<long long>(d2) * chi2;

  auto launch = [&]() {
    // T_b = G * X_b, batched over b.
    CHECK_CUBLAS(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        d2, chi2, d2,
        &alpha,
        d_g, d2, 0,
        d_x, d2, stride_x,
        &beta,
        d_tmp, d2, stride_t,
        batch));

    // Y_b = T_b * M, batched over b.
    CHECK_CUBLAS(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        d2, chi2, chi2,
        &alpha,
        d_tmp, d2, stride_t,
        d_m, chi2, 0,
        &beta,
        d_y, d2, stride_y,
        batch));
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

  // Useful FLOPs of the ordered contraction:
  // batch * [2*d2*d2*chi2 + 2*d2*chi2*chi2] per iteration.
  const double flops_per_iter =
      static_cast<double>(batch) *
      (2.0 * d2 * d2 * chi2 + 2.0 * d2 * chi2 * chi2);
  const double tflops = (flops_per_iter * iters) / (static_cast<double>(ms) * 1.0e9);

  float sample = 0.0f;
  CHECK_CUDA(cudaMemcpy(&sample, d_y, sizeof(float), cudaMemcpyDeviceToHost));

  std::printf("Total time: %.3f ms\n", static_cast<double>(ms));
  std::printf("Avg step:  %.3f ms\n", avg_ms);
  std::printf("Useful throughput: %.2f TFLOP/s\n", tflops);
  std::printf("Sample output: %.6f\n", static_cast<double>(sample));

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUBLAS(cublasDestroy(handle));
  CHECK_CUDA(cudaFree(d_g));
  CHECK_CUDA(cudaFree(d_m));
  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_tmp));
  CHECK_CUDA(cudaFree(d_y));
  return 0;
}
