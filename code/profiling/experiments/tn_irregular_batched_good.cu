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

__global__ void init_array(float* x, size_t n, float base) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] = base + static_cast<float>(i % 257) * 0.0001f;
  }
}

// Single batched launch with coalesced [k][n] access in the inner loop.
__global__ void tn_irregular_batched_good_kernel(const float* a_mk,
                                                 const float* x_bkn,
                                                 float* y_bmn,
                                                 int m,
                                                 int n,
                                                 int k) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int b = blockIdx.z;
  if (row >= m || col >= n) {
    return;
  }

  const float* x_b = x_bkn + static_cast<size_t>(b) * k * n;
  const float* a_row = a_mk + row * k;
  float acc = 0.0f;
  for (int kk = 0; kk < k; ++kk) {
    float a = a_row[kk];
    float x = x_b[kk * n + col];
    acc = fmaf(a, x, acc);
  }
  y_bmn[(static_cast<size_t>(b) * m + row) * n + col] = acc;
}

int main(int argc, char** argv) {
  int batch = 128;
  int m = 47;
  int n = 84;
  int k = 64;
  int warmup = 10;
  int iters = 80;

  if (argc == 7) {
    batch = std::atoi(argv[1]);
    m = std::atoi(argv[2]);
    n = std::atoi(argv[3]);
    k = std::atoi(argv[4]);
    warmup = std::atoi(argv[5]);
    iters = std::atoi(argv[6]);
  } else if (argc != 1) {
    std::fprintf(stderr, "Usage: %s [batch m n k warmup iters]\n", argv[0]);
    return EXIT_FAILURE;
  }

  std::printf("Case: GOOD (irregular contraction with batched launch)\n");
  std::printf("batch=%d m=%d n=%d k=%d warmup=%d iters=%d\n",
              batch, m, n, k, warmup, iters);

  size_t size_a = static_cast<size_t>(m) * k;
  size_t size_x = static_cast<size_t>(batch) * k * n;
  size_t size_y = static_cast<size_t>(batch) * m * n;

  float* d_a = nullptr;
  float* d_x = nullptr;
  float* d_y = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, size_a * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_x, size_x * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_y, size_y * sizeof(float)));

  int init_threads = 256;
  init_array<<<static_cast<int>((size_a + init_threads - 1) / init_threads), init_threads>>>(d_a, size_a, 0.25f);
  init_array<<<static_cast<int>((size_x + init_threads - 1) / init_threads), init_threads>>>(d_x, size_x, 1.00f);
  init_array<<<static_cast<int>((size_y + init_threads - 1) / init_threads), init_threads>>>(d_y, size_y, 0.00f);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  dim3 block(16, 16, 1);
  dim3 grid((n + block.x - 1) / block.x,
            (m + block.y - 1) / block.y,
            batch);

  auto launch = [&]() {
    tn_irregular_batched_good_kernel<<<grid, block>>>(d_a, d_x, d_y, m, n, k);
  };

  for (int i = 0; i < warmup; ++i) {
    launch();
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start{};
  cudaEvent_t stop{};
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    launch();
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaGetLastError());

  float total_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
  double avg_step_ms = static_cast<double>(total_ms) / iters;

  double flops_per_iter =
      2.0 * static_cast<double>(batch) * m * n * k;
  double measured_tflops = (flops_per_iter * iters) / (static_cast<double>(total_ms) * 1.0e9);

  float sample = 0.0f;
  CHECK_CUDA(cudaMemcpy(&sample, d_y, sizeof(float), cudaMemcpyDeviceToHost));

  std::printf("total_ms,avg_step_ms,executed_tflops,sample\n");
  std::printf("%.6f,%.6f,%.6f,%.6f\n",
              static_cast<double>(total_ms),
              avg_step_ms,
              measured_tflops,
              static_cast<double>(sample));

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_y));
  return 0;
}

