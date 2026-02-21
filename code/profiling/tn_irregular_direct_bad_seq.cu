#include <chrono>
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

// Repack from [batch][k][n] to [batch][n][k]. This emulates repeated
// layout transforms around kernels.
__global__ void tn_irregular_repack_bad_kernel(const float* x_kn,
                                               float* x_nk,
                                               int batch,
                                               int n,
                                               int k) {
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t total = static_cast<size_t>(batch) * n * k;
  if (tid >= total) {
    return;
  }

  int nk = n * k;
  int b = static_cast<int>(tid / nk);
  int rem = static_cast<int>(tid % nk);
  int nn = rem / k;
  int kk = rem % k;

  size_t src = (static_cast<size_t>(b) * k + kk) * n + nn;
  size_t dst = (static_cast<size_t>(b) * n + nn) * k + kk;
  x_nk[dst] = x_kn[src];
}

// Intentionally serial workflow: one kernel launch per batch item and explicit
// synchronisation after each launch.
__global__ void tn_irregular_direct_bad_kernel(const float* a_mk,
                                               const float* x_nk,
                                               float* y_bmn,
                                               int m,
                                               int n,
                                               int k,
                                               int b) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || col >= n) {
    return;
  }

  const float* x_b = x_nk + static_cast<size_t>(b) * n * k;
  float acc = 0.0f;
  for (int kk = 0; kk < k; ++kk) {
    float a = a_mk[row * k + kk];
    float x = x_b[col * k + kk];
    acc = fmaf(a, x, acc);
  }
  y_bmn[(static_cast<size_t>(b) * m + row) * n + col] = acc;
}

int main(int argc, char** argv) {
  int batch = 128;
  int m = 47;
  int n = 84;
  int k = 64;
  int warmup = 1;
  int iters = 4;

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

  std::printf("Case: BAD (irregular direct contraction with serial launches)\n");
  std::printf("batch=%d m=%d n=%d k=%d warmup=%d iters=%d\n",
              batch, m, n, k, warmup, iters);

  size_t size_a = static_cast<size_t>(m) * k;
  size_t size_x = static_cast<size_t>(batch) * k * n;
  size_t size_y = static_cast<size_t>(batch) * m * n;

  float* d_a = nullptr;
  float* d_x_kn = nullptr;
  float* d_x_nk = nullptr;
  float* d_y = nullptr;
  float* h_y = nullptr;

  CHECK_CUDA(cudaMalloc(&d_a, size_a * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_x_kn, size_x * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_x_nk, size_x * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_y, size_y * sizeof(float)));
  h_y = static_cast<float*>(std::malloc(size_y * sizeof(float)));
  if (h_y == nullptr) {
    std::fprintf(stderr, "host malloc failed\n");
    return EXIT_FAILURE;
  }

  int init_threads = 256;
  init_array<<<static_cast<int>((size_a + init_threads - 1) / init_threads), init_threads>>>(d_a, size_a, 0.25f);
  init_array<<<static_cast<int>((size_x + init_threads - 1) / init_threads), init_threads>>>(d_x_kn, size_x, 1.00f);
  init_array<<<static_cast<int>((size_x + init_threads - 1) / init_threads), init_threads>>>(d_x_nk, size_x, 0.00f);
  init_array<<<static_cast<int>((size_y + init_threads - 1) / init_threads), init_threads>>>(d_y, size_y, 0.00f);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  dim3 block(16, 16, 1);
  dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y, 1);

  auto start = std::chrono::high_resolution_clock::now();
  for (int rep = 0; rep < warmup + iters; ++rep) {
    tn_irregular_repack_bad_kernel<<<static_cast<int>((size_x + 255) / 256), 256>>>(
        d_x_kn, d_x_nk, batch, n, k);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    for (int b = 0; b < batch; ++b) {
      tn_irregular_direct_bad_kernel<<<grid, block>>>(d_a, d_x_nk, d_y, m, n, k, b);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Intentional full D2H transfer in-loop to expose orchestration overhead.
    CHECK_CUDA(cudaMemcpy(h_y, d_y, size_y * sizeof(float), cudaMemcpyDeviceToHost));
  }
  auto stop = std::chrono::high_resolution_clock::now();
  double total_ms = std::chrono::duration<double, std::milli>(stop - start).count();
  double avg_step_ms = total_ms / static_cast<double>(warmup + iters);

  double flops_per_iter =
      2.0 * static_cast<double>(batch) * m * n * k;
  double measured_tflops = (flops_per_iter * iters) / (total_ms * 1.0e9);

  std::printf("total_ms,avg_step_ms,executed_tflops,sample\n");
  std::printf("%.6f,%.6f,%.6f,%.6f\n",
              total_ms,
              avg_step_ms,
              measured_tflops,
              static_cast<double>(h_y[0]));

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_x_kn));
  CHECK_CUDA(cudaFree(d_x_nk));
  CHECK_CUDA(cudaFree(d_y));
  std::free(h_y);
  return 0;
}

