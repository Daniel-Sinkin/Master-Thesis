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

// Deliberately inefficient: computes Y_{b,p,r} = sum_{q,c} G_{p,q} * X_{b,q,c} * M_{c,r}
// directly in one kernel with poor contraction order and global-memory reuse.
__global__ void tn_two_site_direct_bad_kernel(const float* g,
                                              const float* x,
                                              const float* m,
                                              float* y,
                                              int batch,
                                              int d2,
                                              int chi2) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  int p = blockIdx.y * blockDim.y + threadIdx.y;
  int b = blockIdx.z;

  if (b < batch && p < d2 && r < chi2) {
    const size_t stride = static_cast<size_t>(d2) * chi2;
    const float* x_b = x + static_cast<size_t>(b) * stride;
    float acc = 0.0f;

    for (int q = 0; q < d2; ++q) {
      const float gpq = g[p * d2 + q];
      for (int c = 0; c < chi2; ++c) {
        acc += gpq * x_b[q * chi2 + c] * m[c * chi2 + r];
      }
    }

    y[static_cast<size_t>(b) * stride + p * chi2 + r] = acc;
  }
}

int main(int argc, char** argv) {
  int batch = 512;
  int d2 = 16;
  int chi2 = 256;
  int warmup = 2;
  int iters = 6;

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

  std::printf("Case: BAD (TN two-site direct contraction)\n");
  std::printf("batch=%d d2=%d chi2=%d warmup=%d iters=%d\n",
              batch, d2, chi2, warmup, iters);

  const size_t size_g = static_cast<size_t>(d2) * d2;
  const size_t size_m = static_cast<size_t>(chi2) * chi2;
  const size_t size_x = static_cast<size_t>(batch) * d2 * chi2;

  float* d_g = nullptr;
  float* d_m = nullptr;
  float* d_x = nullptr;
  float* d_y = nullptr;
  CHECK_CUDA(cudaMalloc(&d_g, size_g * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_m, size_m * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_x, size_x * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_y, size_x * sizeof(float)));

  const int init_threads = 256;
  init_array<<<static_cast<int>((size_g + init_threads - 1) / init_threads), init_threads>>>(d_g, size_g, 0.25f);
  init_array<<<static_cast<int>((size_m + init_threads - 1) / init_threads), init_threads>>>(d_m, size_m, 0.50f);
  init_array<<<static_cast<int>((size_x + init_threads - 1) / init_threads), init_threads>>>(d_x, size_x, 1.00f);
  init_array<<<static_cast<int>((size_x + init_threads - 1) / init_threads), init_threads>>>(d_y, size_x, 0.00f);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  dim3 block(16, 16, 1);
  dim3 grid((chi2 + block.x - 1) / block.x,
            (d2 + block.y - 1) / block.y,
            batch);

  auto launch = [&]() {
    tn_two_site_direct_bad_kernel<<<grid, block>>>(d_g, d_x, d_m, d_y, batch, d2, chi2);
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

  // Executed FLOPs of the direct algorithm:
  // 2 * batch * d2 * d2 * chi2 * chi2 per iteration.
  const double flops_per_iter =
      2.0 * static_cast<double>(batch) * d2 * d2 * chi2 * chi2;
  const double tflops = (flops_per_iter * iters) / (static_cast<double>(ms) * 1.0e9);

  float sample = 0.0f;
  CHECK_CUDA(cudaMemcpy(&sample, d_y, sizeof(float), cudaMemcpyDeviceToHost));

  std::printf("Total time: %.3f ms\n", static_cast<double>(ms));
  std::printf("Avg kernel: %.3f ms\n", avg_ms);
  std::printf("Executed throughput: %.2f TFLOP/s\n", tflops);
  std::printf("Sample output: %.6f\n", static_cast<double>(sample));

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(d_g));
  CHECK_CUDA(cudaFree(d_m));
  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_y));
  return 0;
}
