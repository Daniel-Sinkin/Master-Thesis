#include <cstdio>
#include <cstdlib>

#include "bench_common.cuh"

__global__ void dependent_chain_kernel(const float* in, float* out, int n, int inner_iters) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }
  float a = in[tid];
#pragma unroll 1
  for (int i = 0; i < inner_iters; ++i) {
    a = fmaf(a, 1.000001f, 0.000001f);
    a = fmaf(a, 0.999999f, 0.000002f);
    a = fmaf(a, 1.000003f, 0.000001f);
    a = fmaf(a, 0.999997f, 0.000003f);
  }
  out[tid] = a;
}

__global__ void independent_ilp_kernel(const float* in, float* out, int n, int inner_iters) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }
  float a = in[tid];
  float b = in[tid] + 0.1f;
  float c = in[tid] + 0.2f;
  float d = in[tid] + 0.3f;
#pragma unroll 1
  for (int i = 0; i < inner_iters; ++i) {
    a = fmaf(a, 1.000001f, 0.000001f);
    b = fmaf(b, 0.999999f, 0.000002f);
    c = fmaf(c, 1.000003f, 0.000001f);
    d = fmaf(d, 0.999997f, 0.000003f);
  }
  out[tid] = a + b + c + d;
}

template <typename KernelT>
double run_variant(KernelT kernel,
                   const float* d_in,
                   float* d_out,
                   int n,
                   int inner_iters,
                   int warmup,
                   int iters) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  auto launch = [&]() { kernel<<<blocks, threads>>>(d_in, d_out, n, inner_iters); };

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
  return static_cast<double>(ms) / iters;
}

int main(int argc, char** argv) {
  int n = 1 << 22;
  int inner_iters = 256;
  int warmup = 10;
  int iters = 40;

  if (argc == 5) {
    n = std::atoi(argv[1]);
    inner_iters = std::atoi(argv[2]);
    warmup = std::atoi(argv[3]);
    iters = std::atoi(argv[4]);
  } else if (argc != 1) {
    std::fprintf(stderr, "Usage: %s [N inner_iters warmup iters]\n", argv[0]);
    return EXIT_FAILURE;
  }

  float* d_in = nullptr;
  float* d_out = nullptr;
  CHECK_CUDA(cudaMalloc(&d_in, static_cast<size_t>(n) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, static_cast<size_t>(n) * sizeof(float)));
  init_device_buffer(d_in, n, 1.0f);
  init_device_buffer(d_out, n, 0.0f);
  CHECK_CUDA(cudaDeviceSynchronize());

  double dep_ms = run_variant(dependent_chain_kernel, d_in, d_out, n, inner_iters, warmup, iters);
  double ilp_ms = run_variant(independent_ilp_kernel, d_in, d_out, n, inner_iters, warmup, iters);

  double flops = static_cast<double>(n) * inner_iters * 8.0;
  double dep_tflops = flops / (dep_ms * 1.0e9);
  double ilp_tflops = flops / (ilp_ms * 1.0e9);

  std::printf("variant,avg_ms,effective_tflops\n");
  std::printf("dependent_chain,%.6f,%.6f\n", dep_ms, dep_tflops);
  std::printf("independent_ilp,%.6f,%.6f\n", ilp_ms, ilp_tflops);

  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_out));
  return 0;
}
