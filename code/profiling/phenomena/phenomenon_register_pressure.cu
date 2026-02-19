#include <cstdio>
#include <cstdlib>

#include "bench_common.cuh"

__global__ void low_register_kernel(const float* in, float* out, int n, int inner_iters) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }

  float a = in[tid];
  float b = a + 0.1f;
  float c = a + 0.2f;
  float d = a + 0.3f;
#pragma unroll 1
  for (int i = 0; i < inner_iters; ++i) {
    a = fmaf(a, 1.000001f, 0.000001f);
    b = fmaf(b, 0.999999f, 0.000002f);
    c = fmaf(c, 1.000003f, 0.000001f);
    d = fmaf(d, 0.999997f, 0.000003f);
  }
  out[tid] = a + b + c + d;
}

__global__ void high_register_kernel(const float* in,
                                     const float* aux,
                                     float* out,
                                     int n,
                                     int inner_iters) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }

  float base = in[tid];
  float a = base;
  float b = base + 0.1f;
  float c = base + 0.2f;
  float d = base + 0.3f;

  // Keep many values live across the loop to increase register pressure.
  float r0 = base * aux[0];
  float r1 = base * aux[1];
  float r2 = base * aux[2];
  float r3 = base * aux[3];
  float r4 = base * aux[4];
  float r5 = base * aux[5];
  float r6 = base * aux[6];
  float r7 = base * aux[7];
  float r8 = base * aux[8];
  float r9 = base * aux[9];
  float r10 = base * aux[10];
  float r11 = base * aux[11];
  float r12 = base * aux[12];
  float r13 = base * aux[13];
  float r14 = base * aux[14];
  float r15 = base * aux[15];
  float r16 = base * aux[16];
  float r17 = base * aux[17];
  float r18 = base * aux[18];
  float r19 = base * aux[19];
  float r20 = base * aux[20];
  float r21 = base * aux[21];
  float r22 = base * aux[22];
  float r23 = base * aux[23];
  float r24 = base * aux[24];
  float r25 = base * aux[25];
  float r26 = base * aux[26];
  float r27 = base * aux[27];
  float r28 = base * aux[28];
  float r29 = base * aux[29];
  float r30 = base * aux[30];
  float r31 = base * aux[31];

#pragma unroll 1
  for (int i = 0; i < inner_iters; ++i) {
    a = fmaf(a, 1.000001f, 0.000001f);
    b = fmaf(b, 0.999999f, 0.000002f);
    c = fmaf(c, 1.000003f, 0.000001f);
    d = fmaf(d, 0.999997f, 0.000003f);
  }

  float sum = a + b + c + d +
              r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 +
              r8 + r9 + r10 + r11 + r12 + r13 + r14 + r15 +
              r16 + r17 + r18 + r19 + r20 + r21 + r22 + r23 +
              r24 + r25 + r26 + r27 + r28 + r29 + r30 + r31;
  out[tid] = sum;
}

template <typename KernelT, typename... Args>
double run_variant(KernelT kernel,
                   int n,
                   int inner_iters,
                   int warmup,
                   int iters,
                   Args... args) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  auto launch = [&]() { kernel<<<blocks, threads>>>(args..., n, inner_iters); };

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

template <typename KernelT>
double theoretical_occupancy_pct(KernelT kernel, int threads_per_block) {
  int active_blocks_per_sm = 0;
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks_per_sm, kernel, threads_per_block, 0));

  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  double active_warps = static_cast<double>(active_blocks_per_sm * threads_per_block) / 32.0;
  return 100.0 * active_warps / static_cast<double>(prop.maxThreadsPerMultiProcessor / 32);
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
  float* d_aux = nullptr;
  CHECK_CUDA(cudaMalloc(&d_in, static_cast<size_t>(n) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, static_cast<size_t>(n) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_aux, 32 * sizeof(float)));

  float h_aux[32];
  for (int i = 0; i < 32; ++i) {
    h_aux[i] = 1.001f + static_cast<float>(i) * 0.01f;
  }
  CHECK_CUDA(cudaMemcpy(d_aux, h_aux, 32 * sizeof(float), cudaMemcpyHostToDevice));
  init_device_buffer(d_in, n, 1.0f);
  init_device_buffer(d_out, n, 0.0f);
  CHECK_CUDA(cudaDeviceSynchronize());

  double low_ms = run_variant(low_register_kernel, n, inner_iters, warmup, iters, d_in, d_out);
  double high_ms = run_variant(high_register_kernel, n, inner_iters, warmup, iters, d_in, d_aux, d_out);

  double core_flops = static_cast<double>(n) * inner_iters * 8.0;
  double low_tflops = core_flops / (low_ms * 1.0e9);
  double high_tflops = core_flops / (high_ms * 1.0e9);

  double occ_low = theoretical_occupancy_pct(low_register_kernel, 256);
  double occ_high = theoretical_occupancy_pct(high_register_kernel, 256);

  std::printf("variant,avg_ms,core_effective_tflops,theoretical_occupancy_pct\n");
  std::printf("low_register_pressure,%.6f,%.6f,%.2f\n", low_ms, low_tflops, occ_low);
  std::printf("high_register_pressure,%.6f,%.6f,%.2f\n", high_ms, high_tflops, occ_high);

  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_out));
  CHECK_CUDA(cudaFree(d_aux));
  return 0;
}
