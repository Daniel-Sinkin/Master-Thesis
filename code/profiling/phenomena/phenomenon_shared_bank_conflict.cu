#include <cstdio>
#include <cstdlib>

#include "bench_common.cuh"

template <int PAD>
__global__ void shared_transpose_kernel(const float* in,
                                        float* out,
                                        int tile_repeats) {
  __shared__ float tile[32][32 + PAD];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tile_id = blockIdx.x;
  int idx = tile_id * (32 * 32) + ty * 32 + tx;

  float v = in[idx];
#pragma unroll 1
  for (int rep = 0; rep < tile_repeats; ++rep) {
    tile[ty][tx] = v;
    __syncthreads();
    v = tile[tx][ty];
    __syncthreads();
  }
  out[idx] = v;
}

template <int PAD>
double run_variant(const float* d_in,
                   float* d_out,
                   int n_tiles,
                   int tile_repeats,
                   int warmup,
                   int iters) {
  dim3 block(32, 32, 1);
  dim3 grid(n_tiles, 1, 1);
  auto launch = [&]() {
    shared_transpose_kernel<PAD><<<grid, block>>>(d_in, d_out, tile_repeats);
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
  return static_cast<double>(ms) / iters;
}

int main(int argc, char** argv) {
  int n_tiles = 4096;
  int tile_repeats = 64;
  int warmup = 3;
  int iters = 15;

  if (argc == 5) {
    n_tiles = std::atoi(argv[1]);
    tile_repeats = std::atoi(argv[2]);
    warmup = std::atoi(argv[3]);
    iters = std::atoi(argv[4]);
  } else if (argc != 1) {
    std::fprintf(stderr, "Usage: %s [n_tiles tile_repeats warmup iters]\n", argv[0]);
    return EXIT_FAILURE;
  }

  int n = n_tiles * 32 * 32;
  float* d_in = nullptr;
  float* d_out = nullptr;
  CHECK_CUDA(cudaMalloc(&d_in, static_cast<size_t>(n) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, static_cast<size_t>(n) * sizeof(float)));
  init_device_buffer(d_in, n, 1.0f);
  init_device_buffer(d_out, n, 0.0f);
  CHECK_CUDA(cudaDeviceSynchronize());

  double avg_ms_conflict = run_variant<0>(d_in, d_out, n_tiles, tile_repeats, warmup, iters);
  double avg_ms_padded = run_variant<1>(d_in, d_out, n_tiles, tile_repeats, warmup, iters);

  double bytes = static_cast<double>(n) * sizeof(float) * 2.0 * tile_repeats;
  double bw_conflict = bytes / (avg_ms_conflict * 1.0e6);
  double bw_padded = bytes / (avg_ms_padded * 1.0e6);

  std::printf("variant,avg_ms,effective_bw_gbps\n");
  std::printf("conflict,%.6f,%.6f\n", avg_ms_conflict, bw_conflict);
  std::printf("padded,%.6f,%.6f\n", avg_ms_padded, bw_padded);

  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_out));
  return 0;
}
