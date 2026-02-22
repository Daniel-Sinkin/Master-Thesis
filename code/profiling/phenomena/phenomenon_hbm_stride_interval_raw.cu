#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "bench_common.cuh"

__global__ void init_u64_kernel(std::uint64_t* x, std::size_t n) {
  std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid < n) {
    x[tid] = 0x9E3779B97F4A7C15ULL ^ (tid * 0xBF58476D1CE4E5B9ULL);
  }
}

__global__ void hbm_stride_single_read_kernel(const std::uint64_t* in,
                                              std::uint64_t* out,
                                              std::size_t n_mask,
                                              int stride_elems,
                                              std::size_t access_base) {
  std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::size_t access_id = access_base + tid;
  std::size_t idx = (access_id * static_cast<std::size_t>(stride_elems)) & n_mask;
  out[tid] = in[idx];
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "Usage: %s output_csv [runs warmup launches_per_run threads_per_block "
                 "blocks_per_sm n_pow2]\n",
                 argv[0]);
    std::fprintf(stderr,
                 "Example: %s results/hbm_stride_raw.csv 10 5 64 256 4 30\n",
                 argv[0]);
    return EXIT_FAILURE;
  }

  const std::string output_csv = argv[1];
  int runs = (argc > 2) ? std::atoi(argv[2]) : 10;
  int warmup = (argc > 3) ? std::atoi(argv[3]) : 5;
  int target_launches_per_run = (argc > 4) ? std::atoi(argv[4]) : 64;
  int threads_per_block = (argc > 5) ? std::atoi(argv[5]) : 256;
  int blocks_per_sm = (argc > 6) ? std::atoi(argv[6]) : 4;
  int n_pow2 = (argc > 7) ? std::atoi(argv[7]) : 30;

  if (runs <= 0 || warmup < 0 || target_launches_per_run <= 0 || threads_per_block <= 0 ||
      blocks_per_sm <= 0 || n_pow2 <= 0 || n_pow2 >= 62) {
    std::fprintf(stderr, "Invalid arguments.\n");
    return EXIT_FAILURE;
  }

  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  const int sm_count = prop.multiProcessorCount;
  const int blocks = sm_count * blocks_per_sm;
  const std::size_t total_threads =
      static_cast<std::size_t>(blocks) * static_cast<std::size_t>(threads_per_block);
  const std::size_t n_elems = static_cast<std::size_t>(1) << n_pow2;
  const std::size_t n_mask = n_elems - 1;

  if ((n_elems & (n_elems - 1)) != 0) {
    std::fprintf(stderr, "n_elems must be a power of two.\n");
    return EXIT_FAILURE;
  }

  std::uint64_t* d_in = nullptr;
  std::uint64_t* d_out = nullptr;
  CHECK_CUDA(cudaMalloc(&d_in, n_elems * sizeof(std::uint64_t)));
  CHECK_CUDA(cudaMalloc(&d_out, total_threads * sizeof(std::uint64_t)));

  {
    const int init_threads = 256;
    const int init_blocks = static_cast<int>((n_elems + init_threads - 1) / init_threads);
    init_u64_kernel<<<init_blocks, init_threads>>>(d_in, n_elems);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemset(d_out, 0, total_threads * sizeof(std::uint64_t)));
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  constexpr std::array<int, 11> kStrideBytes{
      8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};

  std::ofstream csv(output_csv);
  if (!csv.is_open()) {
    std::fprintf(stderr, "Failed to open output CSV: %s\n", output_csv.c_str());
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_in));
    return EXIT_FAILURE;
  }

  csv << "device,sm_count,threads_per_block,blocks,total_threads,warmup_runs,"
         "target_launches_per_run,launches_per_run_used,stride_bytes,stride_elements,"
         "run_idx,kernel_ms,read_bytes,achieved_bandwidth_gbps\n";

  std::printf("Writing raw CSV to: %s\n", output_csv.c_str());
  std::printf("device=%s sm=%d blocks=%d threads=%d total_threads=%zu warmup=%d runs=%d "
              "target_launches_per_run=%d working_set=%zu MiB\n",
              prop.name,
              sm_count,
              blocks,
              threads_per_block,
              total_threads,
              warmup,
              runs,
              target_launches_per_run,
              (n_elems * sizeof(std::uint64_t)) >> 20);

  for (int stride_bytes : kStrideBytes) {
    const int stride_elems = stride_bytes / 8;
    const std::size_t access_space = n_elems / static_cast<std::size_t>(stride_elems);
    std::size_t max_launches_no_reuse = access_space / total_threads;
    if (max_launches_no_reuse == 0) {
      max_launches_no_reuse = 1;
    }
    const std::size_t launches_per_run = std::min<std::size_t>(
        static_cast<std::size_t>(target_launches_per_run), max_launches_no_reuse);
    const std::size_t accesses_per_run = launches_per_run * total_threads;
    const std::size_t start_mod = (access_space > accesses_per_run)
                                      ? (access_space - accesses_per_run + 1)
                                      : static_cast<std::size_t>(1);
    const double read_bytes = static_cast<double>(accesses_per_run) *
                              static_cast<double>(sizeof(std::uint64_t));

    auto run_once = [&](std::size_t start_access_id) {
      for (std::size_t launch_idx = 0; launch_idx < launches_per_run; ++launch_idx) {
        std::size_t base = start_access_id + launch_idx * total_threads;
        hbm_stride_single_read_kernel<<<blocks, threads_per_block>>>(
            d_in, d_out, n_mask, stride_elems, base);
      }
    };

    for (int w = 0; w < warmup; ++w) {
      const std::size_t start_access_id =
          (static_cast<std::size_t>(w) * accesses_per_run) % start_mod;
      run_once(start_access_id);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    for (int run = 0; run < runs; ++run) {
      const std::size_t start_access_id =
          (static_cast<std::size_t>(warmup + run) * accesses_per_run) % start_mod;

      GpuTimer timer;
      timer.begin();
      run_once(start_access_id);
      const float ms = timer.end_ms();
      CHECK_CUDA(cudaGetLastError());

      const double bw_gbps = read_bytes / (static_cast<double>(ms) * 1.0e6);

      csv << '"' << prop.name << '"' << ','
          << sm_count << ','
          << threads_per_block << ','
          << blocks << ','
          << total_threads << ','
          << warmup << ','
          << target_launches_per_run << ','
          << launches_per_run << ','
          << stride_bytes << ','
          << stride_elems << ','
          << run << ','
          << ms << ','
          << static_cast<std::uint64_t>(read_bytes) << ','
          << bw_gbps << '\n';
    }
  }

  std::uint64_t sample = 0;
  CHECK_CUDA(cudaMemcpy(&sample, d_out, sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
  std::printf("sample=%llu\n", static_cast<unsigned long long>(sample));

  csv.flush();
  csv.close();

  CHECK_CUDA(cudaFree(d_out));
  CHECK_CUDA(cudaFree(d_in));
  return EXIT_SUCCESS;
}
