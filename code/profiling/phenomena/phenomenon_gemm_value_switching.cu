#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "bench_common.cuh"

#define CHECK_CUBLAS(call)                                                    \
  do {                                                                        \
    cublasStatus_t status__ = (call);                                         \
    if (status__ != CUBLAS_STATUS_SUCCESS) {                                  \
      std::fprintf(stderr,                                                     \
                   "cuBLAS error %d at %s:%d\n",                              \
                   static_cast<int>(status__),                                \
                   __FILE__,                                                   \
                   __LINE__);                                                  \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

enum class InitKind {
  AllZero,
  PeriodicZero,
  Uniform,
  Normal,
};

struct InitMode {
  const char* name;
  InitKind kind;
  int zero_period;
};

constexpr std::array<InitMode, 7> kModes{{
    {"all_zero", InitKind::AllZero, 1},
    {"zero_every_2", InitKind::PeriodicZero, 2},
    {"zero_every_3", InitKind::PeriodicZero, 3},
    {"zero_every_4", InitKind::PeriodicZero, 4},
    {"zero_every_5", InitKind::PeriodicZero, 5},
    {"normal", InitKind::Normal, 0},
    {"uniform", InitKind::Uniform, 0},
}};

std::string to_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

const InitMode* find_mode(const std::string& name) {
  for (const auto& mode : kModes) {
    if (name == mode.name) {
      return &mode;
    }
  }
  return nullptr;
}

double estimate_zero_fraction(const InitMode& mode) {
  switch (mode.kind) {
    case InitKind::AllZero:
      return 1.0;
    case InitKind::PeriodicZero:
      return 1.0 / static_cast<double>(mode.zero_period);
    case InitKind::Uniform:
    case InitKind::Normal:
      return 0.0;
  }
  return 0.0;
}

void fill_matrix(std::vector<float>& x, const InitMode& mode, std::uint64_t seed) {
  std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
  std::normal_distribution<float> norm(0.0f, 1.0f);

  for (std::size_t i = 0; i < x.size(); ++i) {
    switch (mode.kind) {
      case InitKind::AllZero:
        x[i] = 0.0f;
        break;
      case InitKind::PeriodicZero:
        if (i % static_cast<std::size_t>(mode.zero_period) == 0) {
          x[i] = 0.0f;
        } else {
          x[i] = uni(rng);
        }
        break;
      case InitKind::Uniform:
        x[i] = uni(rng);
        break;
      case InitKind::Normal:
        x[i] = norm(rng);
        break;
    }
  }
}

void usage(const char* prog) {
  std::fprintf(stderr,
               "Usage: %s output_csv [m n k warmup iters repeats mode]\n"
               "  mode: all | all_zero | zero_every_2 | zero_every_3 | zero_every_4 | "
               "zero_every_5 | normal | uniform\n"
               "Example:\n"
               "  %s results/09_gemm_value_switching.csv 4096 4096 4096 50 1000 3 all\n",
               prog,
               prog);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    usage(argv[0]);
    return EXIT_FAILURE;
  }

  const std::string output_csv = argv[1];
  const int m = (argc > 2) ? std::atoi(argv[2]) : 4096;
  const int n = (argc > 3) ? std::atoi(argv[3]) : 4096;
  const int k = (argc > 4) ? std::atoi(argv[4]) : 4096;
  const int warmup = (argc > 5) ? std::atoi(argv[5]) : 50;
  const int iters = (argc > 6) ? std::atoi(argv[6]) : 1000;
  const int repeats = (argc > 7) ? std::atoi(argv[7]) : 3;
  const std::string mode_arg = (argc > 8) ? to_lower(argv[8]) : "all";

  if (m <= 0 || n <= 0 || k <= 0 || warmup < 0 || iters <= 0 || repeats <= 0) {
    std::fprintf(stderr, "Invalid numeric arguments.\n");
    return EXIT_FAILURE;
  }

  std::vector<const InitMode*> active_modes;
  if (mode_arg == "all") {
    for (const auto& mode : kModes) {
      active_modes.push_back(&mode);
    }
  } else {
    const InitMode* mode = find_mode(mode_arg);
    if (!mode) {
      std::fprintf(stderr, "Unknown mode: %s\n", mode_arg.c_str());
      usage(argv[0]);
      return EXIT_FAILURE;
    }
    active_modes.push_back(mode);
  }

  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

  const std::size_t size_a = static_cast<std::size_t>(m) * static_cast<std::size_t>(k);
  const std::size_t size_b = static_cast<std::size_t>(k) * static_cast<std::size_t>(n);
  const std::size_t size_c = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);

  std::vector<float> h_a(size_a);
  std::vector<float> h_b(size_b);

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, size_a * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_b, size_b * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_c, size_c * sizeof(float)));

  cublasHandle_t handle{};
  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

  std::ofstream csv(output_csv);
  if (!csv.is_open()) {
    std::fprintf(stderr, "Failed to open output CSV: %s\n", output_csv.c_str());
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_a));
    return EXIT_FAILURE;
  }

  csv << "device,mode,repeat,m,n,k,warmup,iters,zero_fraction_estimate,total_ms,avg_gemm_ms,"
         "throughput_tflops,c_sample\n";

  std::printf("Writing CSV to: %s\n", output_csv.c_str());
  std::printf("device=%s m=%d n=%d k=%d warmup=%d iters=%d repeats=%d active_modes=%zu\n",
              prop.name,
              m,
              n,
              k,
              warmup,
              iters,
              repeats,
              active_modes.size());

  const float alpha = 1.0f;
  const float beta = 0.0f;

  cudaEvent_t ev_start{};
  cudaEvent_t ev_stop{};
  CHECK_CUDA(cudaEventCreate(&ev_start));
  CHECK_CUDA(cudaEventCreate(&ev_stop));

  for (const InitMode* mode : active_modes) {
    for (int rep = 0; rep < repeats; ++rep) {
      const std::uint64_t seed_a = 0x1234abcdULL + static_cast<std::uint64_t>(rep) * 1009ULL;
      const std::uint64_t seed_b = 0x9e3779b9ULL + static_cast<std::uint64_t>(rep) * 2029ULL;

      fill_matrix(h_a, *mode, seed_a);
      fill_matrix(h_b, *mode, seed_b);

      CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), size_a * sizeof(float), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), size_b * sizeof(float), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemset(d_c, 0, size_c * sizeof(float)));

      for (int w = 0; w < warmup; ++w) {
        CHECK_CUBLAS(cublasSgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_a, m, d_b, k, &beta, d_c, m));
      }
      CHECK_CUDA(cudaDeviceSynchronize());

      CHECK_CUDA(cudaEventRecord(ev_start));
      for (int it = 0; it < iters; ++it) {
        CHECK_CUBLAS(cublasSgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_a, m, d_b, k, &beta, d_c, m));
      }
      CHECK_CUDA(cudaEventRecord(ev_stop));
      CHECK_CUDA(cudaEventSynchronize(ev_stop));

      float total_ms = 0.0f;
      CHECK_CUDA(cudaEventElapsedTime(&total_ms, ev_start, ev_stop));

      float c_sample = 0.0f;
      CHECK_CUDA(cudaMemcpy(&c_sample, d_c, sizeof(float), cudaMemcpyDeviceToHost));

      const double total_flops =
          2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k) *
          static_cast<double>(iters);
      const double throughput_tflops = total_flops / (static_cast<double>(total_ms) * 1.0e9);
      const double avg_gemm_ms = static_cast<double>(total_ms) / static_cast<double>(iters);
      const double zero_fraction = estimate_zero_fraction(*mode);

      csv << '"' << prop.name << '"' << ','
          << mode->name << ','
          << rep << ','
          << m << ','
          << n << ','
          << k << ','
          << warmup << ','
          << iters << ','
          << zero_fraction << ','
          << total_ms << ','
          << avg_gemm_ms << ','
          << throughput_tflops << ','
          << c_sample << '\n';

      std::printf("mode=%s rep=%d total_ms=%.3f avg_ms=%.6f tflops=%.3f sample=%.6f\n",
                  mode->name,
                  rep,
                  total_ms,
                  avg_gemm_ms,
                  throughput_tflops,
                  c_sample);
    }
  }

  csv.flush();
  csv.close();

  CHECK_CUDA(cudaEventDestroy(ev_stop));
  CHECK_CUDA(cudaEventDestroy(ev_start));
  CHECK_CUBLAS(cublasDestroy(handle));
  CHECK_CUDA(cudaFree(d_c));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_a));
  return EXIT_SUCCESS;
}
