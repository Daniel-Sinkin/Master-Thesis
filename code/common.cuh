// common.cuh
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <utility>

using usize = std::size_t;
using isize = std::ptrdiff_t;

using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using f32 = float;
using f64 = double;

constexpr auto div_ceil(int a, int b) -> int {
    return (a + b - 1) / b;
}

template <typename T, usize Rank>
class Tensor {
public:
    template <typename... Dims>
        requires(sizeof...(Dims) == Rank && (std::is_convertible_v<Dims, usize> && ...))
    explicit Tensor(Dims... dims) : extents_{static_cast<usize>(dims)...} {
        auto res = std::malloc(bytes());
        if (!res) {
            std::terminate();
        }
        data_ = static_cast<T *>(res);
        cudaMalloc(&data_device_, bytes());
    }

    ~Tensor() {
        std::free(data_);
        cudaFree(data_device_);
    }

    Tensor(const Tensor &) = delete;
    auto operator=(const Tensor &) -> Tensor & = delete;

    Tensor(Tensor &&o) noexcept
        : data_{std::exchange(o.data_, nullptr)}, data_device_{std::exchange(o.data_device_, nullptr)}, extents_{o.extents_} {}

    auto upload() -> void {
        cudaMemcpy(data_device_, data_, bytes(), cudaMemcpyHostToDevice);
    }

    auto download() -> void {
        cudaMemcpy(data_, data_device_, bytes(), cudaMemcpyDeviceToHost);
    }

    [[nodiscard]] auto extents() const -> const std::array<usize, Rank> & { return extents_; }
    [[nodiscard]] auto extent(usize i) const -> usize { return extents_[i]; }

    [[nodiscard]] auto data() const -> const T * { return data_; }
    [[nodiscard]] auto data() -> T * { return data_; }

    [[nodiscard]] auto device() const -> const T * { return data_device_; }
    [[nodiscard]] auto device() -> T * { return data_device_; }

    [[nodiscard]] constexpr auto size() const -> usize {
        return std::accumulate(extents_.begin(), extents_.end(), usize{1}, std::multiplies<>{});
    }

    [[nodiscard]] auto begin() -> T * { return data_; }
    [[nodiscard]] auto end() -> T * { return data_ + size(); }
    [[nodiscard]] auto begin() const -> const T * { return data_; }
    [[nodiscard]] auto end() const -> const T * { return data_ + size(); }

    [[nodiscard]] constexpr auto bytes() const -> usize {
        return size() * sizeof(T);
    }

private:
    T *data_{};
    T *data_device_{};
    std::array<usize, Rank> extents_{};
};

template <typename T, usize Rank>
void print(const Tensor<T, Rank> &t, usize max_per_dim = 4) {
    if constexpr (Rank == 1) {
        auto n = std::min(t.extent(0), max_per_dim);
        std::cout << "[";
        for (auto i = usize{0}; i < n; ++i) {
            if (i > 0)
                std::cout << ", ";
            std::cout << t.data()[i];
        }
        if (t.extent(0) > max_per_dim)
            std::cout << ", ...";
        std::cout << "]\n";
    } else if constexpr (Rank == 2) {
        auto rows = std::min(t.extent(0), max_per_dim);
        auto cols = std::min(t.extent(1), max_per_dim);
        std::cout << "Tensor<" << Rank << "> [" << t.extent(0) << " x " << t.extent(1) << "]\n";
        for (auto i = usize{0}; i < rows; ++i) {
            std::cout << "  [";
            for (auto j = usize{0}; j < cols; ++j) {
                if (j > 0)
                    std::cout << ", ";
                std::cout << t.data()[i * t.extent(1) + j];
            }
            if (t.extent(1) > max_per_dim)
                std::cout << ", ...";
            std::cout << "]\n";
        }
        if (t.extent(0) > max_per_dim)
            std::cout << "  ...\n";
    } else {
        std::cout << "Tensor<" << Rank << "> [";
        for (auto i = usize{0}; i < Rank; ++i) {
            if (i > 0)
                std::cout << " x ";
            std::cout << t.extent(i);
        }
        std::cout << "] (" << t.size() << " elements)\n";
        auto n = std::min(t.size(), max_per_dim);
        std::cout << "  flat: [";
        for (auto i = usize{0}; i < n; ++i) {
            if (i > 0)
                std::cout << ", ";
            std::cout << t.data()[i];
        }
        if (t.size() > max_per_dim)
            std::cout << ", ...";
        std::cout << "]\n";
    }
}

__global__ void sgemm_naive(
    int M, int N, int K,
    f32 alpha, f32 beta,
    const f32 *A, const f32 *B, f32 *C) {
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < M && y < N) {
        auto tmp = 0.0f;
        for (auto i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

int main() {
    const auto M = 32;
    const auto N = 96;
    const auto K = 32;
    const auto alpha = 1.0f;
    const auto beta = 0.0f;

    auto A = Tensor<f32, 2>(M, K);
    auto B = Tensor<f32, 2>(K, N);
    auto C = Tensor<f32, 2>(M, N);

    std::fill(A.begin(), A.end(), 1.0f);
    std::fill(B.begin(), B.end(), -2.0f);
    std::fill(C.begin(), C.end(), 0.0f);

    A.upload();
    B.upload();
    C.upload();

    const auto block = dim3(16, 16);
    const auto grid = dim3(div_ceil(M, 16), div_ceil(N, 16));
    sgemm_naive<<<grid, block>>>(M, N, K, alpha, beta, A.device(), B.device(), C.device());
    cudaDeviceSynchronize();

    C.download();
    print(C);
}
