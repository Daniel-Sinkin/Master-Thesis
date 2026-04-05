#include <Accelerate/Accelerate.h>

#include "common.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <print>
#include <ranges>
#include <stdexcept>

namespace ds_tn {

template <typename T>
inline constexpr auto iota_n(std::span<T> x) -> auto { return std::views::iota(0zu, x.size()); }
template <typename T>
inline constexpr auto iota_n(const std::vector<T> &x) -> auto { return std::views::iota(0zu, x.size()); }
inline constexpr auto iota_n(usize n) -> auto { return std::views::iota(0zu, n); }

template <typename A, typename B>
concept compatible_ranges =
    std::ranges::range<A> &&
    std::ranges::range<B> &&
    std::same_as<std::ranges::range_value_t<A>, std::ranges::range_value_t<B>>;

template <typename A, typename B>
    requires compatible_ranges<A, B>
inline constexpr auto inner_product(A &&a, B &&b) {
    using T = std::ranges::range_value_t<A>;
    return std::transform_reduce(
        std::ranges::begin(a), std::ranges::end(a),
        std::ranges::begin(b),
        T{});
}

class Tensor {
public:
    explicit Tensor(std::vector<usize> shape)
        : shape_(std::move(shape)), strides_(shape_.size(), 1) {
        // Reverse cumulative product of shape give
        std::exclusive_scan(shape_.rbegin(), shape_.rend(), strides_.rbegin(), 1zu, std::multiplies<>{});
        data_.resize(strides_[0] * shape_[0]);
    }

    auto rank() const noexcept -> usize { return shape_.size(); }
    auto size() const noexcept -> usize { return data_.size(); }
    auto shape() const noexcept -> std::span<const usize> { return shape_; }
    auto shape_axis() const noexcept -> auto { return iota_n(shape_); }

    auto operator()(std::span<const usize> indices) -> f64 &;
    auto operator()(std::span<const usize> indices) const -> const f64 &;
    auto indices_from_linear(usize linear_index) const -> std::vector<usize>;

private:
    auto linear_index(std::span<const usize> indices) const -> usize;

    std::vector<usize> shape_;
    std::vector<usize> strides_;
    std::vector<f64> data_;
};

auto Tensor::linear_index(std::span<const usize> indices) const -> usize {
    { // Expects
        if (indices.size() != shape_.size()) {
            throw std::invalid_argument("Tensor index rank does not match tensor rank.");
        }
        for (const auto axis : iota_n(indices)) {
            if (indices[axis] >= shape_[axis]) {
                throw std::out_of_range("Tensor index exceeds tensor extent.");
            }
        }
    }
    return inner_product(indices, strides_);
}

auto Tensor::operator()(std::span<const usize> indices) -> f64 & {
    return data_[linear_index(indices)];
}

auto Tensor::operator()(std::span<const usize> indices) const -> const f64 & {
    return data_[linear_index(indices)];
}

auto Tensor::indices_from_linear(usize linear_index) const -> std::vector<usize> {
    if (linear_index >= data_.size()) {
        throw std::out_of_range("Linear index exceeds tensor storage.");
    }

    auto indices = std::vector<usize>(shape_.size(), 0);
    for (const auto axis : iota_n(shape_)) {
        indices[axis] = linear_index / strides_[axis];
        linear_index %= strides_[axis];
    }

    return indices;
}
} // namespace ds_tn

int main() {
    constexpr auto lhs = std::array<double, 6>{
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    };
    constexpr auto rhs = std::array<double, 6>{
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
    };
    auto result = std::array<double, 4>{};
    constexpr auto expected = std::array<double, 4>{
        58.0,
        64.0,
        139.0,
        154.0,
    };
    constexpr auto tolerance = 1.0e-12;
    auto tensor_result = ds_tn::Tensor({2, 2});

    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        2,
        2,
        3,
        1.0,
        lhs.data(),
        3,
        rhs.data(),
        2,
        0.0,
        result.data(),
        2);

    tensor_result(std::array<ds_tn::usize, 2>{0, 0}) = result[0];
    tensor_result(std::array<ds_tn::usize, 2>{0, 1}) = result[1];
    tensor_result(std::array<ds_tn::usize, 2>{1, 0}) = result[2];
    tensor_result(std::array<ds_tn::usize, 2>{1, 1}) = result[3];

    const auto round_trip_indices = tensor_result.indices_from_linear(3);
    const auto is_correct = [&] {
        for (std::size_t index = 0; index < result.size(); ++index) {
            if (std::abs(result[index] - expected[index]) > tolerance) {
                return false;
            }
        }
        return std::abs(tensor_result(std::array<ds_tn::usize, 2>{1, 1}) - expected[3]) <= tolerance &&
               round_trip_indices == std::vector<ds_tn::usize>{1, 1};
    }();

    std::println("Accelerate BLAS dgemm test: {}", is_correct ? "passed" : "failed");
    std::println(
        "[[{:.1f}, {:.1f}], [{:.1f}, {:.1f}]]",
        tensor_result(std::array<ds_tn::usize, 2>{0, 0}),
        tensor_result(std::array<ds_tn::usize, 2>{0, 1}),
        tensor_result(std::array<ds_tn::usize, 2>{1, 0}),
        tensor_result(std::array<ds_tn::usize, 2>{1, 1}));
    std::println("linear index 3 -> indices [{}, {}]", round_trip_indices[0], round_trip_indices[1]);
}
