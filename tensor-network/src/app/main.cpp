// app/main.cpp
#include "tensor.hpp"
#include "tensor_blas.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <print>
#include <stdexcept>

namespace {
using namespace ds_tn;
template <usize N>
[[maybe_unused]] auto copy_into(Tensor &tensor, const std::array<double, N> &values) -> void {
    if (tensor.size() != N) {
        throw std::runtime_error("Tensor size does not match initializer data.");
    }
    std::ranges::copy(std::move(values), tensor.data());
}

[[maybe_unused]] auto copy_into(Tensor &tensor, std::initializer_list<double> values) -> void {
    if (tensor.size() != values.size()) {
        throw std::runtime_error("Tensor size does not match initializer data.");
    }
    std::ranges::copy(values, tensor.data());
}

[[maybe_unused]] [[nodiscard]] auto near(double lhs, double rhs, double tolerance) -> bool {
    return std::abs(lhs - rhs) <= tolerance;
}
} // namespace

auto main() -> int {
    using namespace ds_tn;

    constexpr auto expected_matrix = std::array<double, 4>{
        58.0,
        64.0,
        139.0,
        154.0,
    };
    constexpr auto expected_vector = std::array<double, 2>{14.0, 32.0};
    constexpr auto expected_dot = 32.0;
    constexpr auto tolerance = 1.0e-12;

    Tensor lhs{{2, 3}};
    copy_into(lhs, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

    Tensor rhs{{3, 2}};
    copy_into(rhs, {7.0, 8.0, 9.0, 10.0, 11.0, 12.0});

    Tensor vector{{3}};
    copy_into(vector, {1.0, 2.0, 3.0});

    Tensor other_vector{{3}};
    copy_into(other_vector, {4.0, 5.0, 6.0});

    const auto matrix_result = matrix_matrix_product(lhs, rhs);
    const auto vector_result = matrix_vector_product(lhs, vector);
    const auto dot_result = dot_product(vector, other_vector);
    const auto round_trip_indices = matrix_result.indices_from_linear(3);

    auto scalar = Tensor({});
    scalar(std::span<const usize>{}) = 42.0;
    const auto scalar_indices = scalar.indices_from_linear(0);

    auto rank3 = Tensor({2, 2, 2});
    copy_into(rank3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});

    const auto is_correct = [&] {
        return near(matrix_result(std::array<usize, 2>{0, 0}), expected_matrix[0], tolerance) &&
               near(matrix_result(std::array<usize, 2>{0, 1}), expected_matrix[1], tolerance) &&
               near(matrix_result(std::array<usize, 2>{1, 0}), expected_matrix[2], tolerance) &&
               near(matrix_result(std::array<usize, 2>{1, 1}), expected_matrix[3], tolerance) &&
               near(vector_result(std::array<usize, 1>{0}), expected_vector[0], tolerance) &&
               near(vector_result(std::array<usize, 1>{1}), expected_vector[1], tolerance) &&
               near(dot_result, expected_dot, tolerance) &&
               round_trip_indices == std::vector<usize>{1, 1} &&
               scalar.is_trivial() &&
               scalar.size() == 1 &&
               scalar_indices.empty() &&
               near(scalar(std::span<const usize>{}), 42.0, tolerance);
    }();

    std::println("Accelerate BLAS tensor test: {}", is_correct ? "passed" : "failed");
    std::println("Matrix-matrix product:");
    matrix_result.print(4, false);
    std::println("Matrix-vector product:");
    vector_result.print();
    std::println("Dot product: {:.4f}", dot_result);
    std::println("Rank-0 tensor:");
    scalar.print();
    std::println("Rank-3 tensor:");
    rank3.print(2);
    std::println("linear index 3 -> indices [{}, {}]", round_trip_indices[0], round_trip_indices[1]);

    return is_correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
