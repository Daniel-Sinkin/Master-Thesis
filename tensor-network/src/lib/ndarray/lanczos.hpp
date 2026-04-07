// lib/ndarray/lanczos.hpp
#pragma once

#include "ndarray/ndarray.hpp"

#include <expected>
#include <functional>
#include <optional>
#include <string_view>

namespace ds_tn
{

// Takes in a vector v and returns A * v, A must be symmetrical.
using LanczosApplyAOperator = std::function<NDArray(const NDArray&)>;

struct LanczosResult
{
    NDArray ritz_vector{};
    f64 ritz_value{};
};

enum class LanczosError : u8
{
    not_matrix,
    matrix_not_square,
    matrix_not_symmetric,
    invalid_dimension,
    invalid_iteration_count,
    invalid_operator_output,
};

[[nodiscard]] constexpr auto to_string(LanczosError error) noexcept -> std::string_view
{
    switch (error)
    {
        case LanczosError::not_matrix:
            return "not_matrix";
        case LanczosError::matrix_not_square:
            return "matrix_not_square";
        case LanczosError::matrix_not_symmetric:
            return "matrix_not_symmetric";
        case LanczosError::invalid_dimension:
            return "invalid_dimension";
        case LanczosError::invalid_iteration_count:
            return "invalid_iteration_count";
        case LanczosError::invalid_operator_output:
            return "invalid_operator_output";
    }

    return "unknown_lanczos_error";
}

struct LanczosConfig
{
    usize num_iterations{25};

    RandomOptions random_options{RandomNormalOptions{.mu = 0.0, .sigma = 0.1}};
    std::optional<NDArraySeed> seed{};

    bool do_reorthogonalization{false};

    bool check_symmetric{false};
    f64 symmetry_tolerance{1e-12};

    bool verbose{false};
};

auto lanczos(usize dimension, const LanczosApplyAOperator& apply_A_operator, LanczosConfig cfg = {})
    -> std::expected<LanczosResult, LanczosError>;
auto lanczos(const NDArray& A, LanczosConfig cfg = {})
    -> std::expected<LanczosResult, LanczosError>;

}  // namespace ds_tn
