// lib/ndarray/lanczos.hpp
#pragma once

#include "ndarray/ndarray.hpp"

#include <functional>
#include <optional>

namespace ds_tn
{

// Takes in a vector v and returns A * v, A must be symmetrical.
using LanczosApplyAOperator = std::function<NDArray(const NDArray&)>;

struct LanczosResult
{
    NDArray ritz_vector{};
    f64 ritz_value{};
};

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
    -> LanczosResult;
auto lanczos(const NDArray& A, LanczosConfig cfg = {}) -> LanczosResult;

}  // namespace ds_tn
