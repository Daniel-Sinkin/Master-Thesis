// lib/tensor_stats.cpp
#include "tensor_stats.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include <vecLib/cblas_new.h>

namespace ds_tn {
namespace {

[[nodiscard]] auto as_blas_int(usize value) -> __LAPACK_int {
    if (value > static_cast<usize>(std::numeric_limits<__LAPACK_int>::max())) {
        throw std::overflow_error("Tensor extent exceeds BLAS integer range.");
    }
    return static_cast<__LAPACK_int>(value);
}

auto require_valid_tensor(const Tensor &tensor, const char *function_name) -> void {
    if (tensor.validity() != TensorValidity::valid) {
        throw std::invalid_argument(std::string{function_name} + " requires a valid tensor.");
    }
}

} // namespace

auto l1_norm(const Tensor &tensor) -> f64 {
    require_valid_tensor(tensor, "l1_norm");

    return std::transform_reduce(
        tensor.data(),
        tensor.data() + tensor.size(),
        f64{0.0},
        std::plus<>{},
        [](f64 value) { return std::abs(value); });
}

auto l2_norm(const Tensor &tensor) -> f64 {
    require_valid_tensor(tensor, "l2_norm");
    return cblas_dnrm2(as_blas_int(tensor.size()), tensor.data(), 1);
}

auto lp_norm(const Tensor &tensor, f64 p) -> f64 {
    require_valid_tensor(tensor, "lp_norm");

    if (not std::isfinite(p) or p <= 0.0) {
        throw std::invalid_argument("lp_norm requires a finite p > 0.");
    }

    if (p == 1.0) {
        return l1_norm(tensor);
    }
    if (p == 2.0) {
        return l2_norm(tensor);
    }

    const auto powered_sum = std::transform_reduce(
        tensor.data(),
        tensor.data() + tensor.size(),
        f64{0.0},
        std::plus<>{},
        [p](f64 value) { return std::pow(std::abs(value), p); });

    return std::pow(powered_sum, 1.0 / p);
}

auto infinity_norm(const Tensor &tensor) -> f64 {
    require_valid_tensor(tensor, "infinity_norm");

    if (tensor.size() == 0) {
        throw std::invalid_argument("infinity_norm requires a non-empty tensor.");
    }

    return std::transform_reduce(
        tensor.data(),
        tensor.data() + tensor.size(),
        f64{0.0},
        [](f64 lhs, f64 rhs) { return std::max(lhs, rhs); },
        [](f64 value) { return std::abs(value); });
}

auto element_summary(const Tensor &tensor) -> TensorElementSummary {
    require_valid_tensor(tensor, "element_summary");

    if (tensor.size() == 0) {
        throw std::invalid_argument("element_summary requires a non-empty tensor.");
    }

    const auto begin = tensor.data();
    const auto end = begin + tensor.size();
    const auto [min_it, max_it] = std::minmax_element(begin, end);

    return TensorElementSummary{
        .min = *min_it,
        .max = *max_it,
        .sum = std::accumulate(begin, end, f64{0.0}),
    };
}

} // namespace ds_tn
