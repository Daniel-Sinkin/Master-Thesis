// lib/ndarray/stats.cpp
#include "ndarray/stats.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vecLib/cblas_new.h>

namespace ds_tn
{
namespace
{

[[nodiscard]] auto as_blas_int(usize value) -> __LAPACK_int
{
    if (value > static_cast<usize>(std::numeric_limits<__LAPACK_int>::max()))
    {
        throw std::overflow_error("NDArray extent exceeds BLAS integer range.");
    }
    return static_cast<__LAPACK_int>(value);
}

auto require_valid_array(const NDArray& array, const char* function_name) -> void
{
    if (array.validity() != NDArrayValidity::valid)
    {
        throw std::invalid_argument(std::string{function_name} + " requires a valid NDArray.");
    }
}

}  // namespace

auto l1_norm(const NDArray& array) -> f64
{
    require_valid_array(array, "l1_norm");

    return std::transform_reduce(
        array.data(),
        array.data() + array.size(),
        f64{0.0},
        std::plus<>{},
        [](f64 value) { return std::abs(value); }
    );
}

auto l2_norm(const NDArray& array) -> f64
{
    require_valid_array(array, "l2_norm");
    return cblas_dnrm2(as_blas_int(array.size()), array.data(), 1);
}

auto lp_norm(const NDArray& array, f64 p) -> f64
{
    require_valid_array(array, "lp_norm");

    if (not std::isfinite(p) or p <= 0.0)
    {
        throw std::invalid_argument("lp_norm requires a finite p > 0.");
    }

    if (p == 1.0)
    {
        return l1_norm(array);
    }
    if (p == 2.0)
    {
        return l2_norm(array);
    }

    const auto powered_sum = std::transform_reduce(
        array.data(),
        array.data() + array.size(),
        f64{0.0},
        std::plus<>{},
        [p](f64 value) { return std::pow(std::abs(value), p); }
    );

    return std::pow(powered_sum, 1.0 / p);
}

auto infinity_norm(const NDArray& array) -> f64
{
    require_valid_array(array, "infinity_norm");

    if (array.size() == 0)
    {
        throw std::invalid_argument("infinity_norm requires a non-empty NDArray.");
    }

    return std::transform_reduce(
        array.data(),
        array.data() + array.size(),
        f64{0.0},
        [](f64 lhs, f64 rhs) { return std::max(lhs, rhs); },
        [](f64 value) { return std::abs(value); }
    );
}

auto element_summary(const NDArray& array) -> NDArrayElementSummary
{
    require_valid_array(array, "element_summary");

    if (array.size() == 0)
    {
        throw std::invalid_argument("element_summary requires a non-empty NDArray.");
    }

    const auto begin = array.data();
    const auto end = begin + array.size();
    const auto [min_it, max_it] = std::minmax_element(begin, end);

    return NDArrayElementSummary{
        .min = *min_it,
        .max = *max_it,
        .sum = std::accumulate(begin, end, f64{0.0}),
    };
}

}  // namespace ds_tn
