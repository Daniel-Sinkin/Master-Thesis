// lib/ndarray/compare.cpp
#include "ndarray/compare.hpp"

#include <cmath>
#include <numeric>
#include <ranges>

namespace ds_tn
{
namespace
{

[[nodiscard]] auto comparable(const NDArray& lhs, const NDArray& rhs, f64 tolerance) -> bool
{
    return tolerance >= 0.0 and lhs.validity() == NDArrayValidity::valid
           and rhs.validity() == NDArrayValidity::valid and lhs.same_shape(rhs);
}

}  // namespace

auto close_per_element(const NDArray& lhs, const NDArray& rhs, f64 tolerance) -> bool
{
    if (not comparable(lhs, rhs, tolerance))
    {
        return false;
    }

    return std::ranges::all_of(
        iota_n(lhs.size()),
        [&](usize index) { return std::abs(lhs.data()[index] - rhs.data()[index]) <= tolerance; }
    );
}

auto close_accumulated(const NDArray& lhs, const NDArray& rhs, f64 tolerance) -> bool
{
    if (not comparable(lhs, rhs, tolerance))
    {
        return false;
    }

    const auto accumulated_error = std::transform_reduce(
        lhs.data(),
        lhs.data() + lhs.size(),
        rhs.data(),
        f64{0.0},
        std::plus<>{},
        [](f64 lhs_value, f64 rhs_value) { return std::abs(lhs_value - rhs_value); }
    );

    return accumulated_error <= tolerance;
}

auto is_symmetric(const NDArray& matrix, f64 tolerance) -> bool
{
    if (tolerance < 0.0 or matrix.validity() != NDArrayValidity::valid or !matrix.is_matrix()
        || matrix.shape(0) != matrix.shape(1))
    {
        return false;
    }

    const auto n = matrix.shape(0);
    for (auto row = 0zu; row < n; ++row)
    {
        for (auto col = row + 1; col < n; ++col)
        {
            if (std::abs(matrix(row, col) - matrix(col, row)) > tolerance)
            {
                return false;
            }
        }
    }

    return true;
}

}  // namespace ds_tn
