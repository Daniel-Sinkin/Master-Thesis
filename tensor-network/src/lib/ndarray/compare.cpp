// lib/ndarray/compare.cpp
#include "ndarray/compare.hpp"

#include <cmath>
#include <numeric>
#include <ranges>

namespace ds_tn {
namespace {

[[nodiscard]] auto comparable(const NDArray &lhs, const NDArray &rhs, f64 tolerance) -> bool {
    return tolerance >= 0.0 and lhs.validity() == NDArrayValidity::valid and rhs.validity() == NDArrayValidity::valid
           and std::ranges::equal(lhs.shape(), rhs.shape());
}

} // namespace

auto close_per_element(const NDArray &lhs, const NDArray &rhs, f64 tolerance) -> bool {
    if (not comparable(lhs, rhs, tolerance)) {
        return false;
    }

    return std::ranges::all_of(iota_n(lhs.size()), [&](usize index) {
        return std::abs(lhs.data()[index] - rhs.data()[index]) <= tolerance;
    });
}

auto close_accumulated(const NDArray &lhs, const NDArray &rhs, f64 tolerance) -> bool {
    if (not comparable(lhs, rhs, tolerance)) {
        return false;
    }

    const auto accumulated_error = std::transform_reduce(
        lhs.data(),
        lhs.data() + lhs.size(),
        rhs.data(),
        f64{0.0},
        std::plus<>{},
        [](f64 lhs_value, f64 rhs_value) { return std::abs(lhs_value - rhs_value); });

    return accumulated_error <= tolerance;
}

} // namespace ds_tn
