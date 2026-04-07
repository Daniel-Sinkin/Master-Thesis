// lib/ndarray/compare.hpp
#pragma once

#include "ndarray/ndarray.hpp"

namespace ds_tn
{

[[nodiscard]] auto close_per_element(const NDArray& lhs, const NDArray& rhs, f64 tolerance = 1e-12)
    -> bool;
[[nodiscard]] auto close_accumulated(const NDArray& lhs, const NDArray& rhs, f64 tolerance = 1e-12)
    -> bool;
[[nodiscard]] auto is_symmetric(const NDArray& matrix, f64 tolerance = 1e-12) -> bool;

}  // namespace ds_tn
