// lib/tensor/compare.hpp
#pragma once

#include "tensor/tensor.hpp"

namespace ds_tn
{

[[nodiscard]] auto close_per_element(const Tensor& lhs, const Tensor& rhs, f64 tolerance = 1e-12)
    -> bool;
[[nodiscard]] auto close_accumulated(const Tensor& lhs, const Tensor& rhs, f64 tolerance = 1e-12)
    -> bool;
[[nodiscard]] auto is_zero(const Tensor& tensor, f64 tolerance = 1e-12) -> bool;

}  // namespace ds_tn
