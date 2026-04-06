#pragma once

#include "tensor.hpp"

namespace ds_tn {

[[nodiscard]] auto matrix_matrix_product(const Tensor &lhs, const Tensor &rhs) -> Tensor;
[[nodiscard]] auto matrix_vector_product(const Tensor &matrix, const Tensor &vector) -> Tensor;
[[nodiscard]] auto dot_product(const Tensor &lhs, const Tensor &rhs) -> f64;

} // namespace ds_tn
