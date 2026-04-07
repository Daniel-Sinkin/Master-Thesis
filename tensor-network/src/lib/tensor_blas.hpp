// lib/tensor_blas.hpp
#pragma once

#include "tensor.hpp"

namespace ds_tn {

auto matrix_matrix_product(const Tensor &lhs, const Tensor &rhs, Tensor &out) -> void;
[[nodiscard]] auto matrix_matrix_product(const Tensor &lhs, const Tensor &rhs) -> Tensor;
auto matrix_vector_product(const Tensor &matrix, const Tensor &vector, Tensor &out) -> void;
[[nodiscard]] auto matrix_vector_product(const Tensor &matrix, const Tensor &vector) -> Tensor;
[[nodiscard]] auto dot_product(const Tensor &lhs, const Tensor &rhs) -> f64;

} // namespace ds_tn
