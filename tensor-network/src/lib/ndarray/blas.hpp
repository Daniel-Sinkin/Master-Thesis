// lib/ndarray/blas.hpp
#pragma once

#include "ndarray/ndarray.hpp"

namespace ds_tn {

auto axpy(NDArray &y, f64 alpha, const NDArray &x) -> void;
auto axpy(const NDArray &y, f64 alpha, const NDArray &x, NDArray &out) -> void;
auto matrix_matrix_product(const NDArray &lhs, const NDArray &rhs, NDArray &out) -> void;
[[nodiscard]] auto matrix_matrix_product(const NDArray &lhs, const NDArray &rhs) -> NDArray;
auto matrix_vector_product(const NDArray &matrix, const NDArray &vector, NDArray &out) -> void;
[[nodiscard]] auto matrix_vector_product(const NDArray &matrix, const NDArray &vector) -> NDArray;
[[nodiscard]] auto dot_product(const NDArray &lhs, const NDArray &rhs) -> f64;

} // namespace ds_tn
