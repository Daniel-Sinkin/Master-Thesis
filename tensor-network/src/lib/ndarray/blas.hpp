// lib/ndarray/blas.hpp
#pragma once

#include "ndarray/ndarray.hpp"

namespace ds_tn
{

auto axpy(f64 alpha, const NDArray& x, NDArray& y) -> void;
auto axpy(f64 alpha, const NDArray& x, const NDArray& y, NDArray& out) -> void;
auto gram_matrix(const NDArray& matrix, NDArray& out) -> void;
[[nodiscard]] auto gram_matrix(const NDArray& matrix) -> NDArray;
auto scale_rows(const NDArray& matrix, const NDArray& scales, NDArray& out) -> void;
[[nodiscard]] auto scale_rows(const NDArray& matrix, const NDArray& scales) -> NDArray;
auto scale_cols(const NDArray& matrix, const NDArray& scales, NDArray& out) -> void;
[[nodiscard]] auto scale_cols(const NDArray& matrix, const NDArray& scales) -> NDArray;
auto matrix_matrix_product(const NDArray& lhs, const NDArray& rhs, NDArray& out) -> void;
[[nodiscard]] auto matrix_matrix_product(const NDArray& lhs, const NDArray& rhs) -> NDArray;
auto matrix_vector_product(const NDArray& matrix, const NDArray& vector, NDArray& out) -> void;
[[nodiscard]] auto matrix_vector_product(const NDArray& matrix, const NDArray& vector) -> NDArray;
[[nodiscard]] auto dot_product(const NDArray& lhs, const NDArray& rhs) -> f64;

}  // namespace ds_tn
