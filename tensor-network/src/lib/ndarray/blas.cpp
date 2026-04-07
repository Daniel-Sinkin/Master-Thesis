// lib/ndarray/blas.cpp
#include "ndarray/blas.hpp"

#include <limits>
#include <ranges>
#include <stdexcept>
#include <string>

#include <vecLib/cblas_new.h>

namespace ds_tn {
namespace {

[[nodiscard]] auto as_blas_int(usize value) -> __LAPACK_int {
    if (value > static_cast<usize>(std::numeric_limits<__LAPACK_int>::max())) {
        throw std::overflow_error("NDArray extent exceeds BLAS integer range.");
    }
    return static_cast<__LAPACK_int>(value);
}

auto require_valid_array(const NDArray &array, const char *function_name, const char *argument_name) -> void {
    if (array.validity() != NDArrayValidity::valid) {
        throw std::invalid_argument(
            std::string{function_name} + " requires " + argument_name + " to be a valid NDArray.");
    }
}

auto require_same_shape(const NDArray &lhs, const NDArray &rhs, const char *function_name) -> void {
    if (not std::ranges::equal(lhs.shape(), rhs.shape())) {
        throw std::runtime_error(std::string{function_name} + " requires NDArrays with identical shapes.");
    }
}

} // namespace

auto axpy(f64 alpha, const NDArray &x, NDArray &y) -> void {
    require_valid_array(y, "axpy", "y");
    require_valid_array(x, "axpy", "x");
    require_same_shape(y, x, "axpy");

    if (&y == &x) {
        y.multiply_scalar(1.0 + alpha);
        return;
    }

    cblas_daxpy(as_blas_int(y.size()), alpha, x.data(), 1, y.data(), 1);
}

auto axpy(f64 alpha, const NDArray &x, const NDArray &y, NDArray &out) -> void {
    require_valid_array(y, "axpy", "y");
    require_valid_array(x, "axpy", "x");
    require_valid_array(out, "axpy", "out");
    require_same_shape(y, x, "axpy");
    require_same_shape(y, out, "axpy");

    if (&out == &y) {
        axpy(alpha, x, out);
        return;
    }

    if (&out == &x) {
        out.multiply_scalar(alpha);
        axpy(1.0, y, out);
        return;
    }

    std::ranges::copy(y.data(), y.data() + y.size(), out.data());
    cblas_daxpy(as_blas_int(out.size()), alpha, x.data(), 1, out.data(), 1);
}

auto matrix_matrix_product(const NDArray &lhs, const NDArray &rhs, NDArray &out) -> void {
    require_valid_array(lhs, "matrix_matrix_product", "lhs");
    require_valid_array(rhs, "matrix_matrix_product", "rhs");
    require_valid_array(out, "matrix_matrix_product", "out");

    if (not lhs.is_matrix() or not rhs.is_matrix() or not out.is_matrix()) {
        throw std::runtime_error("matrix_matrix_product requires two rank-2 NDArrays.");
    }
    if (lhs.shape()[1] != rhs.shape()[0]) {
        throw std::runtime_error("matrix_matrix_product requires lhs.cols == rhs.rows.");
    }
    if (out.shape()[0] != lhs.shape()[0] or out.shape()[1] != rhs.shape()[1]) {
        throw std::runtime_error("matrix_matrix_product requires out.shape == {lhs.rows, rhs.cols}.");
    }
    if (&out == &lhs or &out == &rhs) {
        throw std::runtime_error("matrix_matrix_product does not support aliasing out with an input NDArray.");
    }

    const auto m = as_blas_int(lhs.shape()[0]);
    const auto k = as_blas_int(lhs.shape()[1]);
    const auto n = as_blas_int(rhs.shape()[1]);

    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        m,
        n,
        k,
        1.0,
        lhs.data(),
        k,
        rhs.data(),
        n,
        0.0,
        out.data(),
        n);
}

auto matrix_matrix_product(const NDArray &lhs, const NDArray &rhs) -> NDArray {
    require_valid_array(lhs, "matrix_matrix_product", "lhs");
    require_valid_array(rhs, "matrix_matrix_product", "rhs");
    if (not lhs.is_matrix() or not rhs.is_matrix()) {
        throw std::runtime_error("matrix_matrix_product requires two rank-2 NDArrays.");
    }
    if (lhs.shape()[1] != rhs.shape()[0]) {
        throw std::runtime_error("matrix_matrix_product requires lhs.cols == rhs.rows.");
    }

    auto out = NDArray({lhs.shape()[0], rhs.shape()[1]});
    matrix_matrix_product(lhs, rhs, out);
    return out;
}

auto matrix_vector_product(const NDArray &matrix, const NDArray &vector, NDArray &out) -> void {
    require_valid_array(matrix, "matrix_vector_product", "matrix");
    require_valid_array(vector, "matrix_vector_product", "vector");
    require_valid_array(out, "matrix_vector_product", "out");

    if (not matrix.is_matrix() or not vector.is_vector() or not out.is_vector()) {
        throw std::runtime_error("matrix_vector_product requires a rank-2 NDArray and a rank-1 NDArray.");
    }
    if (matrix.shape()[1] != vector.shape()[0]) {
        throw std::runtime_error("matrix_vector_product requires matrix.cols == vector.size.");
    }
    if (out.shape()[0] != matrix.shape()[0]) {
        throw std::runtime_error("matrix_vector_product requires out.shape == {matrix.rows}.");
    }
    if (&out == &matrix or &out == &vector) {
        throw std::runtime_error("matrix_vector_product does not support aliasing out with an input NDArray.");
    }

    const auto m = as_blas_int(matrix.shape()[0]);
    const auto n = as_blas_int(matrix.shape()[1]);

    cblas_dgemv(
        CblasRowMajor,
        CblasNoTrans,
        m,
        n,
        1.0,
        matrix.data(),
        n,
        vector.data(),
        1,
        0.0,
        out.data(),
        1);
}

auto matrix_vector_product(const NDArray &matrix, const NDArray &vector) -> NDArray {
    require_valid_array(matrix, "matrix_vector_product", "matrix");
    require_valid_array(vector, "matrix_vector_product", "vector");
    if (not matrix.is_matrix() or not vector.is_vector()) {
        throw std::runtime_error("matrix_vector_product requires a rank-2 NDArray and a rank-1 NDArray.");
    }
    if (matrix.shape()[1] != vector.shape()[0]) {
        throw std::runtime_error("matrix_vector_product requires matrix.cols == vector.size.");
    }

    auto out = NDArray({matrix.shape()[0]});
    matrix_vector_product(matrix, vector, out);
    return out;
}

auto dot_product(const NDArray &lhs, const NDArray &rhs) -> f64 {
    require_valid_array(lhs, "dot_product", "lhs");
    require_valid_array(rhs, "dot_product", "rhs");
    require_same_shape(lhs, rhs, "dot_product");

    return cblas_ddot(as_blas_int(lhs.size()), lhs.data(), 1, rhs.data(), 1);
}

} // namespace ds_tn
