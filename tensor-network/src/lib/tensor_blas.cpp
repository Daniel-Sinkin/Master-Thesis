// lib/tensor_blas.cpp
#include "tensor_blas.hpp"

#include <limits>
#include <stdexcept>

#include <vecLib/cblas_new.h>

namespace ds_tn {
namespace {

[[nodiscard]] auto as_blas_int(usize value) -> __LAPACK_int {
    if (value > static_cast<usize>(std::numeric_limits<__LAPACK_int>::max())) {
        throw std::overflow_error("Tensor extent exceeds BLAS integer range.");
    }
    return static_cast<__LAPACK_int>(value);
}

auto require_valid_tensor(const Tensor &tensor, const char *function_name, const char *argument_name) -> void {
    if (tensor.validity() != TensorValidity::valid) {
        throw std::invalid_argument(
            std::string{function_name} + " requires " + argument_name + " to be a valid tensor.");
    }
}

} // namespace

auto matrix_matrix_product(const Tensor &lhs, const Tensor &rhs, Tensor &out) -> void {
    require_valid_tensor(lhs, "matrix_matrix_product", "lhs");
    require_valid_tensor(rhs, "matrix_matrix_product", "rhs");
    require_valid_tensor(out, "matrix_matrix_product", "out");

    if (not lhs.is_matrix() or not rhs.is_matrix() or not out.is_matrix()) {
        throw std::runtime_error("matrix_matrix_product requires two rank-2 tensors.");
    }
    if (lhs.shape()[1] != rhs.shape()[0]) {
        throw std::runtime_error("matrix_matrix_product requires lhs.cols == rhs.rows.");
    }
    if (out.shape()[0] != lhs.shape()[0] or out.shape()[1] != rhs.shape()[1]) {
        throw std::runtime_error("matrix_matrix_product requires out.shape == {lhs.rows, rhs.cols}.");
    }
    if (&out == &lhs or &out == &rhs) {
        throw std::runtime_error("matrix_matrix_product does not support aliasing out with an input tensor.");
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

auto matrix_matrix_product(const Tensor &lhs, const Tensor &rhs) -> Tensor {
    require_valid_tensor(lhs, "matrix_matrix_product", "lhs");
    require_valid_tensor(rhs, "matrix_matrix_product", "rhs");
    if (not lhs.is_matrix() or not rhs.is_matrix()) {
        throw std::runtime_error("matrix_matrix_product requires two rank-2 tensors.");
    }
    if (lhs.shape()[1] != rhs.shape()[0]) {
        throw std::runtime_error("matrix_matrix_product requires lhs.cols == rhs.rows.");
    }

    auto out = Tensor({lhs.shape()[0], rhs.shape()[1]});
    matrix_matrix_product(lhs, rhs, out);
    return out;
}

auto matrix_vector_product(const Tensor &matrix, const Tensor &vector, Tensor &out) -> void {
    require_valid_tensor(matrix, "matrix_vector_product", "matrix");
    require_valid_tensor(vector, "matrix_vector_product", "vector");
    require_valid_tensor(out, "matrix_vector_product", "out");

    if (not matrix.is_matrix() or not vector.is_vector() or not out.is_vector()) {
        throw std::runtime_error("matrix_vector_product requires a rank-2 tensor and a rank-1 tensor.");
    }
    if (matrix.shape()[1] != vector.shape()[0]) {
        throw std::runtime_error("matrix_vector_product requires matrix.cols == vector.size.");
    }
    if (out.shape()[0] != matrix.shape()[0]) {
        throw std::runtime_error("matrix_vector_product requires out.shape == {matrix.rows}.");
    }
    if (&out == &matrix or &out == &vector) {
        throw std::runtime_error("matrix_vector_product does not support aliasing out with an input tensor.");
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

auto matrix_vector_product(const Tensor &matrix, const Tensor &vector) -> Tensor {
    require_valid_tensor(matrix, "matrix_vector_product", "matrix");
    require_valid_tensor(vector, "matrix_vector_product", "vector");
    if (not matrix.is_matrix() or not vector.is_vector()) {
        throw std::runtime_error("matrix_vector_product requires a rank-2 tensor and a rank-1 tensor.");
    }
    if (matrix.shape()[1] != vector.shape()[0]) {
        throw std::runtime_error("matrix_vector_product requires matrix.cols == vector.size.");
    }

    auto out = Tensor({matrix.shape()[0]});
    matrix_vector_product(matrix, vector, out);
    return out;
}

auto dot_product(const Tensor &lhs, const Tensor &rhs) -> f64 {
    require_valid_tensor(lhs, "dot_product", "lhs");
    require_valid_tensor(rhs, "dot_product", "rhs");

    if (not lhs.is_vector() or not rhs.is_vector()) {
        throw std::runtime_error("dot_product requires two rank-1 tensors.");
    }
    if (lhs.shape()[0] != rhs.shape()[0]) {
        throw std::runtime_error("dot_product requires vectors of the same length.");
    }

    return cblas_ddot(as_blas_int(lhs.shape()[0]), lhs.data(), 1, rhs.data(), 1);
}

} // namespace ds_tn
