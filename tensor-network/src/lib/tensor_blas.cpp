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

} // namespace

auto matrix_matrix_product(const Tensor &lhs, const Tensor &rhs) -> Tensor {
    if (!lhs.is_matrix() || !rhs.is_matrix()) {
        throw std::runtime_error("matrix_matrix_product requires two rank-2 tensors.");
    }
    if (lhs.shape()[1] != rhs.shape()[0]) {
        throw std::runtime_error("matrix_matrix_product requires lhs.cols == rhs.rows.");
    }

    const auto m = as_blas_int(lhs.shape()[0]);
    const auto k = as_blas_int(lhs.shape()[1]);
    const auto n = as_blas_int(rhs.shape()[1]);

    auto out = Tensor({lhs.shape()[0], rhs.shape()[1]});
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

    return out;
}

auto matrix_vector_product(const Tensor &matrix, const Tensor &vector) -> Tensor {
    if (!matrix.is_matrix() || !vector.is_vector()) {
        throw std::runtime_error("matrix_vector_product requires a rank-2 tensor and a rank-1 tensor.");
    }
    if (matrix.shape()[1] != vector.shape()[0]) {
        throw std::runtime_error("matrix_vector_product requires matrix.cols == vector.size.");
    }

    const auto m = as_blas_int(matrix.shape()[0]);
    const auto n = as_blas_int(matrix.shape()[1]);

    auto out = Tensor({matrix.shape()[0]});
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

    return out;
}

auto dot_product(const Tensor &lhs, const Tensor &rhs) -> f64 {
    if (!lhs.is_vector() || !rhs.is_vector()) {
        throw std::runtime_error("dot_product requires two rank-1 tensors.");
    }
    if (lhs.shape()[0] != rhs.shape()[0]) {
        throw std::runtime_error("dot_product requires vectors of the same length.");
    }

    return cblas_ddot(as_blas_int(lhs.shape()[0]), lhs.data(), 1, rhs.data(), 1);
}

} // namespace ds_tn
