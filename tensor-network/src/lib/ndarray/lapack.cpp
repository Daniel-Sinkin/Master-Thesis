// lib/ndarray/lapack.cpp
#include "ndarray/lapack.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vecLib/lapack.h>
#include <vector>

namespace ds_tn
{
namespace
{

[[nodiscard]] auto as_lapack_int(usize value) -> __LAPACK_int
{
    if (value > static_cast<usize>(std::numeric_limits<__LAPACK_int>::max()))
    {
        throw std::overflow_error("NDArray extent exceeds LAPACK integer range.");
    }
    return static_cast<__LAPACK_int>(value);
}

auto require_valid_matrix(const NDArray& array, const char* function_name) -> void
{
    if (array.validity() != NDArrayValidity::valid)
    {
        throw std::invalid_argument(std::string{function_name} + " requires a valid NDArray.");
    }
    if (!array.is_matrix())
    {
        throw std::invalid_argument(
            std::string{function_name} + " requires a rank-2 NDArray."
        );
    }
}

auto require_valid_tau(std::span<const f64> tau, usize expected_size, const char* function_name)
    -> void
{
    if (tau.size() != expected_size)
    {
        throw std::invalid_argument(
            std::string{function_name} + " requires tau.size() == min(matrix.rows, matrix.cols)."
        );
    }
}

[[nodiscard]] auto to_column_major(const NDArray& array) -> std::vector<f64>
{
    auto out = std::vector<f64>(array.size());
    for (auto row = 0zu; row < array.shape(0); ++row)
    {
        for (auto col = 0zu; col < array.shape(1); ++col)
        {
            out[row + col * array.shape(0)] = array(row, col);
        }
    }
    return out;
}

[[nodiscard]] auto from_column_major(std::span<const f64> storage, usize rows, usize cols)
    -> NDArray
{
    auto out = NDArray({rows, cols});
    for (auto row = 0zu; row < rows; ++row)
    {
        for (auto col = 0zu; col < cols; ++col)
        {
            out(row, col) = storage[row + col * rows];
        }
    }
    return out;
}

[[nodiscard]] auto apply_matrix_transform(const NDArray& array, MatrixTransform transform) -> NDArray
{
    if (transform == MatrixTransform::transpose)
    {
        return transpose_matrix(array);
    }
    return array;
}

}  // namespace

auto sym_tri_eigendecomp(SymmetricTridiagonalMatrix matrix) -> EigenDecomposition
{
    const auto diagonal = matrix.diagonal;
    const auto off_diagonal = matrix.off_diagonal;

    if (diagonal.empty())
    {
        throw std::invalid_argument(
            "sym_tri_eigendecomp requires diagonal.size() >= 1."
        );
    }
    if (off_diagonal.size() + 1 != diagonal.size())
    {
        throw std::invalid_argument(
            "sym_tri_eigendecomp requires off_diagonal.size() + 1 == diagonal.size()."
        );
    }

    auto eigenvalues_storage = std::vector<f64>(diagonal.begin(), diagonal.end());
    auto off_diagonal_storage = std::vector<f64>(off_diagonal.begin(), off_diagonal.end());
    auto eigenvectors_column_major = std::vector<f64>(diagonal.size() * diagonal.size(), 0.0);
    auto work = std::vector<f64>(std::max<usize>(1, 2 * diagonal.size() - 2), 0.0);

    const auto n = as_lapack_int(diagonal.size());
    const auto ldz = n;
    const auto jobz = 'V';
    auto info = __LAPACK_int{0};

    dstev_(
        &jobz,
        &n,
        eigenvalues_storage.data(),
        diagonal.size() > 1 ? off_diagonal_storage.data() : nullptr,
        eigenvectors_column_major.data(),
        &ldz,
        work.data(),
        &info
    );

    if (info < 0)
    {
        throw std::runtime_error("sym_tri_eigendecomp LAPACK dstev_ received an illegal argument.");
    }
    if (info > 0)
    {
        throw std::runtime_error("sym_tri_eigendecomp LAPACK dstev_ failed to converge.");
    }

    auto eigenvalues = NDArray({diagonal.size()});
    std::ranges::copy(eigenvalues_storage, eigenvalues.data());

    auto eigenvectors = NDArray({diagonal.size(), diagonal.size()});
    for (auto row = 0zu; row < diagonal.size(); ++row)
    {
        for (auto col = 0zu; col < diagonal.size(); ++col)
        {
            eigenvectors(row, col) = eigenvectors_column_major[row + col * diagonal.size()];
        }
    }

    return EigenDecomposition{
        .eigenvalues = std::move(eigenvalues),
        .eigenvectors = std::move(eigenvectors),
    };
}

auto svd(const NDArray& array) -> SVDResult
{
    require_valid_matrix(array, "svd");

    const auto rows = array.shape(0);
    const auto cols = array.shape(1);
    const auto rank = std::min(rows, cols);

    auto a_column_major = to_column_major(array);

    auto singular_values = std::vector<f64>(rank);
    auto u_column_major = std::vector<f64>(rows * rank);
    auto vt_column_major = std::vector<f64>(rank * cols);

    const auto m = as_lapack_int(rows);
    const auto n = as_lapack_int(cols);
    const auto k = as_lapack_int(rank);
    const auto lda = std::max<__LAPACK_int>(1, m);
    const auto ldu = std::max<__LAPACK_int>(1, m);
    const auto ldvt = std::max<__LAPACK_int>(1, k);
    const auto jobu = 'S';
    const auto jobvt = 'S';
    auto info = __LAPACK_int{0};
    auto work_size_query = std::array<f64, 1>{0.0};
    auto lwork = __LAPACK_int{-1};

    dgesvd_(
        &jobu,
        &jobvt,
        &m,
        &n,
        a_column_major.data(),
        &lda,
        singular_values.data(),
        u_column_major.data(),
        &ldu,
        vt_column_major.data(),
        &ldvt,
        work_size_query.data(),
        &lwork,
        &info
    );

    if (info < 0)
    {
        throw std::runtime_error("svd LAPACK dgesvd_ received an illegal argument.");
    }
    if (info > 0)
    {
        throw std::runtime_error("svd LAPACK dgesvd_ failed to converge.");
    }

    lwork = static_cast<__LAPACK_int>(std::max(1.0, std::ceil(work_size_query[0])));
    auto work = std::vector<f64>(static_cast<usize>(lwork));

    dgesvd_(
        &jobu,
        &jobvt,
        &m,
        &n,
        a_column_major.data(),
        &lda,
        singular_values.data(),
        u_column_major.data(),
        &ldu,
        vt_column_major.data(),
        &ldvt,
        work.data(),
        &lwork,
        &info
    );

    if (info < 0)
    {
        throw std::runtime_error("svd LAPACK dgesvd_ received an illegal argument.");
    }
    if (info > 0)
    {
        throw std::runtime_error("svd LAPACK dgesvd_ failed to converge.");
    }

    auto u = from_column_major(u_column_major, rows, rank);

    auto s = NDArray({rank});
    std::ranges::copy(singular_values, s.data());

    auto vt = from_column_major(vt_column_major, rank, cols);

    return SVDResult{
        .u = std::move(u),
        .s = std::move(s),
        .vt = std::move(vt),
    };
}

auto householder_qr(const NDArray& array, MatrixTransform transform) -> HouseholderQR
{
    require_valid_matrix(array, "householder_qr");

    const auto transformed = apply_matrix_transform(array, transform);
    const auto rows = transformed.shape(0);
    const auto cols = transformed.shape(1);
    const auto rank = std::min(rows, cols);

    auto factors_column_major = to_column_major(transformed);
    auto tau_storage = std::vector<f64>(rank);

    const auto m = as_lapack_int(rows);
    const auto n = as_lapack_int(cols);
    const auto lda = std::max<__LAPACK_int>(1, m);
    auto info = __LAPACK_int{0};
    auto work_size_query = std::array<f64, 1>{0.0};
    auto lwork = __LAPACK_int{-1};

    dgeqrf_(
        &m,
        &n,
        factors_column_major.data(),
        &lda,
        tau_storage.data(),
        work_size_query.data(),
        &lwork,
        &info
    );

    if (info < 0)
    {
        throw std::runtime_error(
            "householder_qr LAPACK dgeqrf_ received an illegal argument."
        );
    }

    lwork = static_cast<__LAPACK_int>(std::max(1.0, std::ceil(work_size_query[0])));
    auto work = std::vector<f64>(static_cast<usize>(lwork));

    dgeqrf_(
        &m,
        &n,
        factors_column_major.data(),
        &lda,
        tau_storage.data(),
        work.data(),
        &lwork,
        &info
    );

    if (info < 0)
    {
        throw std::runtime_error(
            "householder_qr LAPACK dgeqrf_ received an illegal argument."
        );
    }

    auto factors = from_column_major(factors_column_major, rows, cols);
    auto tau = NDArray({rank});
    std::ranges::copy(tau_storage, tau.data());

    return HouseholderQR{
        .factors = std::move(factors),
        .tau = std::move(tau),
    };
}

auto extract_upper_triangle(const NDArray& factors) -> NDArray
{
    require_valid_matrix(factors, "extract_upper_triangle");

    const auto rows = factors.shape(0);
    const auto cols = factors.shape(1);
    const auto rank = std::min(rows, cols);

    auto out = NDArray({rank, cols});
    for (auto row = 0zu; row < rank; ++row)
    {
        for (auto col = row; col < cols; ++col)
        {
            out(row, col) = factors(row, col);
        }
    }

    return out;
}

auto householder_build_q(const NDArray& factors, const NDArray& tau) -> NDArray
{
    require_valid_matrix(factors, "householder_build_q");
    if (tau.validity() != NDArrayValidity::valid)
    {
        throw std::invalid_argument("householder_build_q requires tau to be a valid NDArray.");
    }
    if (!tau.is_vector())
    {
        throw std::invalid_argument("householder_build_q requires tau to be a rank-1 NDArray.");
    }

    const auto rows = factors.shape(0);
    const auto cols = factors.shape(1);
    const auto rank = std::min(rows, cols);
    require_valid_tau(
        std::span<const f64>{tau.data(), tau.shape(0)}, rank, "householder_build_q"
    );

    auto q_column_major = std::vector<f64>(rows * rank);
    for (auto row = 0zu; row < rows; ++row)
    {
        for (auto col = 0zu; col < rank; ++col)
        {
            q_column_major[row + col * rows] = factors(row, col);
        }
    }

    const auto m = as_lapack_int(rows);
    const auto n = as_lapack_int(rank);
    const auto k = as_lapack_int(rank);
    const auto lda = std::max<__LAPACK_int>(1, m);
    auto info = __LAPACK_int{0};
    auto work_size_query = std::array<f64, 1>{0.0};
    auto lwork = __LAPACK_int{-1};

    dorgqr_(
        &m,
        &n,
        &k,
        q_column_major.data(),
        &lda,
        const_cast<f64*>(tau.data()),
        work_size_query.data(),
        &lwork,
        &info
    );

    if (info < 0)
    {
        throw std::runtime_error(
            "householder_build_q LAPACK dorgqr_ received an illegal argument."
        );
    }

    lwork = static_cast<__LAPACK_int>(std::max(1.0, std::ceil(work_size_query[0])));
    auto work = std::vector<f64>(static_cast<usize>(lwork));

    dorgqr_(
        &m,
        &n,
        &k,
        q_column_major.data(),
        &lda,
        const_cast<f64*>(tau.data()),
        work.data(),
        &lwork,
        &info
    );

    if (info < 0)
    {
        throw std::runtime_error(
            "householder_build_q LAPACK dorgqr_ received an illegal argument."
        );
    }

    return from_column_major(q_column_major, rows, rank);
}

auto householder_build_q(const HouseholderQR& qr) -> NDArray
{
    return householder_build_q(qr.factors, qr.tau);
}

auto qr(const NDArray& array, MatrixTransform transform) -> QRResult
{
    const auto householder = householder_qr(array, transform);
    return QRResult{
        .q = householder_build_q(householder),
        .r = extract_upper_triangle(householder.factors),
    };
}

}  // namespace ds_tn
