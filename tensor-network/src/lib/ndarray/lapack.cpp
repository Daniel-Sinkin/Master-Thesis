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

}  // namespace

auto sym_tri_eigendecomp(SymmetricTridiagonalMatrix matrix) noexcept
    -> std::expected<EigenDecomposition, SymTriEigendecompError>
{
    const auto diagonal = matrix.diagonal;
    const auto off_diagonal = matrix.off_diagonal;

    if (diagonal.empty())
    {
        return std::unexpected{SymTriEigendecompError::empty_diagonal};
    }
    if (off_diagonal.size() + 1 != diagonal.size())
    {
        return std::unexpected{SymTriEigendecompError::wrong_off_diagonal_size};
    }

    try
    {
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
            return std::unexpected{SymTriEigendecompError::lapack_illegal_value};
        }
        if (info > 0)
        {
            return std::unexpected{SymTriEigendecompError::lapack_failed_to_converge};
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
    catch (const std::bad_alloc&)
    {
        return std::unexpected{SymTriEigendecompError::internal_error};
    }
    catch (...)
    {
        return std::unexpected{SymTriEigendecompError::internal_error};
    }
}

auto svd(const NDArray& array) noexcept -> std::expected<SVDResult, SVDError>
{
    if (array.validity() != NDArrayValidity::valid)
    {
        return std::unexpected{SVDError::invalid_array};
    }
    if (!array.is_matrix())
    {
        return std::unexpected{SVDError::not_matrix};
    }

    try
    {
        const auto rows = array.shape(0);
        const auto cols = array.shape(1);
        const auto rank = std::min(rows, cols);

        auto a_column_major = std::vector<f64>(array.size());
        for (auto row = 0zu; row < rows; ++row)
        {
            for (auto col = 0zu; col < cols; ++col)
            {
                a_column_major[row + col * rows] = array(row, col);
            }
        }

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
            return std::unexpected{SVDError::lapack_illegal_value};
        }
        if (info > 0)
        {
            return std::unexpected{SVDError::lapack_failed_to_converge};
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
            return std::unexpected{SVDError::lapack_illegal_value};
        }
        if (info > 0)
        {
            return std::unexpected{SVDError::lapack_failed_to_converge};
        }

        auto u = NDArray({rows, rank});
        for (auto row = 0zu; row < rows; ++row)
        {
            for (auto col = 0zu; col < rank; ++col)
            {
                u(row, col) = u_column_major[row + col * rows];
            }
        }

        auto s = NDArray({rank});
        std::ranges::copy(singular_values, s.data());

        auto vt = NDArray({rank, cols});
        for (auto row = 0zu; row < rank; ++row)
        {
            for (auto col = 0zu; col < cols; ++col)
            {
                vt(row, col) = vt_column_major[row + col * rank];
            }
        }

        return SVDResult{
            .u = std::move(u),
            .s = std::move(s),
            .vt = std::move(vt),
        };
    }
    catch (const std::bad_alloc&)
    {
        return std::unexpected{SVDError::internal_error};
    }
    catch (...)
    {
        return std::unexpected{SVDError::internal_error};
    }
}

}  // namespace ds_tn
