// lib/ndarray/lapack.cpp
#include "ndarray/lapack.hpp"

#include <algorithm>
#include <limits>
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

auto sym_tri_eigendecomp(SymmetricTridiagonalMatrix matrix) -> EigenDecomposition
{
    const auto diagonal = matrix.diagonal;
    const auto off_diagonal = matrix.off_diagonal;

    if (diagonal.empty())
    {
        throw std::invalid_argument(
            "symmetric_tridiagonal_eigendecomposition requires a non-empty diagonal."
        );
    }
    if (off_diagonal.size() + 1 != diagonal.size())
    {
        throw std::invalid_argument(
            "symmetric_tridiagonal_eigendecomposition requires off_diagonal.size() + 1 == "
            "diagonal.size()."
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

    if (info != 0)
    {
        throw std::runtime_error(
            "dstev failed with info = " + std::to_string(static_cast<long long>(info)) + '.'
        );
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

}  // namespace ds_tn
