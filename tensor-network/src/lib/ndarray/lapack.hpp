// lib/ndarray/lapack.hpp
#pragma once

#include "ndarray/ndarray.hpp"

#include <span>

namespace ds_tn
{

struct SymmetricTridiagonalMatrix
{
    std::span<const f64> diagonal;
    std::span<const f64> off_diagonal;
};

struct EigenDecomposition
{
    NDArray eigenvalues;
    NDArray eigenvectors;
};

[[nodiscard]] auto sym_tri_eigendecomp(SymmetricTridiagonalMatrix matrix) -> EigenDecomposition;

}  // namespace ds_tn
