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

struct SVDResult
{
    NDArray u{};
    NDArray s{};
    NDArray vt{};
};

auto sym_tri_eigendecomp(SymmetricTridiagonalMatrix matrix) -> EigenDecomposition;
auto svd(const NDArray& array) -> SVDResult;

}  // namespace ds_tn
