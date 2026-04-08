// lib/ndarray/lapack.hpp
#pragma once

#include "ndarray/ndarray.hpp"

#include <span>

namespace ds_tn
{

enum class MatrixTransform : u8
{
    identity = 0,
    transpose,
};

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

struct HouseholderQR
{
    NDArray factors{};
    NDArray tau{};
};

struct QRResult
{
    NDArray q{};
    NDArray r{};
};

auto sym_tri_eigendecomp(SymmetricTridiagonalMatrix matrix) -> EigenDecomposition;
auto svd(const NDArray& array) -> SVDResult;
auto householder_qr(
    const NDArray& array, MatrixTransform transform = MatrixTransform::identity
) -> HouseholderQR;
auto extract_upper_triangle(const NDArray& factors) -> NDArray;
auto householder_build_q(const NDArray& factors, const NDArray& tau) -> NDArray;
auto householder_build_q(const HouseholderQR& qr) -> NDArray;
auto qr(const NDArray& array, MatrixTransform transform = MatrixTransform::identity) -> QRResult;

}  // namespace ds_tn
