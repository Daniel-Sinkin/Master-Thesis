// lib/ndarray/lapack.hpp
#pragma once

#include "ndarray/ndarray.hpp"

#include <expected>
#include <span>
#include <string_view>

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

enum class SymTriEigendecompError : u8
{
    empty_diagonal,
    wrong_off_diagonal_size,
    lapack_illegal_value,
    lapack_failed_to_converge,
    internal_error,
};

[[nodiscard]] constexpr auto to_string(SymTriEigendecompError error) noexcept -> std::string_view
{
    switch (error)
    {
        case SymTriEigendecompError::empty_diagonal:
            return "empty_diagonal";
        case SymTriEigendecompError::wrong_off_diagonal_size:
            return "wrong_off_diagonal_size";
        case SymTriEigendecompError::lapack_illegal_value:
            return "lapack_illegal_value";
        case SymTriEigendecompError::lapack_failed_to_converge:
            return "lapack_failed_to_converge";
        case SymTriEigendecompError::internal_error:
            return "internal_error";
    }

    return "unknown_sym_tri_eigendecomp_error";
}

struct SVDResult
{
    NDArray u{};
    NDArray s{};
    NDArray vt{};
};

enum class SVDError : u8
{
    invalid_array,
    not_matrix,
    lapack_illegal_value,
    lapack_failed_to_converge,
    internal_error,
};

[[nodiscard]] constexpr auto to_string(SVDError error) noexcept -> std::string_view
{
    switch (error)
    {
        case SVDError::invalid_array:
            return "invalid_array";
        case SVDError::not_matrix:
            return "not_matrix";
        case SVDError::lapack_illegal_value:
            return "lapack_illegal_value";
        case SVDError::lapack_failed_to_converge:
            return "lapack_failed_to_converge";
        case SVDError::internal_error:
            return "internal_error";
    }

    return "unknown_svd_error";
}

auto sym_tri_eigendecomp(SymmetricTridiagonalMatrix matrix) noexcept
    -> std::expected<EigenDecomposition, SymTriEigendecompError>;
auto svd(const NDArray& array) noexcept -> std::expected<SVDResult, SVDError>;

}  // namespace ds_tn
