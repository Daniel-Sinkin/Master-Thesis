// tests/ndarray_lapack_tests.cpp
#include <algorithm>

#include "ndarray/blas.hpp"
#include "ndarray/compare.hpp"
#include "ndarray/lapack.hpp"
#include "ndarray/ndarray.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

namespace ds_tn
{
namespace
{

[[nodiscard]] auto diagonal_matrix(std::span<const f64> diagonal, usize rows, usize cols) -> NDArray
{
    auto out = NDArray({rows, cols});
    const auto n = std::min({diagonal.size(), rows, cols});
    for (auto i = 0zu; i < n; ++i)
    {
        out(i, i) = diagonal[i];
    }
    return out;
}

}  // namespace

TEST_CASE(
    "Symmetric tridiagonal eigendecomposition returns ascending eigenpairs", "[ndarray][lapack]"
)
{
    const auto diagonal = std::array<f64, 2>{2.0, 3.0};
    const auto off_diagonal = std::array<f64, 1>{1.0};

    const auto eig = sym_tri_eigendecomp({.diagonal = diagonal, .off_diagonal = off_diagonal});

    REQUIRE(eig.eigenvalues.is_vector());
    REQUIRE(eig.eigenvectors.is_matrix());
    REQUIRE(eig.eigenvalues(0) == Catch::Approx(1.381966011250105));
    REQUIRE(eig.eigenvalues(1) == Catch::Approx(3.618033988749895));

    const auto matrix = NDArray::matrix({
        {2.0, 1.0},
        {1.0, 3.0},
    });

    for (auto col = 0zu; col < 2; ++col)
    {
        auto eigenvector = NDArray({2});
        for (auto row = 0zu; row < 2; ++row)
        {
            eigenvector(row) = eig.eigenvectors(row, col);
        }

        auto scaled = eigenvector;
        scaled *= eig.eigenvalues(col);

        REQUIRE(close_accumulated(matrix_vector_product(matrix, eigenvector), scaled, 1.e-10));
        REQUIRE(eigenvector.l2_norm() == Catch::Approx(1.0));
    }
}

TEST_CASE("Symmetric tridiagonal eigendecomposition validates input sizes", "[ndarray][lapack]")
{
    const auto diagonal = std::array<f64, 2>{1.0, 2.0};

    REQUIRE_THROWS_AS(
        sym_tri_eigendecomp({.diagonal = {}, .off_diagonal = {}}), std::invalid_argument
    );
    REQUIRE_THROWS_AS(
        sym_tri_eigendecomp({.diagonal = diagonal, .off_diagonal = {}}), std::invalid_argument
    );
}

TEST_CASE("SVD returns thin factors that reconstruct a rectangular matrix", "[ndarray][lapack]")
{
    const auto matrix = NDArray::matrix({
        {3.0, 1.0},
        {1.0, 3.0},
        {0.0, 2.0},
    });

    const auto result = svd(matrix);

    REQUIRE(result.u.same_shape(NDArray({3, 2})));
    REQUIRE(result.s.same_shape(NDArray({2})));
    REQUIRE(result.vt.same_shape(NDArray({2, 2})));
    REQUIRE(result.s(0) >= result.s(1));

    const auto sigma = diagonal_matrix(
        std::span<const f64>{result.s.data(), result.s.shape(0)},
        result.u.shape(1),
        result.vt.shape(0)
    );
    const auto reconstructed = matrix_matrix_product(matrix_matrix_product(result.u, sigma), result.vt);

    REQUIRE(close_accumulated(reconstructed, matrix, 1.e-10));

    const auto utu = gram_matrix(result.u);
    const auto identity = NDArray::matrix({
        {1.0, 0.0},
        {0.0, 1.0},
    });
    REQUIRE(close_accumulated(utu, identity, 1.e-10));
}

TEST_CASE("SVD returns thin factors for wide matrices", "[ndarray][lapack]")
{
    const auto matrix = NDArray::matrix({
        {1.0, 0.0, 2.0, 0.0},
        {0.0, 3.0, 0.0, 4.0},
    });

    const auto result = svd(matrix);

    REQUIRE(result.u.same_shape(NDArray({2, 2})));
    REQUIRE(result.s.same_shape(NDArray({2})));
    REQUIRE(result.vt.same_shape(NDArray({2, 4})));

    const auto sigma = diagonal_matrix(
        std::span<const f64>{result.s.data(), result.s.shape(0)},
        result.u.shape(1),
        result.vt.shape(0)
    );
    const auto reconstructed = matrix_matrix_product(matrix_matrix_product(result.u, sigma), result.vt);
    REQUIRE(close_accumulated(reconstructed, matrix, 1.e-10));
}

TEST_CASE("SVD validates input arrays", "[ndarray][lapack]")
{
    const auto vector = NDArray::vector({1.0, 2.0, 3.0});
    const auto bad = NDArray{};

    REQUIRE_THROWS_AS(svd(vector), std::invalid_argument);
    REQUIRE_THROWS_AS(svd(bad), std::invalid_argument);
}

}  // namespace ds_tn
