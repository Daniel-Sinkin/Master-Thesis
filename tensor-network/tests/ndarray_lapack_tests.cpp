// tests/ndarray_lapack_tests.cpp
#include "ndarray/blas.hpp"
#include "ndarray/compare.hpp"
#include "ndarray/lapack.hpp"
#include "ndarray/ndarray.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

namespace ds_tn
{

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

}  // namespace ds_tn
