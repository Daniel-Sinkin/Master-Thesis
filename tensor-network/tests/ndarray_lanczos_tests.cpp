#include "ndarray/blas.hpp"
#include "ndarray/compare.hpp"
#include "ndarray/lanczos.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace ds_tn
{

TEST_CASE("Lanczos validates NDArray preconditions", "[ndarray][lanczos]")
{
    REQUIRE(lanczos(NDArray::vector(1.0, 2.0, 3.0)).error() == LanczosError::not_matrix);

    REQUIRE(
        lanczos(
            NDArray::matrix({
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0},
            })
        )
            .error()
        == LanczosError::matrix_not_square
    );

    REQUIRE(
        lanczos(
            NDArray::matrix({
                {1.0, 2.0},
                {3.0, 4.0},
            }),
            {.check_symmetric = true}
        )
            .error()
        == LanczosError::matrix_not_symmetric
    );

    const auto apply_bad = [](const NDArray&) -> NDArray { return NDArray({2, 2}); };
    REQUIRE(
        lanczos(2, apply_bad, {.num_iterations = 1}).error()
        == LanczosError::invalid_operator_output
    );
    REQUIRE(
        lanczos(0, apply_bad, {.num_iterations = 1}).error() == LanczosError::invalid_dimension
    );
    REQUIRE(
        lanczos(2, [](const NDArray& v) -> NDArray { return v; }, {.num_iterations = 0}).error()
        == LanczosError::invalid_iteration_count
    );
}

TEST_CASE("Lanczos returns a normalized Ritz pair for a symmetric matrix", "[ndarray][lanczos]")
{
    const auto matrix = NDArray::matrix({
        {2.0, 1.0},
        {1.0, 3.0},
    });

    const auto result = lanczos(
        matrix,
        {
            .num_iterations = 4,
            .seed = 7,
            .do_reorthogonalization = true,
            .check_symmetric = true,
        }
    );

    REQUIRE(result.has_value());
    REQUIRE(result->ritz_value == Catch::Approx(1.381966011250105));
    REQUIRE(result->ritz_vector.l2_norm() == Catch::Approx(1.0));

    auto scaled_ritz_vector = result->ritz_vector;
    scaled_ritz_vector *= result->ritz_value;
    REQUIRE(close_accumulated(
        matrix_vector_product(matrix, result->ritz_vector), scaled_ritz_vector, 1e-10
    ));
}

}  // namespace ds_tn
