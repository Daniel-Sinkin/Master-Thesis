#include "ndarray/blas.hpp"
#include "ndarray/compare.hpp"
#include "ndarray/ndarray.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <stdexcept>

namespace ds_tn {

TEST_CASE("axpy supports normal, self-alias, and out-parameter forms", "[ndarray][blas]") {
    auto x = NDArray::vector(1.0, 2.0, 3.0);
    auto y = NDArray::vector(4.0, 5.0, 6.0);
    axpy(y, 2.0, x);
    REQUIRE(close_per_element(y, NDArray::vector(6.0, 9.0, 12.0), 0.0));

    auto self = NDArray::vector(1.0, 2.0, 3.0);
    axpy(self, 2.0, self);
    REQUIRE(close_per_element(self, NDArray::vector(3.0, 6.0, 9.0), 0.0));

    auto out = NDArray({3});
    axpy(NDArray::vector(4.0, 5.0, 6.0), 2.0, NDArray::vector(1.0, 2.0, 3.0), out);
    REQUIRE(close_per_element(out, NDArray::vector(6.0, 9.0, 12.0), 0.0));

    auto out_is_y = NDArray::vector(4.0, 5.0, 6.0);
    axpy(out_is_y, 2.0, NDArray::vector(1.0, 2.0, 3.0), out_is_y);
    REQUIRE(close_per_element(out_is_y, NDArray::vector(6.0, 9.0, 12.0), 0.0));

    auto out_is_x = NDArray::vector(1.0, 2.0, 3.0);
    const auto y_copy = NDArray::vector(4.0, 5.0, 6.0);
    axpy(y_copy, 2.0, out_is_x, out_is_x);
    REQUIRE(close_per_element(out_is_x, NDArray::vector(6.0, 9.0, 12.0), 0.0));
}

TEST_CASE("matrix-vector product supports return and out overloads and rejects aliasing", "[ndarray][blas]") {
    const auto matrix = NDArray::matrix({
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
    });
    const auto vector = NDArray::vector(7.0, 8.0, 9.0);
    const auto expected = NDArray::vector(50.0, 122.0);

    REQUIRE(close_per_element(matrix_vector_product(matrix, vector), expected, 1e-12));

    auto out = NDArray({2});
    matrix_vector_product(matrix, vector, out);
    REQUIRE(close_per_element(out, expected, 1e-12));

    const auto square = NDArray::matrix({
        {1.0, 2.0},
        {3.0, 4.0},
    });
    auto aliased = NDArray::vector(5.0, 6.0);
    REQUIRE_THROWS_AS(matrix_vector_product(square, aliased, aliased), std::runtime_error);
}

TEST_CASE("matrix-matrix product supports return and out overloads and rejects aliasing", "[ndarray][blas]") {
    const auto lhs = NDArray::matrix({
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
    });
    const auto rhs = NDArray::matrix({
        {7.0, 8.0},
        {9.0, 10.0},
        {11.0, 12.0},
    });
    const auto expected = NDArray::matrix({
        {58.0, 64.0},
        {139.0, 154.0},
    });

    REQUIRE(close_per_element(matrix_matrix_product(lhs, rhs), expected, 1e-12));

    auto out = NDArray({2, 2});
    matrix_matrix_product(lhs, rhs, out);
    REQUIRE(close_per_element(out, expected, 1e-12));

    auto aliased = NDArray::matrix({
        {1.0, 2.0},
        {3.0, 4.0},
    });
    const auto square_rhs = NDArray::matrix({
        {5.0, 6.0},
        {7.0, 8.0},
    });
    REQUIRE_THROWS_AS(matrix_matrix_product(aliased, square_rhs, aliased), std::runtime_error);
}

TEST_CASE("dot_product flattens matching NDArrays", "[ndarray][blas]") {
    const auto lhs = NDArray::matrix({
        {1.0, 2.0},
        {3.0, 4.0},
    });
    const auto rhs = NDArray::matrix({
        {5.0, 6.0},
        {7.0, 8.0},
    });

    REQUIRE(dot_product(lhs, rhs) == Catch::Approx(70.0));
    REQUIRE_THROWS_AS(dot_product(lhs, NDArray::vector(1.0, 2.0, 3.0, 4.0)), std::runtime_error);
}

} // namespace ds_tn
