#include "ndarray/ndarray.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ds_tn {

TEST_CASE("NDArray supports rank-0 scalars and rank queries", "[ndarray]") {
    auto scalar = NDArray::scalar(42.0);

    REQUIRE(scalar.validity() == NDArrayValidity::valid);
    REQUIRE(scalar.rank() == 0zu);
    REQUIRE(scalar.size() == 1zu);
    REQUIRE(scalar.shape().empty());
    REQUIRE(scalar.is_scalar());
    REQUIRE(scalar.is_trivial());
    REQUIRE(not scalar.is_vector());

    REQUIRE(scalar() == Catch::Approx(42.0));
    scalar() = -7.5;
    REQUIRE(scalar(std::span<const usize>{}) == Catch::Approx(-7.5));
    REQUIRE(scalar.indices_from_linear(0).empty());
}

TEST_CASE("NDArray indexing uses row-major strides and supports negative indices", "[ndarray]") {
    auto array = NDArray({2, 3, 4});
    array(1, 1, 1) = 7.0;
    array(std::array<usize, 3>{0, 2, 3}) = 9.0;

    REQUIRE(array(1, 1, 1) == Catch::Approx(7.0));
    REQUIRE(array(-1, -2, -3) == Catch::Approx(7.0));
    REQUIRE(array(std::array<usize, 3>{0, 2, 3}) == Catch::Approx(9.0));
    REQUIRE(array.indices_from_linear(17) == std::vector<usize>{1, 1, 1});
    REQUIRE(array.indices_from_linear(11) == std::vector<usize>{0, 2, 3});

    REQUIRE_THROWS_AS(array(0, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(array(0, 3, 0), std::out_of_range);
    REQUIRE_THROWS_AS(array(-3, 0, 0), std::out_of_range);
}

TEST_CASE("NDArray factories build vectors and matrices", "[ndarray]") {
    const auto vector = NDArray::vector(1.0, 2.0, 3.0);
    const auto matrix = NDArray::matrix({
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
    });
    const auto expected_vector_shape = std::array<usize, 1>{3};
    const auto expected_matrix_shape = std::array<usize, 2>{2, 3};

    REQUIRE(vector.validity() == NDArrayValidity::valid);
    REQUIRE(vector.is_vector());
    REQUIRE(std::ranges::equal(vector.shape(), std::span<const usize>{expected_vector_shape}));
    REQUIRE(vector(-1) == Catch::Approx(3.0));

    REQUIRE(matrix.validity() == NDArrayValidity::valid);
    REQUIRE(matrix.is_matrix());
    REQUIRE(std::ranges::equal(matrix.shape(), std::span<const usize>{expected_matrix_shape}));
    REQUIRE(matrix(-1, -1) == Catch::Approx(6.0));

    REQUIRE_THROWS_AS(
        NDArray::matrix({
            {1.0, 2.0},
            {3.0},
        }),
        std::invalid_argument);
}

TEST_CASE("NDArray scalar operations and normalization behave as expected", "[ndarray]") {
    auto values = NDArray::vector(1.0, -2.0, 3.0);

    values.add_scalar(1.0);
    REQUIRE(values(0) == Catch::Approx(2.0));
    REQUIRE(values(1) == Catch::Approx(-1.0));
    REQUIRE(values(2) == Catch::Approx(4.0));

    values.subtract_scalar(2.0);
    REQUIRE(values(0) == Catch::Approx(0.0));
    REQUIRE(values(1) == Catch::Approx(-3.0));
    REQUIRE(values(2) == Catch::Approx(2.0));

    values *= 2.0;
    REQUIRE(values(0) == Catch::Approx(0.0));
    REQUIRE(values(1) == Catch::Approx(-6.0));
    REQUIRE(values(2) == Catch::Approx(4.0));

    values /= 2.0;
    REQUIRE(values(0) == Catch::Approx(0.0));
    REQUIRE(values(1) == Catch::Approx(-3.0));
    REQUIRE(values(2) == Catch::Approx(2.0));

    auto normalized = NDArray::vector(3.0, 4.0).normalized();
    REQUIRE(normalized(0) == Catch::Approx(0.6));
    REQUIRE(normalized(1) == Catch::Approx(0.8));

    auto in_place = NDArray::vector(3.0, 4.0);
    in_place.normalize();
    REQUIRE(in_place(0) == Catch::Approx(0.6));
    REQUIRE(in_place(1) == Catch::Approx(0.8));

    auto zero = NDArray({2});
    REQUIRE_THROWS_AS(zero.normalize(), std::runtime_error);
    REQUIRE_THROWS_AS(values.divide_scalar(0.0), std::invalid_argument);
}

TEST_CASE("NDArray printing covers rank 0, 1, 2, and 3", "[ndarray]") {
    auto scalar = std::ostringstream{};
    NDArray::scalar(42.0).print(4, true, scalar);
    REQUIRE(scalar.str() == "NDArray(rank=0, shape=[])\n42.0000\n");

    auto vector = std::ostringstream{};
    NDArray::vector(1.0, -2.5, 30.0).print(2, true, vector);
    REQUIRE(vector.str() == "NDArray(rank=1, shape=[3])\n[ 1.00 -2.50 30.00]\n");

    auto matrix = std::ostringstream{};
    NDArray::matrix({
        {40.0, -15.3},
        {-21.0, 21.89},
    }).print(4, true, matrix);
    REQUIRE(matrix.str() == "NDArray(rank=2, shape=[2 x 2])\n[ 40.0000 -15.3000]\n[-21.0000  21.8900]\n");

    auto rank3 = NDArray({2, 2, 2});
    for (usize index = 0; index < rank3.size(); ++index) {
        rank3.data()[index] = static_cast<f64>(index + 1);
    }

    auto tensor3 = std::ostringstream{};
    rank3.print(1, true, tensor3);
    REQUIRE(
        tensor3.str() ==
        "NDArray(rank=3, shape=[2 x 2 x 2])\n"
        "slice 0\n"
        "[1.0 2.0]\n"
        "[3.0 4.0]\n"
        "\n"
        "slice 1\n"
        "[5.0 6.0]\n"
        "[7.0 8.0]\n");
}

} // namespace ds_tn
