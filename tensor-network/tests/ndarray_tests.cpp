#include "ndarray/compare.hpp"
#include "ndarray/ndarray.hpp"

#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ds_tn
{

TEST_CASE("NDArray supports rank-0 scalars and rank queries", "[ndarray]")
{
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

TEST_CASE("NDArray default construction yields a zero scalar", "[ndarray]")
{
    const auto array = NDArray{};

    REQUIRE(array.validity() == NDArrayValidity::valid);
    REQUIRE(array.rank() == 0zu);
    REQUIRE(array.size() == 1zu);
    REQUIRE(array.shape().empty());
    REQUIRE(array() == Catch::Approx(0.0));
}

TEST_CASE("NDArray zeros_like copies shape and zero-initializes values", "[ndarray]")
{
    const auto scalar = NDArray::scalar(42.0).zeros_like();

    REQUIRE(scalar.validity() == NDArrayValidity::valid);
    REQUIRE(scalar.is_scalar());
    REQUIRE(scalar() == Catch::Approx(0.0));

    auto source = NDArray::rank3({
        {
            {0.0, 1.0},
            {2.0, 3.0},
            {4.0, 5.0},
        },
        {
            {6.0, 7.0},
            {8.0, 9.0},
            {10.0, 11.0},
        },
    });
    source(1, 2, 1) = 99.0;

    const auto zeros = source.zeros_like();
    const auto expected_shape = std::array<usize, 3>{2, 3, 2};

    REQUIRE(zeros.validity() == NDArrayValidity::valid);
    REQUIRE(zeros.is_tensor3());
    REQUIRE(source.same_shape(zeros));
    REQUIRE(NDArray::same_shape(source, zeros));
    REQUIRE(std::ranges::equal(zeros.shape(), std::span<const usize>{expected_shape}));
    REQUIRE(zeros.size() == source.size());
    REQUIRE(NDArray::same_shape(NDArray::zeros_like(source), zeros));

    for (usize index = 0; index < zeros.size(); ++index)
    {
        REQUIRE(zeros.data(index) == Catch::Approx(0.0));
    }
}

TEST_CASE("NDArray reshape supports static and member call styles", "[ndarray]")
{
    const auto array = NDArray::matrix({
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
    });

    const auto reshaped_static = NDArray::reshape(array, {3, 2});
    const auto reshaped_member = array.reshape({1, 6});
    const auto scalar = NDArray::scalar(42.0);
    const auto reshaped_scalar = scalar.reshape({});

    REQUIRE(reshaped_static.has_value());
    REQUIRE(reshaped_member.has_value());
    REQUIRE(reshaped_scalar.has_value());

    REQUIRE(reshaped_static->same_shape(NDArray({3, 2})));
    REQUIRE(reshaped_static->data(0) == Catch::Approx(1.0));
    REQUIRE(reshaped_static->data(5) == Catch::Approx(6.0));

    REQUIRE(reshaped_member->same_shape(NDArray({1, 6})));
    REQUIRE(reshaped_member->data(0) == Catch::Approx(1.0));
    REQUIRE(reshaped_member->data(5) == Catch::Approx(6.0));

    REQUIRE(reshaped_scalar->is_scalar());
    REQUIRE((*reshaped_scalar)() == Catch::Approx(42.0));
}

TEST_CASE("NDArray reshape validates size", "[ndarray]")
{
    const auto matrix = NDArray::matrix({
        {1.0, 2.0},
        {3.0, 4.0},
    });

    const auto wrong_total = matrix.reshape({3, 3});

    REQUIRE(!wrong_total.has_value());
    REQUIRE(wrong_total.error() == ReshapeError::wrong_total);
}

TEST_CASE("NDArray diag expands a vector into a dense diagonal matrix", "[ndarray]")
{
    const auto vector = NDArray::vector(1.5, -2.0, 4.25);

    const auto diagonal_static = NDArray::diag(vector);
    const auto diagonal_member = vector.diag();

    REQUIRE(diagonal_static.same_shape(NDArray({3, 3})));
    REQUIRE(diagonal_member.same_shape(NDArray({3, 3})));
    REQUIRE(diagonal_static(0, 0) == Catch::Approx(1.5));
    REQUIRE(diagonal_static(1, 1) == Catch::Approx(-2.0));
    REQUIRE(diagonal_static(2, 2) == Catch::Approx(4.25));
    REQUIRE(diagonal_static(0, 1) == Catch::Approx(0.0));
    REQUIRE(diagonal_static(2, 1) == Catch::Approx(0.0));
    REQUIRE(close_per_element(diagonal_static, diagonal_member, 0.0));

    REQUIRE_THROWS_AS(NDArray::diag(NDArray::scalar(1.0)), std::invalid_argument);
    REQUIRE_THROWS_AS(NDArray::diag(NDArray::matrix({{1.0, 2.0}, {3.0, 4.0}})), std::invalid_argument);
}

TEST_CASE("NDArray indexing uses row-major strides and supports negative indices", "[ndarray]")
{
    auto array = NDArray({2, 3, 4});
    array(1, 1, 1) = 7.0;
    array(std::array<usize, 3>{0, 2, 3}) = 9.0;
    array.data(5) = -2.5;

    REQUIRE(array(1, 1, 1) == Catch::Approx(7.0));
    REQUIRE(array(-1, -2, -3) == Catch::Approx(7.0));
    REQUIRE(array(std::array<usize, 3>{0, 2, 3}) == Catch::Approx(9.0));
    REQUIRE(array.data(5) == Catch::Approx(-2.5));
    REQUIRE(array.indices_from_linear(17) == std::vector<usize>{1, 1, 1});
    REQUIRE(array.indices_from_linear(11) == std::vector<usize>{0, 2, 3});

    REQUIRE_THROWS_AS(array(0, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(array(0, 3, 0), std::out_of_range);
    REQUIRE_THROWS_AS(array(-3, 0, 0), std::out_of_range);
}

TEST_CASE("NDArray same_shape distinguishes identical and different shapes", "[ndarray]")
{
    const auto lhs = NDArray({2, 3, 2});
    const auto rhs = NDArray::rank3({
        {
            {0.0, 1.0},
            {2.0, 3.0},
            {4.0, 5.0},
        },
        {
            {6.0, 7.0},
            {8.0, 9.0},
            {10.0, 11.0},
        },
    });
    const auto different = NDArray({2, 2, 3});

    REQUIRE(lhs.same_shape(rhs));
    REQUIRE(NDArray::same_shape(lhs, rhs));
    REQUIRE_FALSE(lhs.same_shape(different));
    REQUIRE_FALSE(NDArray::same_shape(lhs, different));
}

TEST_CASE("NDArray factories build vectors, matrices, and rank-3 tensors", "[ndarray]")
{
    const auto iota = NDArray::iota(5);
    const auto vector = NDArray::vector(1.0, 2.0, 3.0);
    const auto matrix = NDArray::matrix({
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
    });
    const auto tensor3 = NDArray::rank3({
        {
            {0.0, 1.0},
            {2.0, 3.0},
            {4.0, 5.0},
        },
        {
            {6.0, 7.0},
            {8.0, 9.0},
            {10.0, 11.0},
        },
    });
    const auto empty_iota_shape = std::array<usize, 1>{0};
    const auto expected_vector_shape = std::array<usize, 1>{3};
    const auto expected_matrix_shape = std::array<usize, 2>{2, 3};
    const auto expected_tensor3_shape = std::array<usize, 3>{2, 3, 2};

    REQUIRE(iota.validity() == NDArrayValidity::valid);
    REQUIRE(iota.is_vector());
    REQUIRE(iota.shape(0) == 5zu);
    REQUIRE(iota(0) == Catch::Approx(1.0));
    REQUIRE(iota(4) == Catch::Approx(5.0));
    REQUIRE(close_per_element(iota, NDArray::vector(1.0, 2.0, 3.0, 4.0, 5.0), 0.0));

    const auto empty_iota = NDArray::iota(0);
    REQUIRE(empty_iota.validity() == NDArrayValidity::valid);
    REQUIRE(empty_iota.is_vector());
    REQUIRE(std::ranges::equal(empty_iota.shape(), std::span<const usize>{empty_iota_shape}));
    REQUIRE(empty_iota.size() == 0zu);

    REQUIRE(vector.validity() == NDArrayValidity::valid);
    REQUIRE(vector.is_vector());
    REQUIRE(std::ranges::equal(vector.shape(), std::span<const usize>{expected_vector_shape}));
    REQUIRE(vector.shape(0) == 3zu);
    REQUIRE(vector(-1) == Catch::Approx(3.0));

    REQUIRE(matrix.validity() == NDArrayValidity::valid);
    REQUIRE(matrix.is_matrix());
    REQUIRE(std::ranges::equal(matrix.shape(), std::span<const usize>{expected_matrix_shape}));
    REQUIRE(matrix.shape(0) == 2zu);
    REQUIRE(matrix.shape(1) == 3zu);
    REQUIRE(matrix(-1, -1) == Catch::Approx(6.0));
    REQUIRE_THROWS_AS(matrix.shape(2), std::out_of_range);

    REQUIRE(tensor3.validity() == NDArrayValidity::valid);
    REQUIRE(tensor3.is_tensor3());
    REQUIRE(std::ranges::equal(tensor3.shape(), std::span<const usize>{expected_tensor3_shape}));
    REQUIRE(tensor3.shape(0) == 2zu);
    REQUIRE(tensor3.shape(1) == 3zu);
    REQUIRE(tensor3.shape(2) == 2zu);
    REQUIRE(tensor3(1, 2, 1) == Catch::Approx(11.0));
    REQUIRE(tensor3(-1, -1, -1) == Catch::Approx(11.0));
    REQUIRE_THROWS_AS(tensor3.shape(3), std::out_of_range);

    REQUIRE_THROWS_AS(
        NDArray::matrix({
            {1.0, 2.0},
            {3.0},
        }),
        std::invalid_argument
    );

    REQUIRE_THROWS_AS(
        NDArray::rank3({
            {
                {1.0, 2.0},
                {3.0, 4.0},
            },
            {
                {5.0, 6.0},
            },
        }),
        std::invalid_argument
    );

    REQUIRE_THROWS_AS(
        NDArray::rank3({
            {
                {1.0, 2.0},
                {3.0},
            },
        }),
        std::invalid_argument
    );
}

TEST_CASE("NDArray scalar operations and normalization behave as expected", "[ndarray]")
{
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

TEST_CASE("NDArray supports array addition and subtraction operators", "[ndarray]")
{
    auto lhs = NDArray::vector(1.0, 2.0, 3.0);
    const auto rhs = NDArray::vector(0.5, -1.0, 4.0);

    lhs += rhs;
    REQUIRE(lhs(0) == Catch::Approx(1.5));
    REQUIRE(lhs(1) == Catch::Approx(1.0));
    REQUIRE(lhs(2) == Catch::Approx(7.0));

    lhs -= rhs;
    REQUIRE(lhs(0) == Catch::Approx(1.0));
    REQUIRE(lhs(1) == Catch::Approx(2.0));
    REQUIRE(lhs(2) == Catch::Approx(3.0));

    const auto sum = lhs + rhs;
    REQUIRE(sum(0) == Catch::Approx(1.5));
    REQUIRE(sum(1) == Catch::Approx(1.0));
    REQUIRE(sum(2) == Catch::Approx(7.0));

    const auto matrix = NDArray::matrix({
        {1.0, 2.0},
        {3.0, 4.0},
    });

    REQUIRE_THROWS_AS(lhs += matrix, std::invalid_argument);
    REQUIRE_THROWS_AS(lhs -= matrix, std::invalid_argument);
    REQUIRE_THROWS_AS(static_cast<void>(lhs + matrix), std::invalid_argument);
}

TEST_CASE("NDArray printing covers rank 0, 1, 2, and 3", "[ndarray]")
{
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
                    })
        .print(4, true, matrix);
    REQUIRE(
        matrix.str()
        == "NDArray(rank=2, shape=[2 x 2])\n[ 40.0000 "
           "-15.3000]\n[-21.0000  21.8900]\n"
    );

    auto rank3 = NDArray({2, 2, 2});
    for (usize index = 0; index < rank3.size(); ++index)
    {
        rank3.data()[index] = static_cast<f64>(index + 1);
    }

    auto tensor3 = std::ostringstream{};
    rank3.print(1, true, tensor3);
    REQUIRE(
        tensor3.str()
        == "NDArray(rank=3, shape=[2 x 2 x 2])\n"
           "slice 0\n"
           "[1.0 2.0]\n"
           "[3.0 4.0]\n"
           "\n"
           "slice 1\n"
           "[5.0 6.0]\n"
           "[7.0 8.0]\n"
    );
}

}  // namespace ds_tn
