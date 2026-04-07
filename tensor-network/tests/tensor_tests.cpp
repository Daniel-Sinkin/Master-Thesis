#include "ndarray/compare.hpp"
#include "tensor/tensor.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ds_tn {

TEST_CASE("Tensor wraps NDArray data and auto-generates unique readable leg names", "[tensor]") {
    const auto first = Tensor(NDArray::vector(1.0, 2.0, 3.0));
    const auto second = Tensor(NDArray::vector(4.0, 5.0, 6.0));

    REQUIRE(first.validity() == TensorValidity::valid);
    REQUIRE(first.is_vector());
    REQUIRE(first.leg_names().size() == 1zu);
    REQUIRE(not first.leg_name(0).empty());
    REQUIRE(first.leg_name(0) != second.leg_name(0));
}

TEST_CASE("Tensor accepts explicit leg names and delegates indexing to its NDArray", "[tensor]") {
    auto tensor = Tensor(
        NDArray::matrix({
            {1.0, 2.0},
            {3.0, 4.0},
        }),
        {"row", "col"});

    const auto expected_shape = std::array<usize, 2>{2, 2};
    const auto expected_leg_names = std::array<std::string, 2>{"row", "col"};

    REQUIRE(tensor.rank() == 2zu);
    REQUIRE(std::ranges::equal(tensor.shape(), std::span<const usize>{expected_shape}));
    REQUIRE(std::ranges::equal(tensor.leg_names(), std::span<const std::string>{expected_leg_names}));
    REQUIRE(tensor(1, 0) == Catch::Approx(3.0));

    tensor.array().multiply_scalar(2.0);
    REQUIRE(tensor(1, 0) == Catch::Approx(6.0));
    REQUIRE(tensor.indices_from_linear(3) == std::vector<usize>{1, 1});
}

TEST_CASE("Tensor constructors reject bad leg metadata", "[tensor]") {
    const auto matrix = NDArray::matrix({
        {1.0, 2.0},
        {3.0, 4.0},
    });

    REQUIRE_THROWS_AS(Tensor(matrix, {"only_one"}), std::invalid_argument);
    REQUIRE_THROWS_AS(Tensor(matrix, {"dup", "dup"}), std::invalid_argument);
    REQUIRE_THROWS_AS(Tensor(matrix, {"", "ok"}), std::invalid_argument);
}

TEST_CASE("Tensor factory helpers preserve the NDArray payload and attach default legs", "[tensor]") {
    const auto scalar = Tensor::scalar(5.0);
    const auto vector = Tensor::vector(1.0, 2.0, 3.0);
    const auto matrix = Tensor::matrix({
        {1.0, 2.0},
        {3.0, 4.0},
    });

    REQUIRE(scalar() == Catch::Approx(5.0));
    REQUIRE(close_per_element(vector.array(), NDArray::vector(1.0, 2.0, 3.0), 0.0));
    REQUIRE(close_per_element(matrix.array(), NDArray::matrix({{1.0, 2.0}, {3.0, 4.0}}), 0.0));

    const auto random_a = Tensor::random_uniform({2, 3}, -1.0, 1.0, 123);
    const auto random_b = Tensor::random_uniform({2, 3}, -1.0, 1.0, 123);
    REQUIRE(close_per_element(random_a.array(), random_b.array(), 0.0));
    REQUIRE(random_a.leg_name(0) != random_b.leg_name(0));
}

TEST_CASE("Tensor printing includes leg metadata and the aligned NDArray body", "[tensor]") {
    const auto tensor = Tensor(
        NDArray::matrix({
            {40.0, -15.3},
            {-21.0, 21.89},
        }),
        {"left", "right"});

    auto output = std::ostringstream{};
    tensor.print(4, true, output);

    REQUIRE(
        output.str() ==
        "Tensor(rank=2, shape=[2 x 2], legs=[left, right])\n"
        "[ 40.0000 -15.3000]\n"
        "[-21.0000  21.8900]\n");
}

} // namespace ds_tn
