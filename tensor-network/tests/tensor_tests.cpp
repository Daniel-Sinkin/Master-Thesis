#include "ndarray/compare.hpp"
#include "tensor/compare.hpp"
#include "tensor/tensor.hpp"

#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ds_tn
{

TEST_CASE("Tensor wraps NDArray data and auto-generates unique readable leg names", "[tensor]")
{
    const auto first = Tensor(NDArray::vector(1.0, 2.0, 3.0));
    const auto second = Tensor(NDArray::vector(4.0, 5.0, 6.0));

    REQUIRE(first.validity() == TensorValidity::valid);
    REQUIRE(first.is_vector());
    REQUIRE(first.leg_names().size() == 1zu);
    REQUIRE(not first.leg_name(0).empty());
    REQUIRE(first.leg_name(0) != second.leg_name(0));
}

TEST_CASE("Tensor default construction yields a zero scalar tensor", "[tensor]")
{
    const auto tensor = Tensor{};

    REQUIRE(tensor.validity() == TensorValidity::valid);
    REQUIRE(tensor.rank() == 0zu);
    REQUIRE(tensor.size() == 1zu);
    REQUIRE(tensor.shape().empty());
    REQUIRE(tensor.leg_names().empty());
    REQUIRE(tensor() == Catch::Approx(0.0));
}

TEST_CASE("Tensor accepts explicit leg names and delegates indexing to its NDArray", "[tensor]")
{
    auto tensor = Tensor(
        NDArray::matrix({
            {1.0, 2.0},
            {3.0, 4.0},
        }),
        {"row", "col"}
    );

    const auto expected_shape = std::array<usize, 2>{2, 2};
    const auto expected_leg_names = std::array<std::string, 2>{"row", "col"};

    REQUIRE(tensor.rank() == 2zu);
    REQUIRE(std::ranges::equal(tensor.shape(), std::span<const usize>{expected_shape}));
    REQUIRE(
        std::ranges::equal(tensor.leg_names(), std::span<const std::string>{expected_leg_names})
    );
    REQUIRE(tensor(1, 0) == Catch::Approx(3.0));

    tensor.array().multiply_scalar(2.0);
    REQUIRE(tensor(1, 0) == Catch::Approx(6.0));
    REQUIRE(tensor.indices_from_linear(3) == std::vector<usize>{1, 1});
}

TEST_CASE("Tensor constructors reject bad leg metadata", "[tensor]")
{
    const auto matrix = NDArray::matrix({
        {1.0, 2.0},
        {3.0, 4.0},
    });

    REQUIRE_THROWS_AS(Tensor(matrix, {"only_one"}), std::invalid_argument);
    REQUIRE_THROWS_AS(Tensor(matrix, {"dup", "dup"}), std::invalid_argument);
    REQUIRE_THROWS_AS(Tensor(matrix, {"", "ok"}), std::invalid_argument);
}

TEST_CASE(
    "Tensor factory helpers preserve the NDArray payload and attach "
    "default legs",
    "[tensor]"
)
{
    const auto scalar = Tensor::scalar(5.0);
    const auto iota = Tensor::iota(4);
    const auto vector = Tensor::vector(1.0, 2.0, 3.0);
    const auto matrix = Tensor::matrix({
        {1.0, 2.0},
        {3.0, 4.0},
    });
    const auto rank3 = Tensor::rank3({
        {
            {0.0, 1.0},
            {2.0, 3.0},
        },
        {
            {4.0, 5.0},
            {6.0, 7.0},
        },
    });

    REQUIRE(scalar() == Catch::Approx(5.0));
    REQUIRE(iota.is_vector());
    REQUIRE(close_per_element(iota.array(), NDArray::vector(0.0, 1.0, 2.0, 3.0), 0.0));
    REQUIRE(close_per_element(vector.array(), NDArray::vector(1.0, 2.0, 3.0), 0.0));
    REQUIRE(close_per_element(matrix.array(), NDArray::matrix({{1.0, 2.0}, {3.0, 4.0}}), 0.0));
    REQUIRE(close_per_element(
        rank3.array(),
        NDArray::rank3({
            {
                {0.0, 1.0},
                {2.0, 3.0},
            },
            {
                {4.0, 5.0},
                {6.0, 7.0},
            },
        }),
        0.0
    ));
    REQUIRE(rank3.is_tensor3());

    const auto random_a = Tensor::random(
        {2, 3},
        RandomUniformOptions{
            .lower = -1.0,
            .upper = 1.0,
        },
        123
    );
    const auto random_b = Tensor::random(
        {2, 3},
        RandomUniformOptions{
            .lower = -1.0,
            .upper = 1.0,
        },
        123
    );
    const auto random_c = Tensor::random_uniform(
        {2, 3},
        RandomUniformOptions{
            .lower = -1.0,
            .upper = 1.0,
        },
        123
    );
    REQUIRE(close_per_element(random_a.array(), random_b.array(), 0.0));
    REQUIRE(close_per_element(random_a.array(), random_c.array(), 0.0));
    REQUIRE(random_a.leg_name(0) != random_b.leg_name(0));
}

TEST_CASE("Tensor comparison helpers compare payloads and ignore leg labels", "[tensor][compare]")
{
    const auto lhs = Tensor(
        NDArray::matrix({
            {1.0, 2.0},
            {3.0, 4.0},
        }),
        {"left", "right"}
    );
    const auto rhs = Tensor(
        NDArray::matrix({
            {1.0, 2.0 + 5e-7},
            {3.0, 4.0},
        }),
        {"row", "col"}
    );
    const auto different_shape = Tensor::vector(1.0, 2.0, 3.0);

    REQUIRE(close_per_element(lhs, rhs, 1e-6));
    REQUIRE(not close_per_element(lhs, rhs, 1e-8));
    REQUIRE(close_accumulated(lhs, rhs, 1e-6));
    REQUIRE(not close_accumulated(lhs, rhs, 1e-8));
    REQUIRE(not close_per_element(lhs, different_shape, 1e-6));
}

TEST_CASE("Tensor diag expands a vector into a dense diagonal tensor", "[tensor]")
{
    const auto vector = Tensor(NDArray::vector(1.5, -2.0, 4.25), {"i"});

    const auto diagonal_static = Tensor::diag(vector);
    const auto diagonal_member = vector.diag();
    const auto expected_legs = std::array<std::string, 2>{"i_row", "i_col"};

    REQUIRE(diagonal_static.is_matrix());
    REQUIRE(diagonal_static(0, 0) == Catch::Approx(1.5));
    REQUIRE(diagonal_static(1, 1) == Catch::Approx(-2.0));
    REQUIRE(diagonal_static(2, 2) == Catch::Approx(4.25));
    REQUIRE(diagonal_static(0, 1) == Catch::Approx(0.0));
    REQUIRE(diagonal_static(2, 1) == Catch::Approx(0.0));
    REQUIRE(
        std::ranges::equal(
            diagonal_static.leg_names(), std::span<const std::string>{expected_legs}
        )
    );
    REQUIRE(close_per_element(diagonal_static, diagonal_member, 0.0));

    REQUIRE_THROWS_AS(Tensor::diag(Tensor{}), std::invalid_argument);
    REQUIRE_THROWS_AS(Tensor::diag(Tensor::matrix({{1.0, 2.0}, {3.0, 4.0}})), std::invalid_argument);
}

TEST_CASE("Tensor zero check respects tolerance", "[tensor][compare]")
{
    const auto zero = Tensor{};
    const auto near_zero = Tensor::vector(1e-9, -5e-10, 2e-9);
    const auto non_zero = Tensor::vector(1e-6, 0.0, 0.0);

    REQUIRE(is_zero(zero));
    REQUIRE(is_zero(near_zero, 1e-8));
    REQUIRE(not is_zero(near_zero, 1e-10));
    REQUIRE(not is_zero(non_zero, 1e-8));
    REQUIRE(not is_zero(non_zero, -1.0));
}

TEST_CASE("Tensor metadata formatting reports shape and leg names", "[tensor][metadata]")
{
    REQUIRE(Tensor{}.format_metadata() == "Tensor(shape=[], legs=[])");

    const auto tensor = Tensor(
        NDArray::matrix({
            {1.0, 2.0},
            {3.0, 4.0},
        }),
        {"row", "col"}
    );

    REQUIRE(
        tensor.format_metadata() == "Tensor(shape=[2 x 2], legs=[row, col])"
    );
}

TEST_CASE("Tensor printing includes leg metadata and the aligned NDArray body", "[tensor]")
{
    const auto tensor = Tensor(
        NDArray::matrix({
            {40.0, -15.3},
            {-21.0, 21.89},
        }),
        {"left", "right"}
    );

    auto output = std::ostringstream{};
    tensor.print(4, true, output);

    REQUIRE(
        output.str()
        == "Tensor(shape=[2 x 2], legs=[left, right])\n"
           "[ 40.0000 -15.3000]\n"
           "[-21.0000  21.8900]\n"
    );
}

TEST_CASE("Tensor print_metadata writes the formatted metadata line", "[tensor][metadata]")
{
    const auto tensor = Tensor(
        NDArray::matrix({
            {1.0, 2.0},
            {3.0, 4.0},
        }),
        {"row", "col"}
    );

    auto output = std::ostringstream{};
    tensor.print_metadata(output);

    REQUIRE(output.str() == "Tensor(shape=[2 x 2], legs=[row, col])\n");
}

}  // namespace ds_tn
