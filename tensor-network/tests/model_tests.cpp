#include "models/transverse_ising.hpp"

#include "ndarray/compare.hpp"

#include <array>
#include <catch2/catch_test_macros.hpp>

namespace ds_tn
{
namespace
{

[[nodiscard]] auto mpo_block(const Tensor& tensor, usize left_bond, usize right_bond) -> NDArray
{
    auto out = NDArray({tensor.shape(1), tensor.shape(2)});
    for (auto physical_out = 0zu; physical_out < tensor.shape(1); ++physical_out)
    {
        for (auto physical_in = 0zu; physical_in < tensor.shape(2); ++physical_in)
        {
            out(physical_out, physical_in) = tensor(left_bond, physical_out, physical_in, right_bond);
        }
    }
    return out;
}

}  // namespace

TEST_CASE("transverse_ising_mpo builds the expected bulk MPO tensors", "[models]")
{
    const auto mpo = transverse_ising_mpo(4, 2.0, 3.0);

    const auto I = NDArray::matrix({
        {1.0, 0.0},
        {0.0, 1.0},
    });
    const auto Z = NDArray::matrix({
        {1.0, 0.0},
        {0.0, -1.0},
    });
    const auto minus_h_X = NDArray::matrix({
        {0.0, -3.0},
        {-3.0, 0.0},
    });
    const auto minus_J_Z = NDArray::matrix({
        {-2.0, 0.0},
        {0.0, 2.0},
    });
    const auto zero = NDArray({2, 2});

    REQUIRE(mpo.size() == 4zu);
    REQUIRE(std::ranges::equal(mpo[0].shape(), std::array<usize, 4>{1, 2, 2, 3}));
    REQUIRE(std::ranges::equal(mpo[1].shape(), std::array<usize, 4>{3, 2, 2, 3}));
    REQUIRE(std::ranges::equal(mpo[3].shape(), std::array<usize, 4>{3, 2, 2, 1}));

    REQUIRE(mpo[0].leg_name(0) == "edge_left");
    REQUIRE(mpo[0].leg_name(1) == "physical_out_0");
    REQUIRE(mpo[0].leg_name(2) == "physical_in_0");
    REQUIRE(mpo[0].leg_name(3) == "bond_01");
    REQUIRE(mpo[1].leg_name(0) == "bond_01");
    REQUIRE(mpo[1].leg_name(3) == "bond_12");
    REQUIRE(mpo[3].leg_name(0) == "bond_23");
    REQUIRE(mpo[3].leg_name(3) == "edge_right");

    REQUIRE(close_per_element(mpo_block(mpo[0], 0, 0), minus_h_X, 0.0));
    REQUIRE(close_per_element(mpo_block(mpo[0], 0, 1), minus_J_Z, 0.0));
    REQUIRE(close_per_element(mpo_block(mpo[0], 0, 2), I, 0.0));

    REQUIRE(close_per_element(mpo_block(mpo[1], 0, 0), I, 0.0));
    REQUIRE(close_per_element(mpo_block(mpo[1], 0, 1), zero, 0.0));
    REQUIRE(close_per_element(mpo_block(mpo[1], 1, 0), Z, 0.0));
    REQUIRE(close_per_element(mpo_block(mpo[1], 2, 0), minus_h_X, 0.0));
    REQUIRE(close_per_element(mpo_block(mpo[1], 2, 1), minus_J_Z, 0.0));
    REQUIRE(close_per_element(mpo_block(mpo[1], 2, 2), I, 0.0));

    REQUIRE(close_per_element(mpo_block(mpo[3], 0, 0), I, 0.0));
    REQUIRE(close_per_element(mpo_block(mpo[3], 1, 0), Z, 0.0));
    REQUIRE(close_per_element(mpo_block(mpo[3], 2, 0), minus_h_X, 0.0));
}

TEST_CASE("transverse_ising_mpo handles the single-site case", "[models]")
{
    const auto mpo = transverse_ising_mpo(1, 2.0, 3.0);
    const auto minus_h_X = NDArray::matrix({
        {0.0, -3.0},
        {-3.0, 0.0},
    });

    REQUIRE(mpo.size() == 1zu);
    REQUIRE(std::ranges::equal(mpo[0].shape(), std::array<usize, 4>{1, 2, 2, 1}));
    REQUIRE(mpo[0].leg_name(0) == "edge_left");
    REQUIRE(mpo[0].leg_name(3) == "edge_right");
    REQUIRE(close_per_element(mpo_block(mpo[0], 0, 0), minus_h_X, 0.0));
}

TEST_CASE("transverse_ising_mpo validates its inputs", "[models]")
{
    REQUIRE_THROWS_AS(transverse_ising_mpo(0, 1.0, 1.0), std::invalid_argument);
}

}  // namespace ds_tn
