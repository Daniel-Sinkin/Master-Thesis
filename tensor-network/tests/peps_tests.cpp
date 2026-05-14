#include "tensor/compare.hpp"
#include "tensor/contraction.hpp"
#include "tensor/peps.hpp"

#include <array>
#include <catch2/catch_test_macros.hpp>

namespace ds_tn
{

TEST_CASE("Peps constructor creates the expected grid topology", "[tensor][peps]")
{
    const auto peps = Peps{Peps::Config{
        .n_rows = 3,
        .n_cols = 4,
        .bond_dim = 5,
        .physical_dim = 3,
    }};

    REQUIRE(peps.n_rows() == 3zu);
    REQUIRE(peps.n_cols() == 4zu);
    REQUIRE(peps.bond_dim() == 5zu);
    REQUIRE(peps.physical_dim() == 3zu);
    REQUIRE_FALSE(peps.fully_padded());
    REQUIRE(peps.size() == 12zu);

    REQUIRE(std::ranges::equal(peps(0, 0).shape(), std::array<usize, 5>{5, 1, 1, 5, 3}));
    REQUIRE(std::ranges::equal(peps(0, 1).shape(), std::array<usize, 5>{5, 1, 5, 5, 3}));
    REQUIRE(std::ranges::equal(peps(1, 1).shape(), std::array<usize, 5>{5, 5, 5, 5, 3}));
    REQUIRE(std::ranges::equal(peps(2, 3).shape(), std::array<usize, 5>{1, 5, 5, 1, 3}));

    REQUIRE(peps(0, 0).leg_name(Peps::k_leg_right) == "r0,0");
    REQUIRE(peps(0, 0).leg_name(Peps::k_leg_top) == "t0,0");
    REQUIRE(peps(0, 0).leg_name(Peps::k_leg_left) == "l0,0");
    REQUIRE(peps(0, 0).leg_name(Peps::k_leg_bottom) == "b0,0");
    REQUIRE(peps(0, 0).leg_name(Peps::k_leg_physical) == "p0,0");
    REQUIRE(peps(0, 0).leg_name(Peps::k_leg_right) != peps(0, 1).leg_name(Peps::k_leg_left));

    REQUIRE(peps.total_entries() == 6300zu);
}

TEST_CASE("Peps constructor can keep full virtual padding on boundaries", "[tensor][peps]")
{
    const auto peps = Peps{Peps::Config{
        .n_rows = 3,
        .n_cols = 4,
        .bond_dim = 5,
        .physical_dim = 3,
        .fully_padded = true,
    }};

    REQUIRE(peps.fully_padded());
    REQUIRE(std::ranges::equal(peps(0, 0).shape(), std::array<usize, 5>{5, 5, 5, 5, 3}));
    REQUIRE(std::ranges::equal(peps(0, 1).shape(), std::array<usize, 5>{5, 5, 5, 5, 3}));
    REQUIRE(std::ranges::equal(peps(1, 1).shape(), std::array<usize, 5>{5, 5, 5, 5, 3}));
    REQUIRE(std::ranges::equal(peps(2, 3).shape(), std::array<usize, 5>{5, 5, 5, 5, 3}));
    REQUIRE(peps.total_entries() == 22500zu);
}

TEST_CASE("Peps indexing validates bounds", "[tensor][peps]")
{
    auto peps = Peps{Peps::Config{}};

    REQUIRE_NOTHROW(peps.at(2, 2));
    REQUIRE_THROWS_AS(peps.at(3, 0), std::out_of_range);
    REQUIRE_THROWS_AS(peps.at(0, 3), std::out_of_range);
    REQUIRE_THROWS_AS(peps(4, 4), std::out_of_range);
}

TEST_CASE("random_peps is deterministic when seeded", "[tensor][peps]")
{
    const auto lhs = random_peps(
        3,
        5,
        RandomPepsConfig{
            .physical_dim = 2,
            .bond_dim = 4,
            .seed = 17,
        }
    );
    const auto rhs = random_peps(
        3,
        5,
        RandomPepsConfig{
            .physical_dim = 2,
            .bond_dim = 4,
            .seed = 17,
        }
    );

    REQUIRE(lhs.size() == rhs.size());
    for (auto row = 0zu; row < lhs.n_rows(); ++row)
    {
        for (auto col = 0zu; col < lhs.n_cols(); ++col)
        {
            REQUIRE(close_per_element(lhs(row, col), rhs(row, col), 0.0));
            REQUIRE(std::ranges::equal(lhs(row, col).leg_names(), rhs(row, col).leg_names()));
        }
    }
}

TEST_CASE("random_peps validates inputs and reports unimplemented suppression", "[tensor][peps]")
{
    REQUIRE_THROWS_AS(random_peps(0, 3), std::invalid_argument);
    REQUIRE_THROWS_AS(random_peps(3, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(
        random_peps(3, 3, RandomPepsConfig{.physical_dim = 0, .bond_dim = 2}), std::invalid_argument
    );
    REQUIRE_THROWS_AS(
        random_peps(3, 3, RandomPepsConfig{.physical_dim = 2, .bond_dim = 0}), std::invalid_argument
    );
    REQUIRE_THROWS_AS(
        random_peps(
            3,
            3,
            RandomPepsConfig{
                .apply_algebraic_power_law_suppression = true,
            }
        ),
        std::logic_error
    );
}

TEST_CASE(
    "Contracting copied neighbouring PEPS tensors leaves the original PEPS unchanged",
    "[tensor][peps]"
)
{
    const auto peps = random_peps(
        3,
        5,
        RandomPepsConfig{
            .physical_dim = 2,
            .bond_dim = 3,
            .seed = 23,
        }
    );
    const auto original_left = peps(0, 0);
    const auto original_right = peps(0, 1);

    auto left = original_left;
    auto right = original_right;
    left.rename_leg(left.leg_name(Peps::k_leg_right), right.leg_name(Peps::k_leg_left));
    const auto contracted = contract(left, right);

    REQUIRE(close_per_element(peps(0, 0), original_left, 0.0));
    REQUIRE(close_per_element(peps(0, 1), original_right, 0.0));
    REQUIRE(std::ranges::equal(peps(0, 0).leg_names(), original_left.leg_names()));
    REQUIRE(std::ranges::equal(peps(0, 1).leg_names(), original_right.leg_names()));

    REQUIRE(contracted.validity() == TensorValidity::valid);
    REQUIRE(std::ranges::equal(contracted.shape(), std::array<usize, 8>{1, 1, 3, 2, 3, 1, 3, 2}));
    REQUIRE(
        std::ranges::equal(
            contracted.leg_names(),
            std::array<std::string_view, 8>{
                "t0,0", "l0,0", "b0,0", "p0,0", "r0,1", "t0,1", "b0,1", "p0,1"
            }
        )
    );
}

}  // namespace ds_tn
