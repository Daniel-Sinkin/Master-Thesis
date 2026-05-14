#include "models/heisenberg.hpp"
#include "tensor/compare.hpp"

#include <array>
#include <catch2/catch_test_macros.hpp>

namespace ds_tn
{

TEST_CASE("heisenberg_two_site_operator stores XX + YY + ZZ as a rank-4 tensor", "[models]")
{
    const auto op = heisenberg_two_site_operator();

    REQUIRE(std::ranges::equal(op.shape(), std::array<usize, 4>{2, 2, 2, 2}));
    REQUIRE(op.leg_name(0) == "site_a_out");
    REQUIRE(op.leg_name(1) == "site_b_out");
    REQUIRE(op.leg_name(2) == "site_a_in");
    REQUIRE(op.leg_name(3) == "site_b_in");

    REQUIRE(op(0, 0, 0, 0) == 1.0);
    REQUIRE(op(0, 1, 0, 1) == -1.0);
    REQUIRE(op(0, 1, 1, 0) == 2.0);
    REQUIRE(op(1, 0, 0, 1) == 2.0);
    REQUIRE(op(1, 0, 1, 0) == -1.0);
    REQUIRE(op(1, 1, 1, 1) == 1.0);
}

TEST_CASE("dense_heisenberg_bond_operator embeds one local term in the full basis", "[models]")
{
    const auto bond = NearestNeighborBond{
        .first = LatticeSite{.row = 0, .col = 0},
        .second = LatticeSite{.row = 0, .col = 1},
    };

    const auto dense = dense_heisenberg_bond_operator(1, 2, bond);
    const auto expected = Tensor(
        NDArray::matrix({
            {1.0, 0.0, 0.0, 0.0},
            {0.0, -1.0, 2.0, 0.0},
            {0.0, 2.0, -1.0, 0.0},
            {0.0, 0.0, 0.0, 1.0},
        }),
        {"state_out", "state_in"}
    );
    const auto expected_block = Tensor(
        NDArray::matrix({
            {-1.0, 2.0},
            {2.0, -1.0},
        }),
        {"state_out", "state_in"}
    );

    REQUIRE(std::ranges::equal(dense.shape(), std::array<usize, 2>{4, 4}));
    REQUIRE(dense.leg_name(0) == "state_out");
    REQUIRE(dense.leg_name(1) == "state_in");
    REQUIRE(close_per_element(dense, expected, 0.0));
    REQUIRE(close_per_element(dense.slice({{1, 3}, {1, 3}}), expected_block, 0.0));
}

TEST_CASE("dense_heisenberg_hamiltonian sums open-boundary bond operators", "[models]")
{
    const auto bond = NearestNeighborBond{
        .first = LatticeSite{.row = 0, .col = 0},
        .second = LatticeSite{.row = 0, .col = 1},
    };

    REQUIRE(close_per_element(
        dense_heisenberg_hamiltonian(1, 2), dense_heisenberg_bond_operator(1, 2, bond), 0.0
    ));
    REQUIRE(
        std::ranges::equal(dense_heisenberg_hamiltonian(2, 2).shape(), std::array<usize, 2>{16, 16})
    );

    REQUIRE_THROWS_AS(dense_heisenberg_hamiltonian(0, 2), std::invalid_argument);
    REQUIRE_THROWS_AS(
        dense_heisenberg_bond_operator(
            1,
            2,
            {
                .first = LatticeSite{.row = 0, .col = 0},
                .second = LatticeSite{.row = 1, .col = 0},
            }
        ),
        std::invalid_argument
    );
}

TEST_CASE("square_lattice_nearest_neighbor_bonds lists open-boundary bonds", "[models]")
{
    const auto bonds = square_lattice_nearest_neighbor_bonds(2, 2);

    REQUIRE(bonds.size() == 4zu);
    REQUIRE(bonds[0].first.row == 0zu);
    REQUIRE(bonds[0].first.col == 0zu);
    REQUIRE(bonds[0].second.row == 0zu);
    REQUIRE(bonds[0].second.col == 1zu);

    REQUIRE(bonds[1].first.row == 0zu);
    REQUIRE(bonds[1].first.col == 0zu);
    REQUIRE(bonds[1].second.row == 1zu);
    REQUIRE(bonds[1].second.col == 0zu);

    REQUIRE(bonds[3].first.linear_index(2) == 2zu);
    REQUIRE(bonds[3].second.linear_index(2) == 3zu);

    REQUIRE_THROWS_AS(square_lattice_nearest_neighbor_bonds(0, 2), std::invalid_argument);
    REQUIRE_THROWS_AS(square_lattice_nearest_neighbor_bonds(2, 0), std::invalid_argument);
}

TEST_CASE("make_heisenberg_peps_setup creates the minimal learning setup", "[models][peps]")
{
    const auto setup = make_heisenberg_peps_setup({
        .n_rows = 2,
        .n_cols = 2,
        .physical_dim = 2,
        .bond_dim = 2,
        .seed = 11,
    });

    REQUIRE(setup.peps.n_rows() == 2zu);
    REQUIRE(setup.peps.n_cols() == 2zu);
    REQUIRE(setup.peps.physical_dim() == 2zu);
    REQUIRE(setup.peps.bond_dim() == 2zu);
    REQUIRE(setup.peps.size() == 4zu);
    REQUIRE(setup.bonds.size() == 4zu);

    REQUIRE(std::ranges::equal(setup.peps(0, 0).shape(), std::array<usize, 5>{2, 1, 1, 2, 2}));
    REQUIRE(std::ranges::equal(setup.peps(0, 1).shape(), std::array<usize, 5>{1, 1, 2, 2, 2}));
    REQUIRE(std::ranges::equal(setup.peps(1, 0).shape(), std::array<usize, 5>{2, 2, 1, 1, 2}));
    REQUIRE(std::ranges::equal(setup.peps(1, 1).shape(), std::array<usize, 5>{1, 2, 2, 1, 2}));
    REQUIRE(std::ranges::equal(setup.two_site_operator.shape(), std::array<usize, 4>{2, 2, 2, 2}));
}

TEST_CASE("make_heisenberg_peps_setup can request fully padded PEPS tensors", "[models][peps]")
{
    const auto setup = make_heisenberg_peps_setup({
        .n_rows = 2,
        .n_cols = 2,
        .physical_dim = 2,
        .bond_dim = 2,
        .fully_padded = true,
        .seed = 11,
    });

    REQUIRE(setup.peps.fully_padded());
    REQUIRE(std::ranges::equal(setup.peps(0, 0).shape(), std::array<usize, 5>{2, 2, 2, 2, 2}));
    REQUIRE(std::ranges::equal(setup.peps(0, 1).shape(), std::array<usize, 5>{2, 2, 2, 2, 2}));
    REQUIRE(std::ranges::equal(setup.peps(1, 0).shape(), std::array<usize, 5>{2, 2, 2, 2, 2}));
    REQUIRE(std::ranges::equal(setup.peps(1, 1).shape(), std::array<usize, 5>{2, 2, 2, 2, 2}));
}

}  // namespace ds_tn
