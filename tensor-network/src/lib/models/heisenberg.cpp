// lib/models/heisenberg.cpp
#include "models/heisenberg.hpp"

#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ds_tn
{
namespace
{

[[nodiscard]] auto checked_site_count(usize n_rows, usize n_cols) -> usize
{
    if (n_rows == 0 or n_cols == 0)
    {
        throw std::invalid_argument("Heisenberg lattice requires n_rows >= 1 and n_cols >= 1.");
    }
    if (n_rows > std::numeric_limits<usize>::max() / n_cols)
    {
        throw std::overflow_error("Heisenberg lattice site count overflows usize.");
    }
    return n_rows * n_cols;
}

[[nodiscard]] auto checked_state_count(usize base, usize length) -> usize
{
    auto out = usize{1};
    for (auto i = 0zu; i < length; ++i)
    {
        if (out > std::numeric_limits<usize>::max() / base)
        {
            throw std::overflow_error("Hilbert-space basis size overflows usize.");
        }
        out *= base;
    }
    return out;
}

auto require_dense_matrix_storage(usize hilbert_dim) -> void
{
    if (hilbert_dim > std::numeric_limits<usize>::max() / hilbert_dim)
    {
        throw std::overflow_error("Dense Hamiltonian storage size overflows usize.");
    }
}

auto require_bond_inside_lattice(usize n_rows, usize n_cols, NearestNeighborBond bond) -> void
{
    const auto site_valid = [=](LatticeSite site)
    { return site.row < n_rows and site.col < n_cols; };
    if (not site_valid(bond.first) or not site_valid(bond.second))
    {
        throw std::invalid_argument("Heisenberg bond endpoint lies outside the lattice.");
    }
    if (bond.first.row == bond.second.row and bond.first.col == bond.second.col)
    {
        throw std::invalid_argument("Heisenberg bond endpoints must be distinct.");
    }
}

[[nodiscard]] auto encode_base(std::span<const usize> digits, usize base) -> usize
{
    auto out = usize{0};
    for (const auto digit : digits)
    {
        if (digit >= base)
        {
            throw std::invalid_argument("encode_base digit exceeds base.");
        }
        out = out * base + digit;
    }
    return out;
}

[[nodiscard]] auto decode_base(usize encoded, usize length, usize base) -> std::vector<usize>
{
    auto out = std::vector<usize>(length);
    for (auto i = length; i > 0; --i)
    {
        out[i - 1zu] = encoded % base;
        encoded /= base;
    }
    return out;
}

}  // namespace

auto heisenberg_two_site_operator(f64 coupling) -> Tensor
{
    auto op = Tensor(
        std::vector<usize>{2, 2, 2, 2},
        {
            "site_a_out",
            "site_b_out",
            "site_a_in",
            "site_b_in",
        }
    );

    // Pauli convention: X_a X_b + Y_a Y_b + Z_a Z_b in the |00>, |01>, |10>, |11> basis.
    op(0, 0, 0, 0) = coupling;
    op(0, 1, 0, 1) = -coupling;
    op(0, 1, 1, 0) = 2.0 * coupling;
    op(1, 0, 0, 1) = 2.0 * coupling;
    op(1, 0, 1, 0) = -coupling;
    op(1, 1, 1, 1) = coupling;

    return op;
}

auto square_lattice_nearest_neighbor_bonds(usize n_rows, usize n_cols)
    -> std::vector<NearestNeighborBond>
{
    if (n_rows == 0 or n_cols == 0)
    {
        throw std::invalid_argument(
            "square_lattice_nearest_neighbor_bonds requires n_rows >= 1 and n_cols >= 1."
        );
    }

    auto bonds = std::vector<NearestNeighborBond>{};
    bonds.reserve(n_rows * (n_cols - 1) + (n_rows - 1) * n_cols);

    for (auto row = 0zu; row < n_rows; ++row)
    {
        for (auto col = 0zu; col < n_cols; ++col)
        {
            if (col + 1 < n_cols)
            {
                bonds.push_back({
                    .first = LatticeSite{.row = row, .col = col},
                    .second = LatticeSite{.row = row, .col = col + 1},
                });
            }
            if (row + 1 < n_rows)
            {
                bonds.push_back({
                    .first = LatticeSite{.row = row, .col = col},
                    .second = LatticeSite{.row = row + 1, .col = col},
                });
            }
        }
    }

    return bonds;
}

auto dense_heisenberg_bond_operator(
    usize n_rows, usize n_cols, NearestNeighborBond bond, f64 coupling
) -> Tensor
{
    const auto site_count = checked_site_count(n_rows, n_cols);
    const auto hilbert_dim = checked_state_count(2zu, site_count);
    require_dense_matrix_storage(hilbert_dim);
    require_bond_inside_lattice(n_rows, n_cols, bond);

    const auto site_a = bond.first.linear_index(n_cols);
    const auto site_b = bond.second.linear_index(n_cols);
    const auto local = heisenberg_two_site_operator(coupling);

    auto out = Tensor(std::vector<usize>{hilbert_dim, hilbert_dim}, {"state_out", "state_in"});
    for (auto state_in = 0zu; state_in < hilbert_dim; ++state_in)
    {
        const auto input_spins = decode_base(state_in, site_count, 2zu);
        const auto spin_a_in = input_spins[site_a];
        const auto spin_b_in = input_spins[site_b];

        for (auto spin_a_out = 0zu; spin_a_out < 2zu; ++spin_a_out)
        {
            for (auto spin_b_out = 0zu; spin_b_out < 2zu; ++spin_b_out)
            {
                const auto value = local(spin_a_out, spin_b_out, spin_a_in, spin_b_in);
                if (value == 0.0)
                {
                    continue;
                }

                auto output_spins = input_spins;
                output_spins[site_a] = spin_a_out;
                output_spins[site_b] = spin_b_out;
                const auto state_out = encode_base(output_spins, 2zu);
                out(state_out, state_in) += value;
            }
        }
    }

    return out;
}

auto dense_heisenberg_hamiltonian(usize n_rows, usize n_cols, f64 coupling) -> Tensor
{
    const auto site_count = checked_site_count(n_rows, n_cols);
    const auto hilbert_dim = checked_state_count(2zu, site_count);
    require_dense_matrix_storage(hilbert_dim);

    auto out = Tensor(std::vector<usize>{hilbert_dim, hilbert_dim}, {"state_out", "state_in"});
    for (const auto bond : square_lattice_nearest_neighbor_bonds(n_rows, n_cols))
    {
        const auto local_term = dense_heisenberg_bond_operator(n_rows, n_cols, bond, coupling);
        out.array() += local_term.array();
    }
    return out;
}

auto make_heisenberg_peps_setup(HeisenbergPepsSetupConfig cfg) -> HeisenbergPepsSetup
{
    if (cfg.physical_dim != 2)
    {
        throw std::invalid_argument(
            "make_heisenberg_peps_setup currently models spin-1/2 data only."
        );
    }

    auto peps = random_peps(
        cfg.n_rows,
        cfg.n_cols,
        RandomPepsConfig{
            .physical_dim = cfg.physical_dim,
            .bond_dim = cfg.bond_dim,
            .fully_padded = cfg.fully_padded,
            .random_options = cfg.random_options,
            .seed = cfg.seed,
        }
    );
    return HeisenbergPepsSetup{
        .peps = std::move(peps),
        .two_site_operator = heisenberg_two_site_operator(cfg.coupling),
        .bonds = square_lattice_nearest_neighbor_bonds(cfg.n_rows, cfg.n_cols),
    };
}

}  // namespace ds_tn
