// lib/models/heisenberg.hpp
#pragma once

#include "tensor/peps.hpp"
#include "tensor/tensor.hpp"

#include <optional>
#include <vector>

namespace ds_tn
{

struct LatticeSite
{
    usize row{};
    usize col{};

    [[nodiscard]] constexpr auto linear_index(usize n_cols) const noexcept -> usize
    {
        return row * n_cols + col;
    }
};

struct NearestNeighborBond
{
    LatticeSite first{};
    LatticeSite second{};
};

struct HeisenbergPepsSetupConfig
{
    usize n_rows{2};
    usize n_cols{2};
    usize physical_dim{2};
    usize bond_dim{2};
    bool fully_padded{false};
    f64 coupling{1.0};
    RandomOptions random_options{RandomNormalOptions{.mu = 0.0, .sigma = 0.1}};
    std::optional<TensorSeed> seed{7};
};

struct HeisenbergPepsSetup
{
    Peps peps;
    Tensor two_site_operator;
    std::vector<NearestNeighborBond> bonds;
};

[[nodiscard]] auto heisenberg_two_site_operator(f64 coupling = 1.0) -> Tensor;
[[nodiscard]] auto square_lattice_nearest_neighbor_bonds(usize n_rows, usize n_cols)
    -> std::vector<NearestNeighborBond>;
[[nodiscard]] auto dense_heisenberg_bond_operator(
    usize n_rows, usize n_cols, NearestNeighborBond bond, f64 coupling = 1.0
) -> Tensor;
[[nodiscard]] auto dense_heisenberg_hamiltonian(usize n_rows, usize n_cols, f64 coupling = 1.0)
    -> Tensor;
[[nodiscard]] auto make_heisenberg_peps_setup(HeisenbergPepsSetupConfig cfg = {})
    -> HeisenbergPepsSetup;

}  // namespace ds_tn
