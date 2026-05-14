// app/main.cpp
#include "models/heisenberg.hpp"
#include "tensor/contraction.hpp"
#include "tensor/sampling.hpp"

#include <algorithm>
#include <format>
#include <iterator>
#include <optional>
#include <print>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

constexpr auto k_rows = ds_tn::usize{3};
constexpr auto k_cols = ds_tn::usize{3};
constexpr auto k_physical_dim = ds_tn::usize{2};
constexpr auto k_bond_dim = ds_tn::usize{2};
constexpr auto k_seed = ds_tn::NDArraySeed{7};
constexpr auto k_demo_state_count = ds_tn::usize{8};

[[nodiscard]] auto basis_cap(ds_tn::usize index, ds_tn::usize extent, std::string leg_name)
    -> ds_tn::Tensor
{
    auto values = ds_tn::NDArray{{extent}};
    values(index) = 1.0;
    return ds_tn::Tensor{std::move(values), {std::move(leg_name)}};
}

auto cap_leg_to_zero(ds_tn::Tensor& tensor, std::string leg_name) -> void
{
    const auto legs = tensor.leg_names();
    const auto it = std::ranges::find(legs, leg_name);
    if (it == legs.end())
    {
        throw std::invalid_argument("cap_leg_to_zero requires leg_name to exist.");
    }

    const auto axis = static_cast<ds_tn::usize>(std::distance(legs.begin(), it));
    tensor = ds_tn::contract(tensor, basis_cap(0zu, tensor.shape(axis), std::move(leg_name)));
}

auto rename_if_present(ds_tn::Tensor& tensor, ds_tn::usize axis, std::string new_name) -> void
{
    tensor.rename_leg(tensor.leg_name(axis), std::move(new_name));
}

[[nodiscard]] auto
contraction_ready_site(const ds_tn::Peps& peps, ds_tn::usize row, ds_tn::usize col) -> ds_tn::Tensor
{
    auto site = peps(row, col);

    if (col + 1zu < peps.n_cols())
    {
        rename_if_present(site, ds_tn::Peps::k_leg_right, std::format("h{},{}", row, col));
    }
    if (col > 0zu)
    {
        rename_if_present(site, ds_tn::Peps::k_leg_left, std::format("h{},{}", row, col - 1zu));
    }
    if (row + 1zu < peps.n_rows())
    {
        rename_if_present(site, ds_tn::Peps::k_leg_bottom, std::format("v{},{}", row, col));
    }
    if (row > 0zu)
    {
        rename_if_present(site, ds_tn::Peps::k_leg_top, std::format("v{},{}", row - 1zu, col));
    }

    return site;
}

[[nodiscard]] auto
projected_site(const ds_tn::Peps& peps, ds_tn::usize row, ds_tn::usize col, ds_tn::usize spin)
    -> ds_tn::Tensor
{
    auto site = contraction_ready_site(peps, row, col);

    const auto right_leg = site.leg_name(ds_tn::Peps::k_leg_right);
    const auto top_leg = site.leg_name(ds_tn::Peps::k_leg_top);
    const auto left_leg = site.leg_name(ds_tn::Peps::k_leg_left);
    const auto bottom_leg = site.leg_name(ds_tn::Peps::k_leg_bottom);
    const auto physical_leg = site.leg_name(ds_tn::Peps::k_leg_physical);

    site = ds_tn::contract(site, basis_cap(spin, peps.physical_dim(), physical_leg));

    if (row == 0zu)
    {
        cap_leg_to_zero(site, top_leg);
    }
    if (row + 1zu == peps.n_rows())
    {
        cap_leg_to_zero(site, bottom_leg);
    }
    if (col == 0zu)
    {
        cap_leg_to_zero(site, left_leg);
    }
    if (col + 1zu == peps.n_cols())
    {
        cap_leg_to_zero(site, right_leg);
    }

    return site;
}

[[nodiscard]] auto
projected_peps_layer(const ds_tn::Peps& peps, std::span<const ds_tn::usize> spins)
    -> std::vector<ds_tn::Tensor>
{
    auto out = std::vector<ds_tn::Tensor>{};
    out.reserve(peps.size());
    for (auto row = 0zu; row < peps.n_rows(); ++row)
    {
        for (auto col = 0zu; col < peps.n_cols(); ++col)
        {
            out.push_back(projected_site(peps, row, col, spins[row * peps.n_cols() + col]));
        }
    }
    return out;
}

[[nodiscard]] auto exact_contract_projected_layer(std::span<const ds_tn::Tensor> projected)
    -> double
{
    auto network = std::optional<ds_tn::Tensor>{};
    for (const auto& site : projected)
    {
        network = network.has_value() ? ds_tn::contract(*network, site) : site;
    }

    if (!network.has_value() or network->size() != 1zu)
    {
        throw std::runtime_error("Projected PEPS layer did not contract to one scalar.");
    }
    return network->data()[0];
}

}  // namespace

int main()
{
    using namespace ds_tn;

    const auto setup = make_heisenberg_peps_setup({
        .n_rows = k_rows,
        .n_cols = k_cols,
        .physical_dim = k_physical_dim,
        .bond_dim = k_bond_dim,
        .fully_padded = true,
        .coupling = 1.0,
        .seed = k_seed,
    });
    const auto& peps = setup.peps;

    std::println("Per-basis PEPS capping demo");
    peps.print_metadata({.include_memory = true});
    peps(0, 0).print_metadata("raw_site(0,0)");

    for (auto encoded = 0zu; encoded < k_demo_state_count; ++encoded)
    {
        const auto spins = decode_base(encoded, k_rows * k_cols, k_physical_dim);
        const auto projected = projected_peps_layer(peps, spins);
        const auto amplitude = exact_contract_projected_layer(projected);

        std::println("\nS_{} = |{}>", encoded, spin_configuration_to_string(spins));
        projected.front().print_metadata("projected_site(0,0)");
        std::println("Psi(S_{}) from capped exact contraction = {:.8e}", encoded, amplitude);
        std::println(
            "Psi(S_{}) from peps_amplitude helper     = {:.8e}",
            encoded,
            peps_amplitude(peps, spins)
        );
    }
}
