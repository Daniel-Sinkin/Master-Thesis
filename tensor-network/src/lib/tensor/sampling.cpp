// lib/tensor/sampling.cpp
#include "tensor/sampling.hpp"

#include "tensor/contraction.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace ds_tn
{
namespace
{

[[nodiscard]] auto checked_state_count(usize base, usize length) -> usize
{
    if (base == 0)
    {
        throw std::invalid_argument("checked_state_count requires base >= 1.");
    }

    auto out = usize{1};
    for (auto i = 0zu; i < length; ++i)
    {
        if (out > std::numeric_limits<usize>::max() / base)
        {
            throw std::overflow_error("State space size overflows usize.");
        }
        out *= base;
    }
    return out;
}

auto require_valid_spin_configuration(const Peps& peps, std::span<const usize> spins) -> void
{
    const auto expected_size = peps.n_rows() * peps.n_cols();
    if (spins.size() != expected_size)
    {
        throw std::invalid_argument(
            std::format(
                "PEPS spin configuration has size {}, expected {}.", spins.size(), expected_size
            )
        );
    }

    for (const auto spin : spins)
    {
        if (spin >= peps.physical_dim())
        {
            throw std::invalid_argument("Spin value exceeds PEPS physical dimension.");
        }
    }
}

[[nodiscard]] auto basis_projector(usize index, usize extent, std::string leg_name) -> Tensor
{
    if (index >= extent)
    {
        throw std::invalid_argument("basis_projector index exceeds extent.");
    }

    auto values = NDArray({extent});
    values(index) = 1.0;
    return Tensor{std::move(values), std::vector<std::string>{std::move(leg_name)}};
}

[[nodiscard]] auto physical_projector(usize spin, usize physical_dim, std::string leg_name)
    -> Tensor
{
    return basis_projector(spin, physical_dim, std::move(leg_name));
}

auto cap_leg_to_zero(Tensor& site, std::string leg_name) -> void
{
    const auto legs = site.leg_names();
    const auto it = std::ranges::find(legs, leg_name);
    if (it == legs.end())
    {
        throw std::invalid_argument("cap_leg_to_zero requires leg_name to exist.");
    }

    const auto axis = static_cast<usize>(std::distance(legs.begin(), it));
    site = contract(site, basis_projector(0zu, site.shape(axis), std::move(leg_name)));
}

auto rename_if_present(Tensor& tensor, usize axis, std::string new_name) -> void
{
    tensor.rename_leg(tensor.leg_name(axis), std::move(new_name));
}

[[nodiscard]] auto contraction_ready_site(const Peps& peps, usize row, usize col) -> Tensor
{
    auto site = peps(row, col);

    if (col + 1 < peps.n_cols())
    {
        rename_if_present(site, Peps::k_leg_right, std::format("h{},{}", row, col));
    }
    if (col > 0)
    {
        rename_if_present(site, Peps::k_leg_left, std::format("h{},{}", row, col - 1zu));
    }
    if (row + 1 < peps.n_rows())
    {
        rename_if_present(site, Peps::k_leg_bottom, std::format("v{},{}", row, col));
    }
    if (row > 0)
    {
        rename_if_present(site, Peps::k_leg_top, std::format("v{},{}", row - 1zu, col));
    }

    return site;
}

[[nodiscard]] auto projected_site(const Peps& peps, usize row, usize col, usize spin) -> Tensor
{
    auto site = contraction_ready_site(peps, row, col);
    const auto right_leg = site.leg_name(Peps::k_leg_right);
    const auto top_leg = site.leg_name(Peps::k_leg_top);
    const auto left_leg = site.leg_name(Peps::k_leg_left);
    const auto bottom_leg = site.leg_name(Peps::k_leg_bottom);
    site = contract(
        site, physical_projector(spin, peps.physical_dim(), site.leg_name(Peps::k_leg_physical))
    );
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

[[nodiscard]] auto prefix_matches(std::span<const usize> spins, std::span<const usize> prefix)
    -> bool
{
    if (prefix.size() > spins.size())
    {
        return false;
    }
    return std::ranges::equal(prefix, spins.first(prefix.size()));
}

[[nodiscard]] auto row_slice(std::span<const usize> spins, usize row, usize n_cols)
    -> std::span<const usize>
{
    return spins.subspan(row * n_cols, n_cols);
}

[[nodiscard]] auto row_draw(std::span<const RowProbability> options, std::mt19937_64& rng) -> usize
{
    auto distribution = std::uniform_real_distribution<f64>{0.0, 1.0};
    const auto draw = distribution(rng);

    auto cumulative = f64{0.0};
    for (auto option_index = 0zu; option_index < options.size(); ++option_index)
    {
        const auto& option = options[option_index];
        cumulative += option.probability;
        if (draw <= cumulative)
        {
            return option_index;
        }
    }

    for (auto option_index = options.size(); option_index > 0; --option_index)
    {
        if (options[option_index - 1zu].probability > 0.0)
        {
            return option_index - 1zu;
        }
    }
    throw std::runtime_error("Cannot sample from an all-zero row distribution.");
}

}  // namespace

auto encode_base(std::span<const usize> digits, usize base) -> usize
{
    if (base == 0)
    {
        throw std::invalid_argument("encode_base requires base >= 1.");
    }

    auto out = usize{0};
    for (const auto digit : digits)
    {
        if (digit >= base)
        {
            throw std::invalid_argument("encode_base digit exceeds base.");
        }
        if (out > (std::numeric_limits<usize>::max() - digit) / base)
        {
            throw std::overflow_error("encode_base overflows usize.");
        }
        out = out * base + digit;
    }
    return out;
}

auto decode_base(usize encoded, usize length, usize base) -> SpinConfiguration
{
    if (base == 0)
    {
        throw std::invalid_argument("decode_base requires base >= 1.");
    }

    auto out = SpinConfiguration(length, 0);
    for (auto i = length; i > 0; --i)
    {
        out[i - 1zu] = encoded % base;
        encoded /= base;
    }
    if (encoded != 0)
    {
        throw std::invalid_argument("decode_base encoded value needs more digits.");
    }
    return out;
}

auto spin_configuration_to_string(std::span<const usize> spins) -> std::string
{
    auto out = std::string{};
    out.reserve(spins.size());
    for (const auto spin : spins)
    {
        if (spin < 10)
        {
            out.push_back(static_cast<char>('0' + spin));
        }
        else
        {
            out += std::format("[{}]", spin);
        }
    }
    return out;
}

auto peps_amplitude(const Peps& peps, std::span<const usize> spins) -> f64
{
    require_valid_spin_configuration(peps, spins);

    auto network = std::optional<Tensor>{};
    for (auto row = 0zu; row < peps.n_rows(); ++row)
    {
        for (auto col = 0zu; col < peps.n_cols(); ++col)
        {
            const auto spin = spins[row * peps.n_cols() + col];
            auto site = projected_site(peps, row, col, spin);
            network = network.has_value() ? contract(*network, site) : std::move(site);
        }
    }

    if (!network.has_value() or network->size() != 1)
    {
        throw std::runtime_error("Projected PEPS contraction did not reduce to one scalar value.");
    }
    return network->data()[0];
}

auto exact_peps_distribution(const Peps& peps) -> ExactPepsDistribution
{
    const auto site_count = peps.n_rows() * peps.n_cols();
    const auto state_count = checked_state_count(peps.physical_dim(), site_count);

    auto states = std::vector<ConfigurationProbability>{};
    states.reserve(state_count);

    auto norm_squared = f64{0.0};
    for (auto encoded = 0zu; encoded < state_count; ++encoded)
    {
        auto spins = decode_base(encoded, site_count, peps.physical_dim());
        const auto amplitude = peps_amplitude(peps, spins);
        const auto weight = amplitude * amplitude;
        norm_squared += weight;
        states.push_back({
            .encoded = encoded,
            .spins = std::move(spins),
            .amplitude = amplitude,
            .weight = weight,
        });
    }

    if (not std::isfinite(norm_squared) or norm_squared <= 0.0)
    {
        throw std::runtime_error("Exact PEPS distribution has zero or non-finite norm.");
    }

    for (auto& state : states)
    {
        state.probability = state.weight / norm_squared;
    }

    return ExactPepsDistribution{
        .n_rows = peps.n_rows(),
        .n_cols = peps.n_cols(),
        .physical_dim = peps.physical_dim(),
        .norm_squared = norm_squared,
        .states = std::move(states),
    };
}

auto conditional_row_probabilities(
    const ExactPepsDistribution& distribution, usize row, std::span<const usize> prefix
) -> std::vector<RowProbability>
{
    if (row >= distribution.n_rows)
    {
        throw std::out_of_range("conditional_row_probabilities row exceeds lattice height.");
    }
    if (prefix.size() != row * distribution.n_cols)
    {
        throw std::invalid_argument(
            "conditional_row_probabilities prefix must contain all previous rows."
        );
    }

    const auto row_state_count =
        checked_state_count(distribution.physical_dim, distribution.n_cols);
    auto row_weights = std::vector<f64>(row_state_count, 0.0);
    auto prefix_weight = f64{0.0};

    for (const auto& state : distribution.states)
    {
        if (!prefix_matches(state.spins, prefix))
        {
            continue;
        }

        prefix_weight += state.probability;
        const auto encoded_row = encode_base(
            row_slice(state.spins, row, distribution.n_cols), distribution.physical_dim
        );
        row_weights[encoded_row] += state.probability;
    }

    if (not std::isfinite(prefix_weight) or prefix_weight <= 0.0)
    {
        throw std::runtime_error("Row prefix has zero probability mass.");
    }

    auto out = std::vector<RowProbability>{};
    out.reserve(row_state_count);
    for (auto encoded = 0zu; encoded < row_state_count; ++encoded)
    {
        out.push_back({
            .encoded = encoded,
            .spins = decode_base(encoded, distribution.n_cols, distribution.physical_dim),
            .weight = row_weights[encoded],
            .probability = row_weights[encoded] / prefix_weight,
        });
    }
    return out;
}

auto sample_direct_exact(const ExactPepsDistribution& distribution, std::mt19937_64& rng)
    -> DirectSample
{
    auto out = DirectSample{
        .probability = 1.0,
        .log_probability = 0.0,
    };
    out.spins.reserve(distribution.n_rows * distribution.n_cols);
    out.steps.reserve(distribution.n_rows);

    for (auto row = 0zu; row < distribution.n_rows; ++row)
    {
        auto options = conditional_row_probabilities(distribution, row, out.spins);
        const auto selected_option = row_draw(options, rng);
        const auto selected_row = options[selected_option].spins;
        const auto selected_probability = options[selected_option].probability;

        auto step = DirectSamplingStep{
            .row = row,
            .prefix_before = out.spins,
            .options = std::move(options),
            .selected_row = selected_row,
            .selected_probability = selected_probability,
        };

        out.spins.insert(out.spins.end(), step.selected_row.begin(), step.selected_row.end());
        out.probability *= step.selected_probability;
        out.log_probability += std::log(step.selected_probability);
        out.steps.push_back(std::move(step));
    }

    return out;
}

auto sample_direct_exact(const Peps& peps, std::mt19937_64& rng) -> DirectSample
{
    return sample_direct_exact(exact_peps_distribution(peps), rng);
}

auto sample_direct_exact(const Peps& peps, ExactDirectSamplingConfig cfg) -> DirectSample
{
    auto rng = cfg.seed.has_value() ? std::mt19937_64{*cfg.seed} : std::mt19937_64{};
    return sample_direct_exact(peps, rng);
}

}  // namespace ds_tn
