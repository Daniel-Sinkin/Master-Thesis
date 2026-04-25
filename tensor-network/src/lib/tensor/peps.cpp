// lib/tensor/peps.cpp
#include "tensor/peps.hpp"

#include <array>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace ds_tn
{
namespace
{

[[nodiscard]] auto node_shape(const Peps::Config& cfg, usize row, usize col) -> std::vector<usize>
{
    auto shape = std::vector<usize>{
        cfg.bond_dim, cfg.bond_dim, cfg.bond_dim, cfg.bond_dim, cfg.physical_dim
    };

    if (row == 0zu)
    {
        shape[Peps::k_leg_top] = Peps::k_dummy_dim;
    }
    if (row + 1 == cfg.n_rows)
    {
        shape[Peps::k_leg_bottom] = Peps::k_dummy_dim;
    }
    if (col == 0zu)
    {
        shape[Peps::k_leg_left] = Peps::k_dummy_dim;
    }
    if (col + 1 == cfg.n_cols)
    {
        shape[Peps::k_leg_right] = Peps::k_dummy_dim;
    }

    return shape;
}

[[nodiscard]] auto node_leg_names(usize row, usize col) -> std::array<std::string, 5>
{
    const auto format_leg = [row, col](std::string_view symbol)
    {
        return std::format("{}{},{}", symbol, row, col);
    };
    return {
        format_leg("r"), format_leg("t"), format_leg("l"), format_leg("b"), format_leg("p")
    };
}

[[nodiscard]] auto derived_seed(std::optional<TensorSeed> base_seed, usize linear_index)
    -> std::optional<TensorSeed>
{
    if (!base_seed)
    {
        return std::nullopt;
    }
    return *base_seed + static_cast<TensorSeed>(linear_index);
}

auto require_valid_config(const Peps::Config& cfg, const char* function_name) -> void
{
    const auto validity = cfg.check_validity();
    if (validity != Peps::ConfigValidity::valid)
    {
        throw std::invalid_argument(
            std::format(
                "{} requires a valid Peps::Config (got {}, validity={}).",
                function_name,
                cfg.to_string(),
                Peps::to_string(validity)
            )
        );
    }
}

}  // namespace

auto Peps::Config::check_validity() const -> ConfigValidity
{
    if (n_rows < 3)
    {
        return ConfigValidity::too_few_rows;
    }
    if (n_cols < 3)
    {
        return ConfigValidity::too_few_cols;
    }
    if (bond_dim < 2)
    {
        return ConfigValidity::bond_dim_too_small;
    }
    if (physical_dim < 2)
    {
        return ConfigValidity::physical_dim_too_small;
    }
    return ConfigValidity::valid;
}

auto Peps::Config::to_string() const -> std::string
{
    return std::format(
        "Peps::Config(n_rows={},n_cols={},bond_dim={},physical_dim={})",
        n_rows,
        n_cols,
        bond_dim,
        physical_dim
    );
}

Peps::Peps(Config cfg) : cfg_(std::move(cfg))
{
    {  // Expects
        require_valid_config(cfg_, "Peps::Peps");
    }

    nodes_.reserve(cfg_.n_rows * cfg_.n_cols);
    for (auto row = 0zu; row < cfg_.n_rows; ++row)
    {
        for (auto col = 0zu; col < cfg_.n_cols; ++col)
        {
            nodes_.emplace_back(node_shape(cfg_, row, col), node_leg_names(row, col));
        }
    }
}

auto Peps::to_string(ConfigValidity validity) -> std::string_view
{
    switch (validity)
    {
        case ConfigValidity::valid:
            return "valid";
        case ConfigValidity::too_few_rows:
            return "too_few_rows";
        case ConfigValidity::too_few_cols:
            return "too_few_cols";
        case ConfigValidity::bond_dim_too_small:
            return "bond_dim_too_small";
        case ConfigValidity::physical_dim_too_small:
            return "physical_dim_too_small";
    }
    std::unreachable();
}

auto Peps::config() const noexcept -> const Config&
{
    return cfg_;
}

auto Peps::n_rows() const noexcept -> usize
{
    return cfg_.n_rows;
}

auto Peps::n_cols() const noexcept -> usize
{
    return cfg_.n_cols;
}

auto Peps::bond_dim() const noexcept -> usize
{
    return cfg_.bond_dim;
}

auto Peps::physical_dim() const noexcept -> usize
{
    return cfg_.physical_dim;
}

auto Peps::size() const noexcept -> usize
{
    return nodes_.size();
}

auto Peps::tensors() noexcept -> std::span<Tensor>
{
    return nodes_;
}

auto Peps::tensors() const noexcept -> std::span<const Tensor>
{
    return nodes_;
}

auto Peps::operator()(usize row, usize col) -> Tensor&
{
    return at(row, col);
}

auto Peps::operator()(usize row, usize col) const -> const Tensor&
{
    return at(row, col);
}

auto Peps::at(usize row, usize col) -> Tensor&
{
    return nodes_.at(storage_index(row, col));
}

auto Peps::at(usize row, usize col) const -> const Tensor&
{
    return nodes_.at(storage_index(row, col));
}

auto Peps::total_entries() const noexcept -> usize
{
    return std::transform_reduce(
        nodes_.begin(),
        nodes_.end(),
        usize{0},
        std::plus<>{},
        [](const Tensor& tensor) { return tensor.size(); }
    );
}

auto Peps::print_metadata(MetadataConfig cfg) const -> void
{
    const auto inner = std::format(
        "n_rows={},n_cols={},bond_dim={},physical_dim={}",
        n_rows(),
        n_cols(),
        bond_dim(),
        physical_dim()
    );
    if (cfg.include_classname)
    {
        std::println("Peps({})", inner);
    }
    else
    {
        std::println("{}", inner);
    }

    if (cfg.include_memory)
    {
        const auto memory_bytes = total_entries() * sizeof(Scalar);
        std::println(
            "total_entries={}, memory footprint={}",
            total_entries(),
            format_bytes(memory_bytes, cfg.memory_digits)
        );
    }
}

auto Peps::print_metadata() const -> void
{
    print_metadata(MetadataConfig{});
}

auto Peps::is_valid(ConfigValidity validity) -> bool
{
    return validity == ConfigValidity::valid;
}

auto Peps::storage_index(usize row, usize col) const -> usize
{
    if (row >= n_rows() or col >= n_cols())
    {
        throw std::out_of_range(
            std::format(
                "Peps index ({}, {}) is out of bounds for a {}x{} grid.",
                row,
                col,
                n_rows(),
                n_cols()
            )
        );
    }

    return row * n_cols() + col;
}

auto random_peps(usize n_rows, usize n_cols, RandomPepsConfig cfg) -> Peps
{
    const auto peps_cfg = Peps::Config{
        .n_rows = n_rows,
        .n_cols = n_cols,
        .bond_dim = cfg.bond_dim,
        .physical_dim = cfg.physical_dim,
    };

    {  // Expects
        require_valid_config(peps_cfg, "random_peps");
    }
    if (cfg.apply_algebraic_power_law_suppression)
    {
        throw std::logic_error(
            "random_peps algebraic power law suppression is not implemented yet."
        );
    }

    auto peps = Peps{peps_cfg};
    for (auto row = 0zu; row < peps.n_rows(); ++row)
    {
        for (auto col = 0zu; col < peps.n_cols(); ++col)
        {
            const auto linear_index = row * peps.n_cols() + col;
            peps(row, col) = Tensor{
                NDArray::random(
                    node_shape(peps_cfg, row, col),
                    cfg.random_options,
                    derived_seed(cfg.seed, linear_index)
                ),
                node_leg_names(row, col)
            };
        }
    }

    return peps;
}

}  // namespace ds_tn
