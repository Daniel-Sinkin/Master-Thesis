// lib/tensor/mps.cpp
#include "tensor/mps.hpp"

#include "ndarray/blas.hpp"
#include "ndarray/lapack.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ds_tn
{
namespace
{

[[nodiscard]] auto mps_leg_names(usize site, usize num_sites) -> std::array<std::string, 3>
{
    return {
        site == 0 ? "edge_left" : "bond_" + std::to_string(site - 1) + std::to_string(site),
        "physical_" + std::to_string(site),
        site + 1 == num_sites
            ? "edge_right"
            : "bond_" + std::to_string(site) + std::to_string(site + 1),
    };
}

[[nodiscard]] auto
truncated_bond_dim(const SVDResult& svd_result, std::optional<usize> max_bond_dim) -> usize
{
    return max_bond_dim.has_value()
        ? std::min(svd_result.s.shape(0), *max_bond_dim)
        : svd_result.s.shape(0);
}

[[nodiscard]] auto derived_seed(std::optional<TensorSeed> base_seed, usize site)
    -> std::optional<TensorSeed>
{
    if (!base_seed.has_value())
    {
        return std::nullopt;
    }

    return *base_seed + static_cast<TensorSeed>(site);
}

}  // namespace

MPS::MPS(std::vector<Tensor> tensors) : tensors_(std::move(tensors))
{
}

auto MPS::size() const noexcept -> usize
{
    return tensors_.size();
}

auto MPS::tensors() noexcept -> std::span<Tensor>
{
    return tensors_;
}

auto MPS::tensors() const noexcept -> std::span<const Tensor>
{
    return tensors_;
}

auto MPS::operator[](usize site) noexcept -> Tensor&
{
    return tensors_[site];
}

auto MPS::operator[](usize site) const noexcept -> const Tensor&
{
    return tensors_[site];
}

auto MPS::operator()(usize site) noexcept -> Tensor&
{
    return tensors_[site];
}

auto MPS::operator()(usize site) const noexcept -> const Tensor&
{
    return tensors_[site];
}

auto MPS::at(usize site) -> Tensor&
{
    return tensors_.at(site);
}

auto MPS::at(usize site) const -> const Tensor&
{
    return tensors_.at(site);
}

auto to_mps(const NDArray& tensor, std::optional<usize> max_bond_dim) -> MPS
{
    if (tensor.validity() != NDArrayValidity::valid)
    {
        throw std::invalid_argument("to_mps requires a valid NDArray.");
    }
    if (tensor.rank() == 0)
    {
        throw std::invalid_argument("to_mps requires a non-scalar NDArray.");
    }
    if (max_bond_dim.has_value() and *max_bond_dim == 0)
    {
        throw std::invalid_argument("to_mps requires max_bond_dim >= 1 when provided.");
    }

    const auto num_sites = tensor.rank();
    auto tensors = std::vector<Tensor>{};
    tensors.reserve(num_sites);

    auto remainder_shape = std::vector<usize>{1};
    remainder_shape.insert(remainder_shape.end(), tensor.shape().begin(), tensor.shape().end());
    auto remainder = tensor.reshape(remainder_shape);

    for (auto site = 0zu; site + 1 < num_sites; ++site)
    {
        const auto left_bond_dim = remainder.shape(0);
        const auto physical_dim = remainder.shape(1);
        const auto remainder_cols = remainder.size() / (left_bond_dim * physical_dim);

        const auto svd_result =
            svd(remainder.reshape({left_bond_dim * physical_dim, remainder_cols}));
        const auto& [u, s, vt] = svd_result;
        const auto bond_dim = truncated_bond_dim(svd_result, max_bond_dim);

        tensors.emplace_back(
            truncate_cols(u, bond_dim).reshape({left_bond_dim, physical_dim, bond_dim}),
            mps_leg_names(site, num_sites)
        );

        const auto sigma = truncate_rows(truncate_cols(s.diag(), bond_dim), bond_dim);
        const auto next_matrix = matrix_matrix_product(sigma, truncate_rows(vt, bond_dim));

        auto next_shape = std::vector<usize>{bond_dim};
        next_shape.insert(next_shape.end(), remainder.shape().begin() + 2, remainder.shape().end());
        remainder = next_matrix.reshape(next_shape);
    }

    tensors.emplace_back(
        remainder.reshape({remainder.shape(0), remainder.shape(1), 1}),
        mps_leg_names(num_sites - 1, num_sites)
    );

    return MPS{std::move(tensors)};
}

auto random_mps(usize num_sites, RandomMPSConfig cfg) -> MPS
{
    if (num_sites == 0)
    {
        throw std::invalid_argument("random_mps requires num_sites >= 1.");
    }
    if (cfg.physical_dim == 0)
    {
        throw std::invalid_argument("random_mps requires physical_dim >= 1.");
    }
    if (cfg.max_bond_dim == 0)
    {
        throw std::invalid_argument("random_mps requires max_bond_dim >= 1.");
    }

    auto tensors = std::vector<Tensor>{};
    tensors.reserve(num_sites);

    for (auto site = 0zu; site < num_sites; ++site)
    {
        const auto left_bond_dim = site == 0 ? 1zu : cfg.max_bond_dim;
        const auto right_bond_dim = site + 1 == num_sites ? 1zu : cfg.max_bond_dim;

        tensors.emplace_back(
            NDArray::random(
                {left_bond_dim, cfg.physical_dim, right_bond_dim},
                cfg.random_options,
                derived_seed(cfg.seed, site)
            ),
            mps_leg_names(site, num_sites)
        );
    }

    return MPS{std::move(tensors)};
}

}  // namespace ds_tn
