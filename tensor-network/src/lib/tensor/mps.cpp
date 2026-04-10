// lib/tensor/mps.cpp
#include "tensor/mps.hpp"

#include "ndarray/blas.hpp"
#include "ndarray/lapack.hpp"
#include "ndarray/ndarray.hpp"
#include "tensor/contraction.hpp"
#include "tensor/tensor.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <ranges>
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
    const auto left = [&]
    {
        if (site == 0) return std::string{"edge_left"};
        return std::format("bond_{}{}", site - 1, site);
    };
    const auto right = [&]
    {
        if (site + 1 == num_sites) return std::string{"edge_right"};
        return std::format("bond_{}{}", site, site + 1);
    };
    return {left(), std::format("physical_{}", site), right()};
}

[[nodiscard]] auto derived_seed(std::optional<TensorSeed> base_seed, usize site)
    -> std::optional<TensorSeed>
{
    if (!base_seed) return std::nullopt;
    return *base_seed + static_cast<TensorSeed>(site);
}

auto require_orthogonalizable_mps(const MPS& mps, const char* function_name) -> void
{
    namespace rv = std::ranges::views;
    for (const auto& tensor : mps.tensors())
    {
        if (tensor.validity() != TensorValidity::valid)
        {
            throw std::runtime_error(
                std::format("{} requires all tensors to be valid", function_name)
            );
        }
        if (!tensor.is_tensor3())
        {
            throw std::runtime_error(
                std::format("{} requires all tensors to be rank-3", function_name)
            );
        }
    }
    for (const auto& [left, right] : mps.tensors() | rv::pairwise | rv::reverse)
    {
        if (left.leg_name(2) != right.leg_name(0))
        {
            throw std::invalid_argument(
                std::string{function_name}
                + " requires adjacent tensors to share matching bond leg names."
            );
        }
    }
}

[[nodiscard]] auto temporary_leg_name(const Tensor& tensor) -> std::string
{
    auto candidate = std::string{"tmp"};
    while (std::ranges::contains(tensor.leg_names(), candidate))
    {
        candidate += '_';
    }
    return candidate;
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

auto MPS::left_orthogonalize() -> void
{
    {  // Expects
        require_orthogonalizable_mps(*this, "MPS::left_orthogonalize");
    }
    if (tensors_.size() <= 1)
    {
        return;
    }

    for (auto site = 0zu; site + 1 < tensors_.size(); ++site)
    {
        auto& curr = tensors_[site];
        auto& next = tensors_[site + 1];

        const auto bond_left = curr.shape(0);
        const auto d = curr.shape(1);

        const auto [q, r] = qr(curr.array().reshape({bond_left * d, curr.shape(2)}));
        curr.array() = NDArray::reshape(q, {bond_left, d, q.shape(1)});

        const auto tmp = temporary_leg_name(next);
        next = contract(Tensor{r, {tmp, curr.leg_name(2)}}, next);
        next.rename_leg(tmp, curr.leg_name(2));
    }
}

auto MPS::right_orthogonalize() -> void
{
    namespace sv = std::ranges::views;

    {  // Expects
        require_orthogonalizable_mps(*this, "MPS::right_orthogonalize");
    }
    if (tensors_.size() <= 1) return;

    for (const auto& [prev, curr] : tensors_ | sv::pairwise | sv::reverse)
    {
        const auto bond_left = curr.shape(0);
        const auto d = curr.shape(1);
        const auto bond_right = curr.shape(2);

        const auto reshaped = curr.array().reshape({bond_left, d * bond_right});
        const auto [qt, rt] = qr(reshaped, MatrixTransform::transpose);
        const auto q = transpose_matrix(qt);
        const auto r = transpose_matrix(rt);
        curr.array() = NDArray::reshape(q, {q.shape(0), d, bond_right});

        const auto tmp = temporary_leg_name(prev);
        prev = contract(prev, Tensor{r, {prev.leg_name(2), tmp}});
        prev.rename_leg(tmp, curr.leg_name(0));
    }
}

auto to_mps(const NDArray& tensor, std::optional<usize> max_bond_dim) -> MPS
{
    {  // Expects
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

        const auto reshaped = remainder.reshape({left_bond_dim * physical_dim, remainder_cols});

        const auto svd_result = svd(reshaped);
        const auto& [u, s, vt] = svd_result;
        const auto bond_dim = [&s, &max_bond_dim]
        {
            const auto full_bond_dim = s.shape(0);
            return (max_bond_dim) ? std::min(full_bond_dim, *max_bond_dim) : full_bond_dim;
        }();

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
    constexpr usize k_edge_bond_dim{1zu};

    {  // Expects
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
    }
    const auto bond_dims = [&](usize site) -> std::pair<usize, usize>
    {
        if (site == 0) return {k_edge_bond_dim, cfg.max_bond_dim};
        if (site + 1 == num_sites) return {cfg.max_bond_dim, k_edge_bond_dim};
        return {cfg.max_bond_dim, cfg.max_bond_dim};
    };
    const auto random_data = [&](usize site) -> NDArray
    {
        const auto [bond_left, bond_right] = bond_dims(site);
        return NDArray::random(
            {bond_left, cfg.physical_dim, bond_right},
            cfg.random_options,
            derived_seed(cfg.seed, site)
        );
    };
    const auto create_tensors = [&]
    {
        auto tensors = std::vector<Tensor>{};
        tensors.reserve(num_sites);
        for (auto site = 0zu; site < num_sites; ++site)
        {
            tensors.emplace_back(random_data(site), mps_leg_names(site, num_sites));
        }
        return tensors;
    };
    return MPS{create_tensors()};
}

}  // namespace ds_tn
