// lib/tensor/mps.hpp
#pragma once

#include "tensor/tensor.hpp"

#include <optional>
#include <span>
#include <vector>

namespace ds_tn
{

class MPS
{
  public:
    MPS() = default;
    explicit MPS(std::vector<Tensor> tensors);

    [[nodiscard]] auto size() const noexcept -> usize;
    [[nodiscard]] auto tensors() noexcept -> std::span<Tensor>;
    [[nodiscard]] auto tensors() const noexcept -> std::span<const Tensor>;
    [[nodiscard]] auto operator[](usize site) noexcept -> Tensor&;
    [[nodiscard]] auto operator[](usize site) const noexcept -> const Tensor&;
    [[nodiscard]] auto operator()(usize site) noexcept -> Tensor&;
    [[nodiscard]] auto operator()(usize site) const noexcept -> const Tensor&;
    [[nodiscard]] auto at(usize site) -> Tensor&;
    [[nodiscard]] auto at(usize site) const -> const Tensor&;

  private:
    std::vector<Tensor> tensors_{};
};

struct RandomMPSConfig
{
    usize physical_dim{2};
    usize max_bond_dim{2};

    RandomOptions random_options{RandomNormalOptions{.mu = 0.0, .sigma = 0.1}};
    std::optional<TensorSeed> seed{};
};

[[nodiscard]] auto to_mps(const NDArray& tensor, std::optional<usize> max_bond_dim = std::nullopt)
    -> MPS;
[[nodiscard]] auto random_mps(usize num_sites, RandomMPSConfig cfg = {}) -> MPS;

}  // namespace ds_tn
