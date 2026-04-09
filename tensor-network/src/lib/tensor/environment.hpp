// lib/tensor/environment.hpp
#pragma once

#include "tensor/mps.hpp"

#include <span>
#include <vector>

namespace ds_tn
{

[[nodiscard]] auto right_boundary_environment(const Tensor& mps_tensor, const Tensor& mpo_tensor)
    -> Tensor;
[[nodiscard]] auto
update_right_environment(const Tensor& right_environment, const Tensor& mps_tensor, const Tensor& mpo_tensor)
    -> Tensor;
[[nodiscard]] auto right_environments(const MPS& mps, std::span<const Tensor> mpo)
    -> std::vector<Tensor>;

}  // namespace ds_tn
